import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random

from load_dataset import load_dataset
from utils import get_device, clone_state_dict_to_cpu


def safe_roc_auc(y_true, y_score):
    y_true_np = y_true.detach().cpu().numpy().astype(np.int64)
    y_score_np = y_score.detach().cpu().numpy().astype(np.float64)
    if len(np.unique(y_true_np)) < 2:
        return 0.5
    return float(roc_auc_score(y_true_np, y_score_np))


class GraphSAGE_Manual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_neighbors=5):
        super().__init__()
        self.conv1 = nn.Linear(in_channels * 2, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.num_neighbors = num_neighbors
        self.adj_lists = None

    def set_adj_lists(self, adj_lists):
        self.adj_lists = adj_lists

    def forward(self, x, batch_nodes):
        # Model device
        dev = self.conv1.weight.device

        # batch_nodes may be a tensor or a Python list
        if isinstance(batch_nodes, torch.Tensor):
            batch_nodes = batch_nodes.detach().cpu().tolist()
        else:
            batch_nodes = [int(n) for n in batch_nodes]

        # Sample neighbors on CPU using adjacency lists
        sampled_neighbors = []
        for node in batch_nodes:
            neigh = self.adj_lists[node]
            if len(neigh) > self.num_neighbors:
                neigh = random.sample(neigh, self.num_neighbors)
            sampled_neighbors.append(neigh)

        # Collect all unique nodes touched by this batch
        all_nodes = set(batch_nodes)
        for neighs in sampled_neighbors:
            all_nodes.update(neighs)
        all_nodes = list(all_nodes)
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}

        # Keep full x on CPU, move only the needed subset to model device
        x_sub = x[all_nodes].to(dev, non_blocking=True)

        # Compute embeddings for batch nodes
        batch_embs = []
        for i, node in enumerate(batch_nodes):
            self_feat = x_sub[node_to_idx[node]]
            neigh_feat = [x_sub[node_to_idx[n]] for n in sampled_neighbors[i]]

            if neigh_feat:
                neigh_mean = torch.stack(neigh_feat).mean(dim=0)
            else:
                neigh_mean = torch.zeros_like(self_feat, device=dev)

            combined = torch.cat([self_feat, neigh_mean], dim=0)
            h = F.relu(self.conv1(combined))
            batch_embs.append(h)

        out = self.fc(torch.stack(batch_embs, dim=0)).squeeze(-1)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ds = load_dataset("B", args.data_dir)
    data = ds[0]

    # Keep features on CPU to avoid moving the full graph to GPU
    x = data.x.float()
    if x.is_sparse:
        x = x.to_dense()
    edge_index = data.edge_index.long()

    y_full = data.y.long()               # labels only for labeled_nodes
    labeled_nodes = data.labeled_nodes.long()
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()

    # Build label lookup by original node id
    num_total_nodes = x.size(0)
    y_lookup = torch.full((num_total_nodes,), -1, dtype=torch.long)
    for i, node in enumerate(labeled_nodes.cpu().numpy()):
        y_lookup[node] = y_full[i].item()
    y_lookup = y_lookup.to(device)

    # Build adjacency lists
    print("Building adjacency lists...")
    adj_lists = [[] for _ in range(num_total_nodes)]
    for u, v in edge_index.t().cpu().numpy():
        adj_lists[u].append(v)
        adj_lists[v].append(u)
    for i in range(num_total_nodes):
        adj_lists[i] = list(set(adj_lists[i]) - {i})

    # Train/val node ids as Python ints
    train_nodes = labeled_nodes[train_mask].cpu().tolist()
    val_nodes = labeled_nodes[val_mask].cpu().tolist()

    # Class imbalance weight
    train_y = y_full[train_mask]
    pos_count = train_y.sum().item()
    neg_count = train_y.numel() - pos_count
    pos_weight = torch.tensor(neg_count / max(pos_count, 1), device=device)
    print(f"pos_weight = {pos_weight.item():.2f}")

    model = GraphSAGE_Manual(
        in_channels=x.size(1),
        hidden_channels=args.hidden_dim,
        out_channels=1,
        num_neighbors=args.num_neighbors,
    )
    model.set_adj_lists(adj_lists)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_auc = -1.0
    best_state = None
    patience = 10
    bad_epochs = 0

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        shuffled = train_nodes.copy()
        random.shuffle(shuffled)

        pbar = tqdm(range(0, len(shuffled), args.batch_size), desc=f"Epoch {epoch:02d} [Train]")
        for start in pbar:
            batch_nodes = shuffled[start:start + args.batch_size]
            if not batch_nodes:
                continue

            optimizer.zero_grad()

            logits = model(x, batch_nodes)
            labels = y_lookup[batch_nodes].float()  # already on device

            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(num_batches, 1)

        # Validation
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for start in range(0, len(val_nodes), args.batch_size):
                batch_nodes = val_nodes[start:start + args.batch_size]
                if not batch_nodes:
                    continue

                logits = model(x, batch_nodes)
                scores = torch.sigmoid(logits)
                labels = y_lookup[batch_nodes].float()

                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        val_scores = torch.cat(all_scores)
        val_labels = torch.cat(all_labels)
        val_auc = safe_roc_auc(val_labels, val_scores)

        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = clone_state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(
        {
            "model_name": "GraphSAGE_Manual",
            "model_kwargs": {
                "in_channels": x.size(1),
                "hidden_channels": args.hidden_dim,
                "out_channels": 1,
                "num_neighbors": args.num_neighbors,
            },
            "state_dict": best_state,
            "best_val_auc": best_val_auc,
        },
        os.path.join(args.model_dir, f"{args.kerberos}_model_B.pt"),
    )
    print(f"\nBest validation AUC = {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
