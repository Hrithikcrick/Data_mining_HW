import argparse
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from load_dataset import load_dataset
from utils import set_seed, clone_state_dict_to_cpu, get_device


def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--kerberos", type=str, required=True)
    parser.add_argument("--mode", type=str, default="full", choices=["search", "final", "full"])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--final_epochs", type=int, default=600)
    return parser.parse_args()


class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3,
                 heads=4, dropout=0.3, edge_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        if in_channels != hidden_channels:
            self.input_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.input_proj = nn.Identity()

        for i in range(num_layers):
            in_dim = hidden_channels if i > 0 else hidden_channels
            out_dim = hidden_channels
            conv = GATConv(in_dim, out_dim, heads=heads, dropout=dropout,
                           concat=False, edge_dim=edge_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(out_dim))
            if i > 0 and hidden_channels == out_dim:
                self.skips.append(nn.Identity())
            else:
                self.skips.append(None)

        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_proj(x)
        for i, conv in enumerate(self.convs):
            x_in = x
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            if self.skips[i] is not None and x_in.shape == x.shape:
                x = x + x_in
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)
        return x


def predict_link(embeddings, edge_index):
    if edge_index.shape[0] == 2:
        src, dst = edge_index[0], edge_index[1]
    else:
        src, dst = edge_index[:, 0], edge_index[:, 1]
    return (embeddings[src] * embeddings[dst]).sum(dim=1)


def sample_negatives(num_nodes, num_pos_edges, num_neg_per_pos, device):
    total_neg = num_pos_edges * num_neg_per_pos
    src = torch.randint(0, num_nodes, (total_neg,), device=device)
    dst = torch.randint(0, num_nodes, (total_neg,), device=device)
    return torch.stack([src, dst], dim=1)


def margin_ranking_loss(pos_scores, neg_scores, margin=0.5):
    K = neg_scores.size(1)
    pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, K)
    return F.margin_ranking_loss(pos_scores_expanded, neg_scores,
                                 torch.ones_like(pos_scores_expanded), margin=margin)


def train_search_once(
    cfg,
    seed,
    x,
    edge_index,
    train_pos,
    valid_pos,
    valid_neg,
    num_nodes,
    device,
    epochs,
    patience,
    neg_per_pos=10,
):
    set_seed(seed)
    model = GATLinkPredictor(**cfg["model_kwargs"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_h50 = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    train_pos = train_pos.to(device)
    valid_pos = valid_pos.to(device)
    valid_neg = valid_neg.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        neg_edges = sample_negatives(num_nodes, train_pos.size(0), neg_per_pos, device)
        emb = model(x, edge_index)
        pos_scores = predict_link(emb, train_pos)
        neg_scores = predict_link(emb, neg_edges).view(-1, neg_per_pos)
        loss = margin_ranking_loss(pos_scores, neg_scores, margin=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            emb_val = model(x, edge_index)
            pos_val = predict_link(emb_val, valid_pos)
            neg_val = predict_link(emb_val, valid_neg.view(-1, 2)).view(valid_neg.size(0), -1)
            val_h50 = hits_at_k(pos_val, neg_val, k=50)

        if val_h50 > best_val_h50:
            best_val_h50 = val_h50
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"[search] seed={seed} epoch={epoch:04d} loss={loss.item():.4f} val_hits50={val_h50:.4f}")

        if bad_epochs >= patience:
            break

    return {
        "seed": seed,
        "model_name": cfg["model_name"],
        "model_kwargs": deepcopy(cfg["model_kwargs"]),
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "best_val_h50": best_val_h50,
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }


def build_search_space(in_channels: int):
    return [
        {
            "model_name": "GATLinkPredictor",
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": 256,
                "out_channels": 128,
                "num_layers": 3,
                "heads": 4,
                "dropout": 0.3,
            },
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "model_name": "GATLinkPredictor",
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": 256,
                "out_channels": 128,
                "num_layers": 3,
                "heads": 4,
                "dropout": 0.4,
            },
            "lr": 1e-3,
            "weight_decay": 5e-4,
        },
        {
            "model_name": "GATLinkPredictor",
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": 384,
                "out_channels": 128,
                "num_layers": 4,
                "heads": 4,
                "dropout": 0.3,
            },
            "lr": 8e-4,
            "weight_decay": 1e-4,
        },
    ]


def run_search(args, x, edge_index, train_pos, valid_pos, valid_neg, num_nodes, in_channels,
               device, search_ckpt_path, final_ckpt_path):
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    search_space = build_search_space(in_channels)
    start_time = time.time()
    best_run = None
    all_runs = []

    for cfg in search_space:
        print("\n" + "=" * 100)
        print(f"Trying config: {cfg['model_name']} | kwargs={cfg['model_kwargs']} | lr={cfg['lr']} | wd={cfg['weight_decay']}")
        print("=" * 100)

        for seed in seeds:
            result = train_search_once(
                cfg=cfg,
                seed=seed,
                x=x,
                edge_index=edge_index,
                train_pos=train_pos,
                valid_pos=valid_pos,
                valid_neg=valid_neg,
                num_nodes=num_nodes,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                neg_per_pos=10,
            )
            all_runs.append(result)
            print(f"Finished seed={result['seed']} | best_val_hits50={result['best_val_h50']:.4f} | best_epoch={result['best_epoch']}")

            if best_run is None or result["best_val_h50"] > best_run["best_val_h50"]:
                best_run = result

    ranked = sorted(all_runs, key=lambda r: r["best_val_h50"], reverse=True)

    checkpoint = {
        "mode": "search",
        "model_name": best_run["model_name"],
        "model_kwargs": best_run["model_kwargs"],
        "state_dict": best_run["state_dict"],
        "best_val_h50": best_run["best_val_h50"],
        "best_epoch": best_run["best_epoch"],
        "seed": best_run["seed"],
        "lr": best_run["lr"],
        "weight_decay": best_run["weight_decay"],
        "all_runs": [{
            "model_name": r["model_name"],
            "seed": r["seed"],
            "best_val_h50": r["best_val_h50"],
            "best_epoch": r["best_epoch"],
            "model_kwargs": r["model_kwargs"],
            "lr": r["lr"],
            "weight_decay": r["weight_decay"],
        } for r in ranked],
    }

    torch.save(checkpoint, search_ckpt_path)
    torch.save(checkpoint, final_ckpt_path)

    print("\n" + "#" * 100)
    print("BEST SEARCH RUN FOR C")
    print(f"model={best_run['model_name']} | seed={best_run['seed']} | best_val_hits50={best_run['best_val_h50']:.4f} | best_epoch={best_run['best_epoch']}")
    print(f"model_kwargs={best_run['model_kwargs']}")
    print(f"Saved search checkpoint to: {search_ckpt_path}")
    print(f"Copied current best checkpoint to: {final_ckpt_path}")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    return checkpoint


def train_final_once(
    cfg,
    seed,
    x,
    edge_index,
    train_pos,
    num_nodes,
    device,
    final_epochs,
    neg_per_pos=10,
):
    set_seed(seed)
    model = GATLinkPredictor(**cfg["model_kwargs"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=final_epochs, eta_min=1e-5
    )
    train_pos = train_pos.to(device)
    best_state = None
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, final_epochs + 1):
        model.train()
        optimizer.zero_grad()
        neg_edges = sample_negatives(num_nodes, train_pos.size(0), neg_per_pos, device)
        emb = model(x, edge_index)
        pos_scores = predict_link(emb, train_pos)
        neg_scores = predict_link(emb, neg_edges).view(-1, neg_per_pos)
        loss = margin_ranking_loss(pos_scores, neg_scores, margin=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)
        if epoch % 20 == 0 or epoch == 1 or epoch == final_epochs:
            print(f"[final] epoch={epoch:04d} loss={current_loss:.4f}")
    return {"state_dict": best_state, "best_loss": best_loss, "best_epoch": best_epoch}


def run_final(args, base, x, edge_index, train_pos, num_nodes, device, final_ckpt_path, search_ckpt_path):
    best_cfg = {
        "model_name": base["model_name"],
        "model_kwargs": base["model_kwargs"],
        "lr": base["lr"],
        "weight_decay": base["weight_decay"],
    }
    best_seed = int(base["seed"])
    print("\n" + "#" * 100)
    print("FINAL RETRAIN FOR C")
    print(f"Using best config from search checkpoint: {search_ckpt_path}")
    print(f"best_seed={best_seed} | final_epochs={args.final_epochs}")
    print(f"model_kwargs={best_cfg['model_kwargs']}")
    print("#" * 100 + "\n")
    start_time = time.time()
    final_result = train_final_once(
        cfg=best_cfg,
        seed=best_seed,
        x=x,
        edge_index=edge_index,
        train_pos=train_pos,
        num_nodes=num_nodes,
        device=device,
        final_epochs=args.final_epochs,
        neg_per_pos=10,
    )
    final_checkpoint = {
        "mode": "final",
        "model_name": best_cfg["model_name"],
        "model_kwargs": best_cfg["model_kwargs"],
        "state_dict": final_result["state_dict"],
        "seed": best_seed,
        "final_epochs": args.final_epochs,
        "best_train_loss": final_result["best_loss"],
        "best_epoch": final_result["best_epoch"],
        "source_search_checkpoint": search_ckpt_path,
    }
    torch.save(final_checkpoint, final_ckpt_path)
    print("\nSaved final submission model to:", final_ckpt_path)
    print(f"Best train loss: {final_result['best_loss']:.4f}")
    print(f"Best epoch     : {final_result['best_epoch']}")
    print(f"Elapsed time   : {time.time() - start_time:.2f} seconds")


def main():
    args = parse_args()
    device = get_device()
    ds = load_dataset("C", args.data_dir)
    x = ds.x.float().to(device)
    edge_index = ds.edge_index.long().to(device)
    train_pos = ds.train_pos.long()
    valid_pos = ds.valid_pos.long()
    valid_neg = ds.valid_neg.long()
    num_nodes = x.size(0)
    in_channels = x.size(1)

    os.makedirs(args.model_dir, exist_ok=True)
    search_ckpt_path = os.path.join(args.model_dir, f"{args.kerberos}_model_C_search.pt")
    final_ckpt_path = os.path.join(args.model_dir, f"{args.kerberos}_model_C.pt")

    if args.mode == "search":
        run_search(args, x, edge_index, train_pos, valid_pos, valid_neg, num_nodes, in_channels,
                   device, search_ckpt_path, final_ckpt_path)
    elif args.mode == "final":
        if not os.path.isfile(search_ckpt_path):
            raise FileNotFoundError(
                f"Search checkpoint not found: {search_ckpt_path}\n"
                f"Run --mode search or --mode full first."
            )
        base = torch.load(search_ckpt_path, map_location="cpu", weights_only=False)
        run_final(args, base, x, edge_index, train_pos, num_nodes, device,
                  final_ckpt_path, search_ckpt_path)
    else:  # full
        base = run_search(args, x, edge_index, train_pos, valid_pos, valid_neg, num_nodes,
                          in_channels, device, search_ckpt_path, final_ckpt_path)
        run_final(args, base, x, edge_index, train_pos, num_nodes, device,
                  final_ckpt_path, search_ckpt_path)


if __name__ == "__main__":
    main()
