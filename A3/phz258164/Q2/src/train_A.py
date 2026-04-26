import argparse
import os
import time
from copy import deepcopy

import torch
import torch.nn.functional as F

from load_dataset import load_dataset
from models import GATv2Classifier
from utils import set_seed, accuracy_from_logits, clone_state_dict_to_cpu, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--kerberos", type=str, required=True)

    parser.add_argument("--mode", type=str, default="search", choices=["search", "final"])

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=220)

    parser.add_argument("--num_seeds", type=int, default=12)
    parser.add_argument("--start_seed", type=int, default=0)

    parser.add_argument("--final_epochs", type=int, default=400)
    return parser.parse_args()


def build_search_space(in_channels, num_classes):
    return [
        {
            "model_name": "GATv2Classifier",
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": 16,
                "out_channels": num_classes,
                "heads": 4,
                "dropout": 0.60,
            },
            "lr": 0.003,
            "weight_decay": 5e-4,
            "label_smoothing": 0.00,
        },
        {
            "model_name": "GATv2Classifier",
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": 16,
                "out_channels": num_classes,
                "heads": 4,
                "dropout": 0.50,
            },
            "lr": 0.003,
            "weight_decay": 5e-4,
            "label_smoothing": 0.03,
        },
        {
            "model_name": "GATv2Classifier",
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": 24,
                "out_channels": num_classes,
                "heads": 4,
                "dropout": 0.60,
            },
            "lr": 0.002,
            "weight_decay": 1e-4,
            "label_smoothing": 0.03,
        },
        {
            "model_name": "GATv2Classifier",
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": 12,
                "out_channels": num_classes,
                "heads": 8,
                "dropout": 0.60,
            },
            "lr": 0.003,
            "weight_decay": 5e-4,
            "label_smoothing": 0.00,
        },
    ]


def make_model(cfg, device):
    model = GATv2Classifier(**cfg["model_kwargs"]).to(device)
    return model


def train_search_once(
    cfg,
    seed,
    x,
    edge_index,
    y,
    labeled_nodes,
    train_mask,
    val_mask,
    device,
    epochs,
    patience,
):
    set_seed(seed)

    model = make_model(cfg, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=30,
        min_lr=1e-4,
    )

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits_all = model(x, edge_index)       # [N, C]
        logits_lab = logits_all[labeled_nodes]  # [L, C]

        train_logits = logits_lab[train_mask]
        train_y = y[train_mask]

        loss = F.cross_entropy(
            train_logits,
            train_y,
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(x, edge_index)
            logits_lab = logits_all[labeled_nodes]

            train_acc = accuracy_from_logits(logits_lab[train_mask], y[train_mask])
            val_acc = accuracy_from_logits(logits_lab[val_mask], y[val_mask])

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"[search] hidden={cfg['model_kwargs']['hidden_channels']} "
                f"heads={cfg['model_kwargs']['heads']} "
                f"dropout={cfg['model_kwargs']['dropout']} "
                f"seed={seed} epoch={epoch:04d} "
                f"loss={loss.item():.4f} train={train_acc:.4f} val={val_acc:.4f}"
            )

        if bad_epochs >= patience:
            break

    return {
        "seed": seed,
        "model_name": cfg["model_name"],
        "model_kwargs": deepcopy(cfg["model_kwargs"]),
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "label_smoothing": cfg.get("label_smoothing", 0.0),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }


def train_final_once(
    cfg,
    seed,
    x,
    edge_index,
    y,
    labeled_nodes,
    device,
    final_epochs,
):
    set_seed(seed)

    model = make_model(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=final_epochs,
        eta_min=max(cfg["lr"] * 0.05, 1e-5),
    )

    best_state = None
    best_loss = float("inf")
    best_acc = -1.0
    best_epoch = -1

    for epoch in range(1, final_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits_all = model(x, edge_index)
        logits_lab = logits_all[labeled_nodes]

        loss = F.cross_entropy(
            logits_lab,
            y,
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(x, edge_index)
            logits_lab = logits_all[labeled_nodes]
            labeled_loss = F.cross_entropy(
                logits_lab,
                y,
                label_smoothing=cfg.get("label_smoothing", 0.0),
            ).item()
            labeled_acc = accuracy_from_logits(logits_lab, y)

        improved = False
        if labeled_loss < best_loss - 1e-7:
            improved = True
        elif abs(labeled_loss - best_loss) <= 1e-7 and labeled_acc > best_acc:
            improved = True

        if improved:
            best_loss = labeled_loss
            best_acc = labeled_acc
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)

        if epoch % 25 == 0 or epoch == 1 or epoch == final_epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[final] epoch={epoch:04d} "
                f"loss={labeled_loss:.4f} "
                f"labeled_acc={labeled_acc:.4f} "
                f"lr={current_lr:.6f}"
            )

    if best_state is None:
        raise RuntimeError("Final retrain failed: no checkpoint was saved.")

    return {
        "state_dict": best_state,
        "best_loss": best_loss,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
    }


def main():
    args = parse_args()
    device = get_device()

    dataset = load_dataset("A", args.data_dir)
    data = dataset[0]

    # Must match predict.py
    x = data.x.float()
    x = F.normalize(x, p=1, dim=1)
    x = x.to(device)

    edge_index = data.edge_index.long().to(device)

    # Labels exist only for labeled_nodes
    y = data.y.long().to(device)
    labeled_nodes = data.labeled_nodes.long().to(device)
    train_mask = data.train_mask.bool().to(device)
    val_mask = data.val_mask.bool().to(device)

    num_classes = int(y.max().item() + 1)
    in_channels = x.size(1)

    os.makedirs(args.model_dir, exist_ok=True)
    search_ckpt_path = os.path.join(args.model_dir, f"{args.kerberos}_model_A_search.pt")
    final_ckpt_path = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")

    if args.mode == "search":
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
        search_space = build_search_space(in_channels, num_classes)

        start_time = time.time()
        best_run = None
        all_runs = []

        for cfg in search_space:
            print("\n" + "=" * 100)
            print(f"Trying config: {cfg['model_name']} | kwargs={cfg['model_kwargs']}")
            print("=" * 100)

            for seed in seeds:
                result = train_search_once(
                    cfg=cfg,
                    seed=seed,
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    labeled_nodes=labeled_nodes,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                )
                all_runs.append(result)

                print(
                    f"Finished seed={result['seed']} | "
                    f"best_val={result['best_val_acc']:.4f} | "
                    f"best_epoch={result['best_epoch']}"
                )

                if best_run is None or result["best_val_acc"] > best_run["best_val_acc"]:
                    best_run = result

        ranked = sorted(all_runs, key=lambda r: r["best_val_acc"], reverse=True)

        checkpoint = {
            "mode": "search",
            "preprocess": {"normalize": "l1"},
            "model_name": best_run["model_name"],
            "model_kwargs": best_run["model_kwargs"],
            "state_dict": best_run["state_dict"],
            "best_val_acc": best_run["best_val_acc"],
            "best_epoch": best_run["best_epoch"],
            "seed": best_run["seed"],
            "lr": best_run["lr"],
            "weight_decay": best_run["weight_decay"],
            "label_smoothing": best_run["label_smoothing"],
            "all_runs": [
                {
                    "model_name": r["model_name"],
                    "seed": r["seed"],
                    "best_val_acc": r["best_val_acc"],
                    "best_epoch": r["best_epoch"],
                    "model_kwargs": r["model_kwargs"],
                    "lr": r["lr"],
                    "weight_decay": r["weight_decay"],
                    "label_smoothing": r["label_smoothing"],
                }
                for r in ranked
            ],
        }

        torch.save(checkpoint, search_ckpt_path)
        torch.save(checkpoint, final_ckpt_path)

        print("\n" + "#" * 100)
        print("BEST SEARCH RUN")
        print(
            f"model={best_run['model_name']} | "
            f"seed={best_run['seed']} | "
            f"best_val={best_run['best_val_acc']:.4f} | "
            f"best_epoch={best_run['best_epoch']}"
        )
        print(f"model_kwargs={best_run['model_kwargs']}")
        print(f"Saved search checkpoint to: {search_ckpt_path}")
        print(f"Copied current best checkpoint to: {final_ckpt_path}")
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

        print("\nTop runs:")
        for i, r in enumerate(ranked[:10], start=1):
            print(
                f"{i:02d}. {r['model_name']:16s} seed={r['seed']} "
                f"val={r['best_val_acc']:.4f} epoch={r['best_epoch']}"
            )

    else:
        if not os.path.isfile(search_ckpt_path):
            raise FileNotFoundError(
                f"Search checkpoint not found: {search_ckpt_path}\n"
                f"Run --mode search first."
            )

        base = torch.load(search_ckpt_path, map_location="cpu", weights_only=False)

        best_cfg = {
            "model_name": base["model_name"],
            "model_kwargs": base["model_kwargs"],
            "lr": base["lr"],
            "weight_decay": base["weight_decay"],
            "label_smoothing": base.get("label_smoothing", 0.0),
        }

        best_seed = int(base["seed"])

        print("\n" + "#" * 100)
        print("STRONG FINAL RETRAIN ON ALL LABELED NODES")
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
            y=y,
            labeled_nodes=labeled_nodes,
            device=device,
            final_epochs=args.final_epochs,
        )

        final_checkpoint = {
            "mode": "final_all_labeled_strong",
            "preprocess": {"normalize": "l1"},
            "model_name": best_cfg["model_name"],
            "model_kwargs": best_cfg["model_kwargs"],
            "state_dict": final_result["state_dict"],
            "seed": best_seed,
            "final_epochs": args.final_epochs,
            "best_labeled_loss": final_result["best_loss"],
            "best_labeled_acc": final_result["best_acc"],
            "best_epoch": final_result["best_epoch"],
            "source_search_checkpoint": search_ckpt_path,
        }

        torch.save(final_checkpoint, final_ckpt_path)

        print("\nSaved final submission model to:", final_ckpt_path)
        print(f"Best labeled loss: {final_result['best_loss']:.4f}")
        print(f"Best labeled acc : {final_result['best_acc']:.4f}")
        print(f"Best epoch       : {final_result['best_epoch']}")
        print(f"Elapsed time     : {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()