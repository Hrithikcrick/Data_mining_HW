"""
predict.py  –  COL761 Assignment 3 prediction script
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from load_dataset import COL761NodeDataset, COL761LinkDataset, load_dataset, _load_edge_list
import models as student_models


def load_model(model_path: str) -> torch.nn.Module:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    obj = torch.load(model_path, weights_only=False, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        model = obj

    elif isinstance(obj, dict):
        model_name = obj.get("model_name", obj.get("model_class"))
        model_kwargs = obj.get("model_kwargs", {})
        state_dict = obj.get("state_dict", None)

        if model_name is None or state_dict is None:
            raise ValueError(
                "Checkpoint dict must contain model_name/model_class and state_dict."
            )

        if not hasattr(student_models, model_name):
            raise ValueError(f"Model class '{model_name}' not found in models.py")

        model_cls = getattr(student_models, model_name)
        model = model_cls(**model_kwargs)
        model.load_state_dict(state_dict)

    else:
        raise TypeError("Unsupported checkpoint format. Expected torch.nn.Module or dict.")

    model.eval()
    return model


def _random_A(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.randint(0, dataset[0].num_classes, (dataset[0].num_nodes,), dtype=torch.long)


def _random_B(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.rand(dataset[0].num_nodes)


def _random_C(V: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(V), torch.rand(V, K)


@torch.no_grad()
def predict_A(model: torch.nn.Module, dataset: COL761NodeDataset) -> torch.Tensor:
    data = dataset[0]

    x = data.x.float()
    x = F.normalize(x, p=1, dim=1)
    edge_index = data.edge_index.long()

    logits = model(x, edge_index)
    return logits.argmax(dim=1).long()


@torch.no_grad()
def predict_B(model: torch.nn.Module, dataset: COL761NodeDataset) -> torch.Tensor:
    data = dataset[0]
    logits = model(data.x.float(), data.edge_index.long())

    if logits.ndim == 1:
        return torch.sigmoid(logits.float())

    if logits.shape[1] == 1:
        return torch.sigmoid(logits.float()).squeeze(1)

    return torch.softmax(logits.float(), dim=1)[:, 1]


@torch.no_grad()
def predict_C(
    model: torch.nn.Module,
    dataset: COL761LinkDataset,
    test_dir: str = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if test_dir is None:
        pos = dataset.valid_pos
        neg = dataset.valid_neg
        split = "valid"
    else:
        pos = _load_edge_list(os.path.join(test_dir, "test_pos.txt"))
        npy = os.path.join(test_dir, "test_neg_hard.npy")
        with open(npy, "rb") as f:
            neg = torch.from_numpy(np.load(f))
        split = "test"

    # Compute node embeddings once
    x = dataset.x.float()
    edge_index = dataset.edge_index.long()
    embeddings = model(x, edge_index)   # shape (num_nodes, out_channels)

    # Positive scores: dot product for each edge in pos (shape (P, 2))
    pos_scores = (embeddings[pos[:, 0]] * embeddings[pos[:, 1]]).sum(dim=1)

    # Negative edges: shape (P, K, 2) or (num_neg, 2)
    if neg.dim() == 3:
        P, K, _ = neg.shape
        neg_flat = neg.view(P * K, 2)
        neg_scores = (embeddings[neg_flat[:, 0]] * embeddings[neg_flat[:, 1]]).sum(dim=1)
        neg_scores = neg_scores.view(P, K)
    else:
        neg_scores = (embeddings[neg[:, 0]] * embeddings[neg[:, 1]]).sum(dim=1)

    return pos_scores.float(), neg_scores.float(), split


def predict_and_save(
    dataset_name: str,
    data_dir: str,
    model_path: str,
    out_dir: str,
    test_dir: str = None,
    kerberos: str = "student",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset {dataset_name} ...")
    ds = load_dataset(dataset_name, data_dir)

    if model_path is not None:
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path)
    else:
        print("No --model_dir given — using random predictions.")
        model = None

    if dataset_name == "A":
        y_pred = predict_A(model, ds) if model else _random_A(ds)
        assert y_pred.shape == (ds[0].num_nodes,)
        assert y_pred.dtype == torch.long

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_A.pt")
        torch.save({"y_pred": y_pred.cpu()}, out_path)
        print(f"Saved {out_path}  shape={y_pred.shape}")

    elif dataset_name == "B":
        y_score = predict_B(model, ds) if model else _random_B(ds)
        assert y_score.shape == (ds[0].num_nodes,)
        assert y_score.is_floating_point()

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_B.pt")
        torch.save({"y_score": y_score.cpu()}, out_path)
        print(f"Saved {out_path}  shape={y_score.shape}")

    elif dataset_name == "C":
        if model:
            pos_scores, neg_scores, split = predict_C(model, ds, test_dir=test_dir)
        else:
            if test_dir or not hasattr(ds, "valid_pos"):
                pos = ds.test_pos
                neg = ds.test_neg
                split = "test"
            else:
                pos = ds.valid_pos
                neg = ds.valid_neg
                split = "valid"
            V, K = pos.shape[0], neg.shape[1]
            pos_scores, neg_scores = _random_C(V, K)

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_C.pt")
        torch.save(
            {
                "pos_scores": pos_scores.cpu(),
                "neg_scores": neg_scores.cpu(),
                "split": split,
            },
            out_path,
        )
        print(f"Saved {out_path}  split={split}")
        print(f"  pos_scores : {pos_scores.shape}")
        print(f"  neg_scores : {neg_scores.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for COL761 A3 datasets."
    )
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--test_dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    model_path = None
    if args.model_dir is not None:
        model_path = os.path.join(
            args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt"
        )

    predict_and_save(
        args.dataset,
        args.data_dir,
        model_path,
        args.output_dir,
        test_dir=args.test_dir,
        kerberos=args.kerberos,
    )


if __name__ == "__main__":
    main()