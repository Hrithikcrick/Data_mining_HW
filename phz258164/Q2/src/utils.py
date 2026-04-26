import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y_true).float().mean().item()


def clone_state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")