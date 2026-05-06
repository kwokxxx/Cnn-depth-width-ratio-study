from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .metrics import compute_gradient_norm
except ImportError:
    from metrics import compute_gradient_norm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    show_progress: bool = True,
    track_gradient_norm: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    gradient_norm_sum = 0.0
    gradient_norm_count = 0

    start_time = time.perf_counter()
    iterator = tqdm(loader, leave=False, disable=not show_progress)
    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                if track_gradient_norm:
                    gradient_norm_sum += compute_gradient_norm(model)
                    gradient_norm_count += 1
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    elapsed_seconds = time.perf_counter() - start_time
    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples_per_second": total_examples / elapsed_seconds,
    }
    if gradient_norm_count > 0:
        metrics["gradient_norm"] = gradient_norm_sum / gradient_norm_count
    else:
        metrics["gradient_norm"] = 0.0
    return metrics
