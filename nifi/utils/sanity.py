from typing import Dict

import torch


def assert_no_nan(name: str, tensor: torch.Tensor) -> None:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"Found NaN/Inf in tensor: {name}")


def assert_shape(name: str, tensor: torch.Tensor, ndims: int) -> None:
    if tensor.ndim != ndims:
        raise ValueError(f"Unexpected ndim for {name}: {tensor.ndim} vs expected {ndims}")


def summarize_grad_norm(module: torch.nn.Module) -> float:
    sq_norm = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        sq_norm += p.grad.detach().float().norm(2).item() ** 2
    return sq_norm ** 0.5


def tiny_overfit_guard(history: Dict[str, float], tolerance: float = 1e-3) -> bool:
    """Returns True if training appears to make progress."""
    if "start_loss" not in history or "end_loss" not in history:
        return True
    return history["end_loss"] <= history["start_loss"] + tolerance
