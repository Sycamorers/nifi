from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: Path,
    step: int,
    phi_minus_state: Dict[str, Any],
    phi_plus_state: Dict[str, Any],
    opt_minus_state: Optional[Dict[str, Any]] = None,
    opt_plus_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "phi_minus": phi_minus_state,
        "phi_plus": phi_plus_state,
        "opt_minus": opt_minus_state,
        "opt_plus": opt_plus_state,
        "scaler": scaler_state,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
