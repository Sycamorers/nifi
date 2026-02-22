#!/usr/bin/env python3
"""Inspect adapter checkpoints and fail if they look missing/untrained."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect Garden adapter checkpoint statistics")
    p.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--min_checkpoint_size_mb", type=float, default=0.5)
    p.add_argument("--min_abs_sum", type=float, default=1e-2)
    p.add_argument("--min_l2_norm", type=float, default=1e-5)
    return p.parse_args()


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _read_model_paths(path: Path) -> Dict[str, Path]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    restoration = payload.get("restoration", {})
    if not isinstance(restoration, dict):
        raise ValueError(f"`restoration` mapping missing in {path}")
    model_config = restoration.get("model_config")
    checkpoint = restoration.get("adapter_checkpoint")
    if not isinstance(model_config, str) or not isinstance(checkpoint, str):
        raise ValueError("model_paths.yaml must define restoration.model_config and restoration.adapter_checkpoint")
    return {"model_config": _as_abs(model_config), "checkpoint": _as_abs(checkpoint)}


def _stack_params(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [v.detach().float().reshape(-1) for v in state_dict.values() if torch.is_tensor(v)]
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _adapter_stats(state_dict: Dict[str, torch.Tensor], topk: int) -> Dict[str, object]:
    vec = _stack_params(state_dict)
    if vec.numel() == 0:
        return {
            "num_params": 0,
            "sum_abs": 0.0,
            "l2_norm": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "topk_abs": [],
        }
    abs_vec = vec.abs()
    k = min(int(topk), int(abs_vec.numel()))
    largest = torch.topk(abs_vec, k=k).values.tolist() if k > 0 else []
    return {
        "num_params": int(vec.numel()),
        "sum_abs": float(abs_vec.sum().item()),
        "l2_norm": float(torch.linalg.vector_norm(vec, ord=2).item()),
        "mean_abs": float(abs_vec.mean().item()),
        "max_abs": float(abs_vec.max().item()),
        "topk_abs": [float(x) for x in largest],
    }


def _guess_model_family(model_name_or_path: str) -> str:
    low = model_name_or_path.lower()
    if "sd3" in low or "stable-diffusion-3" in low:
        return "sd3-family"
    if "xl" in low or "sdxl" in low:
        return "sdxl-family"
    if "stable-diffusion-v1" in low or "sd15" in low or "v1-5" in low:
        return "sd1.x-family"
    if "stable-diffusion-v2" in low:
        return "sd2.x-family"
    return "unknown-family"


def _is_effectively_zero(stats: Dict[str, object], min_abs_sum: float, min_l2_norm: float) -> bool:
    return bool(
        int(stats["num_params"]) == 0
        or float(stats["sum_abs"]) < float(min_abs_sum)
        or float(stats["l2_norm"]) < float(min_l2_norm)
        or (math.isfinite(float(stats["max_abs"])) and float(stats["max_abs"]) <= 0.0)
    )


def main() -> None:
    args = parse_args()
    model_paths_file = _as_abs(args.model_paths)
    if not model_paths_file.exists():
        raise FileNotFoundError(f"Missing model paths file: {model_paths_file}")
    paths = _read_model_paths(model_paths_file)
    model_config_path = paths["model_config"]
    checkpoint_path = paths["checkpoint"]
    if not model_config_path.exists():
        raise FileNotFoundError(f"Missing model config: {model_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing adapter checkpoint: {checkpoint_path}")

    cfg = load_config(str(model_config_path))
    model_name_or_path = str(cfg["model"]["pretrained_model_name_or_path"])
    model_family = _guess_model_family(model_name_or_path)
    vae_scaling_factor = float(cfg["model"]["vae_scaling_factor"])

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if "phi_minus" not in ckpt or "phi_plus" not in ckpt:
        raise KeyError("Checkpoint must contain `phi_minus` and `phi_plus`")

    phi_minus_stats = _adapter_stats(ckpt["phi_minus"], topk=int(args.topk))
    phi_plus_stats = _adapter_stats(ckpt["phi_plus"], topk=int(args.topk))

    file_size_bytes = checkpoint_path.stat().st_size
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)

    failures: List[str] = []
    if file_size_mb < float(args.min_checkpoint_size_mb):
        failures.append(
            f"checkpoint too small ({file_size_mb:.3f} MB < {float(args.min_checkpoint_size_mb):.3f} MB)"
        )
    if _is_effectively_zero(phi_minus_stats, args.min_abs_sum, args.min_l2_norm):
        failures.append("phi_minus appears near-zero/untrained")
    if _is_effectively_zero(phi_plus_stats, args.min_abs_sum, args.min_l2_norm):
        failures.append("phi_plus appears near-zero/untrained")

    payload = {
        "model_paths": str(model_paths_file),
        "model_config": str(model_config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": int(file_size_bytes),
        "checkpoint_size_mb": float(file_size_mb),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_best_metric": None if ckpt.get("best_metric") is None else float(ckpt.get("best_metric")),
        "model_name_or_path": model_name_or_path,
        "model_family_guess": model_family,
        "vae_scaling_factor": vae_scaling_factor,
        "phi_minus": phi_minus_stats,
        "phi_plus": phi_plus_stats,
        "status": "fail" if failures else "ok",
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))

    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
