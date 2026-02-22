#!/usr/bin/env python3
"""Inspect Garden trainability and checkpoint integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect trainable params and Garden checkpoints")
    p.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    p.add_argument("--min_checkpoint_size_mb", type=float, default=0.5)
    p.add_argument("--min_l2_norm", type=float, default=1e-5)
    p.add_argument("--min_abs_sum", type=float, default=1e-2)
    return p.parse_args()


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _stack_stats(state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    parts = [v.detach().float().reshape(-1) for v in state.values() if torch.is_tensor(v)]
    if not parts:
        return {"num_params": 0, "sum_abs": 0.0, "l2_norm": 0.0, "max_abs": 0.0}
    vec = torch.cat(parts, dim=0)
    abs_vec = vec.abs()
    return {
        "num_params": int(vec.numel()),
        "sum_abs": float(abs_vec.sum().item()),
        "l2_norm": float(torch.linalg.vector_norm(vec, ord=2).item()),
        "max_abs": float(abs_vec.max().item()),
    }


def main() -> None:
    args = parse_args()
    model_paths_file = _as_abs(args.model_paths)
    if not model_paths_file.exists():
        raise FileNotFoundError(f"model_paths missing: {model_paths_file}")
    model_paths = yaml.safe_load(model_paths_file.read_text()) or {}
    restoration = model_paths.get("restoration", {}) if isinstance(model_paths, dict) else {}
    model_config = _as_abs(str(restoration.get("model_config", "configs/default_sd15.yaml")))
    checkpoint = _as_abs(str(restoration.get("adapter_checkpoint", "checkpoints/garden/adapter_best.pt")))

    failures: List[str] = []
    if not model_config.exists():
        failures.append(f"model config missing: {model_config}")
    if not checkpoint.exists():
        failures.append(f"checkpoint missing: {checkpoint}")
    if failures:
        print(json.dumps({"status": "fail", "failures": failures}, indent=2))
        raise SystemExit(" | ".join(failures))

    cfg = load_config(str(model_config))
    model_cfg = ArtifactRestorationDiffusionConfig(
        model_name_or_path=str(cfg["model"]["pretrained_model_name_or_path"]),
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        lora_rank=int(cfg["model"]["lora_rank"]),
        guidance_scale=float(cfg["model"]["guidance_scale"]),
        prompt_dropout=float(cfg["model"]["prompt_dropout"]),
        max_token_length=int(cfg["model"]["max_token_length"]),
        vae_scaling_factor=float(cfg["model"]["vae_scaling_factor"]),
    )
    model = FrozenBackboneArtifactRestorationModel(model_cfg, device=torch.device("cpu"), dtype=torch.float32)
    model.freeze_backbone()
    model.phi_minus.requires_grad_(True)
    model.phi_plus.requires_grad_(True)

    trainable_rows = [{"name": n, "numel": int(p.numel())} for n, p in model.named_parameters() if p.requires_grad]
    minus_params = [p for p in model.phi_minus.parameters() if p.requires_grad]
    plus_params = [p for p in model.phi_plus.parameters() if p.requires_grad]
    opt_minus = torch.optim.AdamW(minus_params, lr=5e-6) if minus_params else None
    opt_plus = torch.optim.AdamW(plus_params, lr=1e-6) if plus_params else None

    if not minus_params:
        failures.append("phi_minus has no trainable parameters")
    if not plus_params:
        failures.append("phi_plus has no trainable parameters")
    if opt_minus is None or len(opt_minus.param_groups[0]["params"]) == 0:
        failures.append("optimizer for phi_minus is empty")
    if opt_plus is None or len(opt_plus.param_groups[0]["params"]) == 0:
        failures.append("optimizer for phi_plus is empty")

    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    phi_minus_state = ckpt.get("phi_minus", {})
    phi_plus_state = ckpt.get("phi_plus", {})
    minus_stats = _stack_stats(phi_minus_state)
    plus_stats = _stack_stats(phi_plus_state)
    ckpt_size_mb = float(checkpoint.stat().st_size / (1024.0 * 1024.0))
    if ckpt_size_mb < float(args.min_checkpoint_size_mb):
        failures.append(
            f"checkpoint too small: {ckpt_size_mb:.3f} MB < {float(args.min_checkpoint_size_mb):.3f} MB"
        )
    if float(minus_stats["sum_abs"]) < float(args.min_abs_sum) or float(minus_stats["l2_norm"]) < float(args.min_l2_norm):
        failures.append("phi_minus checkpoint weights appear near-zero/untrained")
    if float(plus_stats["sum_abs"]) < float(args.min_abs_sum) or float(plus_stats["l2_norm"]) < float(args.min_l2_norm):
        failures.append("phi_plus checkpoint weights appear near-zero/untrained")

    payload = {
        "status": "fail" if failures else "ok",
        "model_paths": str(model_paths_file),
        "model_config": str(model_config),
        "checkpoint": str(checkpoint),
        "checkpoint_size_mb": ckpt_size_mb,
        "checkpoint_step": int(ckpt.get("step", -1)),
        "trainable_params_total": int(sum(r["numel"] for r in trainable_rows)),
        "trainable_param_tensors": trainable_rows,
        "phi_minus_param_count": int(sum(p.numel() for p in minus_params)),
        "phi_plus_param_count": int(sum(p.numel() for p in plus_params)),
        "optimizer_phi_minus_param_tensors": len(opt_minus.param_groups[0]["params"]) if opt_minus is not None else 0,
        "optimizer_phi_plus_param_tensors": len(opt_plus.param_groups[0]["params"]) if opt_plus is not None else 0,
        "phi_minus_checkpoint_stats": minus_stats,
        "phi_plus_checkpoint_stats": plus_stats,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
