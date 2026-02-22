#!/usr/bin/env python3
"""Verify restoration checkpoint/model compatibility and print signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _abs_sum(module: torch.nn.Module) -> float:
    s = 0.0
    with torch.no_grad():
        for p in module.parameters():
            s += float(p.detach().abs().sum().cpu().item())
    return s


def _resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _read_model_paths(path: Path) -> Tuple[Path, Path]:
    if not path.exists():
        raise FileNotFoundError(f"model paths config missing: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in: {path}")
    restoration = payload.get("restoration", {})
    if not isinstance(restoration, dict):
        raise ValueError("`restoration` section is required in model_paths.yaml")

    cfg_rel = restoration.get("model_config")
    ckpt_rel = restoration.get("adapter_checkpoint")
    if not isinstance(cfg_rel, str) or not cfg_rel:
        raise ValueError("model_paths.yaml restoration.model_config is required")
    if not isinstance(ckpt_rel, str) or not ckpt_rel:
        raise ValueError("model_paths.yaml restoration.adapter_checkpoint is required")
    return _as_abs(cfg_rel), _as_abs(ckpt_rel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify NiFi restoration checkpoint wiring")
    parser.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths_file = _as_abs(args.model_paths)
    model_config_path, checkpoint_path = _read_model_paths(model_paths_file)

    if not model_config_path.exists():
        raise FileNotFoundError(f"model config not found: {model_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"adapter checkpoint not found: {checkpoint_path}")

    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("verify_checkpoints requires GPU for parity with restoration runs")
    torch.cuda.set_device(device)

    cfg = load_config(str(model_config_path))
    model_cfg = ArtifactRestorationDiffusionConfig(
        model_name_or_path=str(cfg["model"]["pretrained_model_name_or_path"]),
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        lora_rank=int(cfg["model"]["lora_rank"]),
        guidance_scale=float(cfg["model"]["guidance_scale"]),
        prompt_dropout=float(cfg["model"]["prompt_dropout"]),
        max_token_length=int(cfg["model"]["max_token_length"]),
        vae_scaling_factor=float(cfg["model"]["vae_scaling_factor"]),
    )

    mp = str(cfg.get("train", {}).get("mixed_precision", "fp16"))
    dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    use_amp = device.type == "cuda" and mp in {"fp16", "bf16"}

    model = FrozenBackboneArtifactRestorationModel(
        model_cfg,
        device=device,
        dtype=dtype if use_amp else torch.float32,
    )

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if "phi_minus" not in ckpt or "phi_plus" not in ckpt:
        raise KeyError("checkpoint must contain `phi_minus` and `phi_plus`")

    model.phi_minus.load_state_dict(ckpt["phi_minus"])
    model.phi_plus.load_state_dict(ckpt["phi_plus"])
    model.freeze_backbone()
    model.eval()

    sig = {
        "model_paths_file": str(model_paths_file),
        "model_config": str(model_config_path),
        "adapter_checkpoint": str(checkpoint_path),
        "device": str(device),
        "gpu_name": str(torch.cuda.get_device_name(device)),
        "model_name_or_path": model_cfg.model_name_or_path,
        "phi_minus_params": _count_params(model.phi_minus),
        "phi_plus_params": _count_params(model.phi_plus),
        "phi_minus_abs_weight_sum": _abs_sum(model.phi_minus),
        "phi_plus_abs_weight_sum": _abs_sum(model.phi_plus),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_best_metric": None if ckpt.get("best_metric") is None else float(ckpt["best_metric"]),
    }

    # One lightweight runtime forward check on latent tensors.
    b = 1
    latent = torch.randn((b, 4, 32, 32), device=device, dtype=torch.float16)
    t = torch.full((b,), 0, device=device, dtype=torch.long)
    with torch.no_grad():
        _ = model.predict_eps(latent, t, [""], adapter_type="minus", train_mode=False)
        _ = model.predict_eps(latent, t, [""], adapter_type="plus", train_mode=False)

    print(json.dumps(sig, indent=2))
    print("[done] checkpoint verification passed")


if __name__ == "__main__":
    main()

