#!/usr/bin/env python3
"""Train phi- / phi+ adapters with reproducible small/full presets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NiFi adapters (small/full presets)")
    parser.add_argument("--preset", type=str, default="small", choices=["small", "full"])
    parser.add_argument("--base_config", type=str, default="configs/default_sd15.yaml")
    parser.add_argument("--data_root", type=str, default="pairs_real")
    parser.add_argument("--exp", type=str, default="runs/nifi_sd15_adapters")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--update_model_paths", action="store_true")
    parser.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    return parser.parse_args()


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _preset_overrides(preset: str) -> Dict[str, object]:
    if preset == "full":
        return {
            "train.max_steps": 6000,
            "train.batch_size": 1,
            "train.eval_every": 200,
            "train.save_every": 200,
            "train.max_eval_samples": 64,
            "diffusion.t0": 0,
        }
    return {
        "train.max_steps": 120,
        "train.batch_size": 1,
        "train.eval_every": 20,
        "train.save_every": 20,
        "train.max_eval_samples": 24,
        "diffusion.t0": 0,
    }


def _set_nested(cfg: Dict[str, object], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
        except Exception as exc:
            raise RuntimeError("CUDA check failed; use --device cpu or fix GPU runtime") from exc

    base_cfg_path = _as_abs(args.base_config)
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base config not found: {base_cfg_path}")
    data_root = _as_abs(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    exp_dir = _as_abs(args.exp)
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(base_cfg_path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping YAML in {base_cfg_path}")

    overrides = _preset_overrides(args.preset)
    for key, value in overrides.items():
        _set_nested(cfg, key, value)

    _set_nested(cfg, "runtime.device", args.device)
    _set_nested(cfg, "output_dir", str(exp_dir))

    if args.max_steps is not None:
        _set_nested(cfg, "train.max_steps", int(args.max_steps))
    if args.batch_size is not None:
        _set_nested(cfg, "train.batch_size", int(args.batch_size))
    if args.image_size is not None:
        _set_nested(cfg, "image_size", int(args.image_size))

    resolved_config_path = exp_dir / "resolved_train_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_nifi.py"),
        "--config",
        str(resolved_config_path),
        "--data_root",
        str(data_root),
        "--exp",
        str(exp_dir),
    ]
    if args.resume:
        cmd.extend(["--resume", str(_as_abs(args.resume))])

    print("[info] launching adapter training")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    best_ckpt = exp_dir / "best.pt"
    latest_ckpt = exp_dir / "latest.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"training finished but best checkpoint missing: {best_ckpt}")

    export_ckpt = ROOT / "checkpoints" / f"{exp_dir.name}_best.pt"
    export_ckpt.parent.mkdir(parents=True, exist_ok=True)
    export_ckpt.write_bytes(best_ckpt.read_bytes())

    summary = {
        "preset": args.preset,
        "resolved_config": str(resolved_config_path),
        "data_root": str(data_root),
        "exp_dir": str(exp_dir),
        "best_checkpoint": str(best_ckpt),
        "latest_checkpoint": str(latest_ckpt) if latest_ckpt.exists() else None,
        "export_checkpoint": str(export_ckpt),
    }
    print(json.dumps(summary, indent=2))

    if args.update_model_paths:
        model_paths_file = _as_abs(args.model_paths)
        payload = yaml.safe_load(model_paths_file.read_text()) if model_paths_file.exists() else {}
        if not isinstance(payload, dict):
            payload = {}
        restoration = payload.get("restoration", {})
        if not isinstance(restoration, dict):
            restoration = {}
        restoration["adapter_checkpoint"] = str(export_ckpt.relative_to(ROOT))
        restoration.setdefault("model_config", str(Path(args.base_config)))
        payload["restoration"] = restoration
        model_paths_file.parent.mkdir(parents=True, exist_ok=True)
        model_paths_file.write_text(yaml.safe_dump(payload, sort_keys=False))
        print(f"[done] updated {model_paths_file} -> {restoration['adapter_checkpoint']}")


if __name__ == "__main__":
    main()

