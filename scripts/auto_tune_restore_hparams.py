#!/usr/bin/env python3
"""Auto-tune restoration hyperparameters and emit demo config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune restoration hparams (wrapper)")
    parser.add_argument("--dataset", type=str, default="mipnerf360")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    parser.add_argument("--out_config", type=str, default="configs/demo_realdata_known_good.yaml")
    parser.add_argument("--out_summary", type=str, default=None)
    parser.add_argument("--views", nargs="*", default=["auto"])
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--rates", type=str, default="0.1,0.5,1.0")
    parser.add_argument("--t0_values", type=str, default="0,1,2,5,10,20")
    parser.add_argument("--adapter_scales", type=str, default="0.0,0.25,0.5,1.0")
    parser.add_argument("--min_lpips_margin", type=float, default=0.01)
    parser.add_argument("--min_dists_margin", type=float, default=0.005)
    parser.add_argument("--min_ssim_gain", type=float, default=0.0)
    return parser.parse_args()


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _load_model_paths(path: Path) -> Dict[str, str]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    restoration = payload.get("restoration", {})
    if not isinstance(restoration, dict):
        raise ValueError(f"`restoration` block missing in {path}")
    cfg = restoration.get("model_config")
    ckpt = restoration.get("adapter_checkpoint")
    if not isinstance(cfg, str) or not isinstance(ckpt, str):
        raise ValueError(f"`model_config` and `adapter_checkpoint` are required in {path}")
    return {"model_config": cfg, "checkpoint": ckpt}


def main() -> None:
    args = parse_args()
    model_paths_file = _as_abs(args.model_paths)
    if not model_paths_file.exists():
        raise FileNotFoundError(f"model paths file not found: {model_paths_file}")
    mp = _load_model_paths(model_paths_file)

    out_summary = args.out_summary
    if out_summary is None:
        out_summary = str(ROOT / "outputs" / "auto_hparams" / args.dataset / args.scene / "sweep_summary.json")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "auto_select_best_demo_hparams.py"),
        "--dataset",
        args.dataset,
        "--scene",
        args.scene,
        "--num_views",
        str(args.num_views),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--rates",
        args.rates,
        "--t0_values",
        args.t0_values,
        "--adapter_scales",
        args.adapter_scales,
        "--checkpoint",
        str(mp["checkpoint"]),
        "--model_config",
        str(mp["model_config"]),
        "--min_lpips_margin",
        str(args.min_lpips_margin),
        "--min_dists_margin",
        str(args.min_dists_margin),
        "--min_ssim_gain",
        str(args.min_ssim_gain),
        "--out_summary",
        str(out_summary),
        "--out_config",
        str(args.out_config),
    ]
    if args.views:
        cmd.extend(["--views", *args.views])

    print("[info] running sweep command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    out_cfg_path = _as_abs(args.out_config)
    if not out_cfg_path.exists():
        raise FileNotFoundError(f"Sweep did not emit config: {out_cfg_path}")

    cfg_payload = yaml.safe_load(out_cfg_path.read_text()) or {}
    if not isinstance(cfg_payload, dict):
        raise ValueError(f"Expected mapping YAML in {out_cfg_path}")
    restoration = cfg_payload.get("restoration", {})
    if not isinstance(restoration, dict):
        restoration = {}

    # Keep tuned values but wire through the single source-of-truth model_paths file.
    restoration["model_paths"] = str(args.model_paths)
    restoration["require_nifi_restorer"] = True
    restoration.pop("checkpoint", None)
    restoration.pop("model_config", None)
    cfg_payload["restoration"] = restoration

    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))
    print(f"[done] tuned config written: {out_cfg_path}")
    print(f"[done] sweep summary: {out_summary}")


if __name__ == "__main__":
    main()

