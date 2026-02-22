#!/usr/bin/env python3
"""GPU smoke sanity for Garden restoration forward pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.garden_utils import (
    build_restorer,
    ensure_garden_artifact_scene,
    ensure_rate_dir,
    resolve_view_names,
    restore_eq7,
    to_minus1_1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Garden CUDA smoke test")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    p.add_argument("--rate", type=float, default=0.1)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--t0", type=int, default=199)
    p.add_argument("--adapter_scale", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device != "cuda":
        raise SystemExit("garden_gpu_smoke.py requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; failing Garden GPU smoke")

    artifact_scene = ensure_garden_artifact_scene(dataset="mipnerf360", scene="garden", rates=[args.rate])
    rate_dir = ensure_rate_dir(artifact_scene, rate=float(args.rate))
    view_names = resolve_view_names(artifact_scene, rate_dir=rate_dir, requested_indices=[args.view])
    view_name = view_names[0]

    restorer = build_restorer(model_paths=ROOT / args.model_paths, device_flag="cuda")
    if restorer.device.type != "cuda":
        raise SystemExit(f"Expected CUDA device, got {restorer.device}")

    lq = to_minus1_1(artifact_scene / rate_dir / view_name, device=restorer.device)

    torch.cuda.reset_peak_memory_stats(restorer.device)
    torch.cuda.synchronize(restorer.device)
    t0 = time.perf_counter()
    for _ in range(max(1, int(args.passes))):
        _ = restore_eq7(
            restorer=restorer,
            lq_minus1_1=lq,
            prompt="",
            t0=int(args.t0),
            adapter_scale=float(args.adapter_scale),
        )
    torch.cuda.synchronize(restorer.device)
    elapsed = time.perf_counter() - t0

    payload = {
        "status": "ok",
        "device": str(restorer.device),
        "gpu_name": torch.cuda.get_device_name(restorer.device),
        "view_name": view_name,
        "passes": int(args.passes),
        "elapsed_sec": float(elapsed),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(restorer.device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(restorer.device)),
        "memory_peak_bytes": int(torch.cuda.max_memory_allocated(restorer.device)),
        "checkpoint": str(restorer.checkpoint_path),
        "model_config": str(restorer.model_config_path),
        "model_name_or_path": str(restorer.model_cfg.model_name_or_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
