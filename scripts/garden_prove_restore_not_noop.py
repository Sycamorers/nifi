#!/usr/bin/env python3
"""Garden no-op proof with explicit disable/scale toggles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.utils.triptych import save_labeled_triptych
from scripts.garden_utils import (
    build_restorer,
    ensure_garden_artifact_scene,
    ensure_rate_dir,
    restore_eq7,
    save_abs_diff_map,
    save_minus1_1,
    to_minus1_1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prove Garden restoration is not a no-op")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    p.add_argument("--rate", type=float, default=0.1)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--t0", type=int, default=149)
    p.add_argument("--adapter_scale", type=float, default=0.5)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--disable_restore", action="store_true", help="Bypass restoration; restored == compressed")
    p.add_argument("--out_dir", type=str, default="outputs/garden/noop_check")
    p.add_argument("--min_abs_diff", type=float, default=0.01)
    return p.parse_args()


def _diff_stats(restored: torch.Tensor, lq: torch.Tensor) -> Dict[str, object]:
    diff = torch.abs(restored - lq).detach().cpu().float().numpy().reshape(-1)
    hist, edges = np.histogram(diff, bins=16, range=(0.0, float(diff.max()) if diff.size else 1.0))
    return {
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "std_abs_diff": float(np.std(diff)) if diff.size else 0.0,
        "p50_abs_diff": float(np.percentile(diff, 50.0)) if diff.size else 0.0,
        "p90_abs_diff": float(np.percentile(diff, 90.0)) if diff.size else 0.0,
        "p99_abs_diff": float(np.percentile(diff, 99.0)) if diff.size else 0.0,
        "hist_counts": hist.tolist(),
        "hist_edges": [float(x) for x in edges.tolist()],
    }


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise SystemExit("garden_prove_restore_not_noop.py requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; cannot run no-op proof")

    artifact_scene = ensure_garden_artifact_scene(dataset="mipnerf360", scene="garden", rates=[args.rate])
    rate_dir = ensure_rate_dir(artifact_scene, rate=float(args.rate))
    view_name = f"{int(args.view):05d}.png"
    hq_path = artifact_scene / "clean" / view_name
    lq_path = artifact_scene / rate_dir / view_name
    if not hq_path.exists() or not lq_path.exists():
        raise FileNotFoundError(f"Missing Garden view files for index {args.view} in {artifact_scene}")

    restorer = build_restorer(model_paths=ROOT / args.model_paths, device_flag="cuda")
    hq = to_minus1_1(hq_path, device=restorer.device)
    lq = to_minus1_1(lq_path, device=restorer.device)

    if args.disable_restore:
        restored = lq.clone()
    else:
        restored = restore_eq7(
            restorer=restorer,
            lq_minus1_1=lq,
            prompt=args.prompt,
            t0=int(args.t0),
            adapter_scale=float(args.adapter_scale),
        )

    stats = _diff_stats(restored, lq)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hq_out = out_dir / "hq.png"
    lq_out = out_dir / "compressed.png"
    rst_out = out_dir / "restored.png"
    triptych_out = out_dir / "triptych.png"
    diff_out = out_dir / "diff_restored_minus_compressed_x12.png"
    save_minus1_1(hq_out, hq)
    save_minus1_1(lq_out, lq)
    save_minus1_1(rst_out, restored)
    save_abs_diff_map(diff_out, restored, lq, amplify=12.0)
    save_labeled_triptych(
        hq_path=hq_out,
        lq_path=lq_out,
        restored_path=rst_out,
        out_path=triptych_out,
        title=f"Garden {view_name} | HQ | Compressed | Restored",
    )

    payload = {
        "status": "ok",
        "disable_restore": bool(args.disable_restore),
        "device": str(restorer.device),
        "gpu_name": torch.cuda.get_device_name(restorer.device),
        "checkpoint": str(restorer.checkpoint_path),
        "model_config": str(restorer.model_config_path),
        "rate_dir": rate_dir,
        "view_name": view_name,
        "t0": int(args.t0),
        "adapter_scale": float(args.adapter_scale),
        "diff_stats": stats,
        "paths": {
            "hq": str(hq_out.resolve()),
            "compressed": str(lq_out.resolve()),
            "restored": str(rst_out.resolve()),
            "triptych": str(triptych_out.resolve()),
            "diff_map": str(diff_out.resolve()),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    if args.disable_restore:
        # In bypass mode, enforce near-identical output to prove the toggle path works.
        if float(stats["mean_abs_diff"]) > 1e-6:
            raise SystemExit("disable_restore mode expected restored == compressed")
        return

    if float(stats["mean_abs_diff"]) < float(args.min_abs_diff):
        raise SystemExit(
            f"Restored output is too close to compressed: mean_abs_diff={stats['mean_abs_diff']:.6f} "
            f"< min_abs_diff={float(args.min_abs_diff):.6f}"
        )


if __name__ == "__main__":
    main()
