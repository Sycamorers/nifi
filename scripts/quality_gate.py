#!/usr/bin/env python3
"""Standalone restoration quality gate with hard pass/fail semantics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy as np
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.metrics.perceptual_metrics import PerceptualMetrics
from nifi.metrics.simple_metrics import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality gate on HQ/LQ/Restored outputs")
    parser.add_argument("--demo_dir", type=str, required=True, help="Directory containing hq/, lq/, restored/")
    parser.add_argument("--out", type=str, default=None, help="Output report path (JSON)")
    parser.add_argument("--min_lpips_margin", type=float, default=0.01)
    parser.add_argument("--min_dists_margin", type=float, default=0.005)
    parser.add_argument("--min_ssim_gain", type=float, default=0.0)
    parser.add_argument("--min_abs_diff", type=float, default=0.0, help="Minimum mean(|restored-lq|) to reject no-op outputs")
    parser.add_argument("--require_sharpness_non_decrease", action="store_true", default=True)
    parser.add_argument("--allow_sharpness_drop", action="store_true", help="Disable strict sharpness non-decrease check")
    parser.add_argument("--garden_mode", action="store_true", help="Enable stricter defaults for Garden quality checks")
    return parser.parse_args()


def _to_minus1_1(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device) * 2.0 - 1.0


def _lap_var(x: torch.Tensor) -> float:
    gray = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1, keepdim=True).float()
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return float(lap.var().item())


def _save_abs_delta(path: Path, restored: torch.Tensor, lq: torch.Tensor, amplify: float = 10.0) -> None:
    d = torch.abs(restored - lq).mean(dim=1).squeeze(0).detach().cpu().numpy()
    vis = np.clip(d * amplify, 0.0, 1.0)
    arr = np.clip(np.round(vis * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def _choose_metric(
    *,
    lpips_before: Optional[float],
    lpips_after: Optional[float],
    dists_before: Optional[float],
    dists_after: Optional[float],
    ssim_before: float,
    ssim_after: float,
    min_lpips_margin: float,
    min_dists_margin: float,
    min_ssim_gain: float,
) -> Tuple[str, float, bool]:
    if lpips_before is not None and lpips_after is not None:
        gain = float(lpips_before - lpips_after)
        return "lpips", gain, bool(gain >= min_lpips_margin)
    if dists_before is not None and dists_after is not None:
        gain = float(dists_before - dists_after)
        return "dists", gain, bool(gain >= min_dists_margin)
    gain = float(ssim_after - ssim_before)
    return "ssim", gain, bool(gain >= min_ssim_gain)


def main() -> None:
    args = parse_args()
    if args.allow_sharpness_drop:
        args.require_sharpness_non_decrease = False
    if args.garden_mode:
        args.min_lpips_margin = max(float(args.min_lpips_margin), 0.02)
        args.min_abs_diff = max(float(args.min_abs_diff), 0.01)
        args.require_sharpness_non_decrease = True

    demo_dir = Path(args.demo_dir)
    hq_dir = demo_dir / "hq"
    lq_dir = demo_dir / "lq"
    rst_dir = demo_dir / "restored"
    if not hq_dir.exists() or not lq_dir.exists() or not rst_dir.exists():
        raise FileNotFoundError(f"Expected hq/, lq/, restored/ under {demo_dir}")

    names = sorted({p.name for p in hq_dir.glob("*.png")} & {p.name for p in lq_dir.glob("*.png")} & {p.name for p in rst_dir.glob("*.png")})
    if not names:
        raise RuntimeError(f"No matching PNG triplets found in {demo_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    perceptual = None
    perceptual_error = None
    try:
        perceptual = PerceptualMetrics(device=device)
    except Exception as exc:
        perceptual_error = str(exc)

    rows: List[Dict[str, object]] = []
    all_passed = True
    delta_dir = demo_dir / "deltas"
    for name in names:
        hq = _to_minus1_1(hq_dir / name, device=device)
        lq = _to_minus1_1(lq_dir / name, device=device)
        rst = _to_minus1_1(rst_dir / name, device=device)

        lp_b = lp_a = ds_b = ds_a = None
        if perceptual is not None:
            try:
                lp_b = float(perceptual.lpips(lq, hq).item())
                lp_a = float(perceptual.lpips(rst, hq).item())
                ds_b = float(perceptual.dists(lq, hq).item())
                ds_a = float(perceptual.dists(rst, hq).item())
            except Exception:
                lp_b = lp_a = ds_b = ds_a = None

        lq01 = ((lq + 1.0) * 0.5).float()
        hq01 = ((hq + 1.0) * 0.5).float()
        rst01 = ((rst + 1.0) * 0.5).float()
        ssim_before = float(ssim(lq01, hq01))
        ssim_after = float(ssim(rst01, hq01))
        sharp_before = _lap_var(lq)
        sharp_after = _lap_var(rst)
        mean_abs_diff = float(torch.abs(rst - lq).mean().item())

        metric_name, gain, metric_ok = _choose_metric(
            lpips_before=lp_b,
            lpips_after=lp_a,
            dists_before=ds_b,
            dists_after=ds_a,
            ssim_before=ssim_before,
            ssim_after=ssim_after,
            min_lpips_margin=float(args.min_lpips_margin),
            min_dists_margin=float(args.min_dists_margin),
            min_ssim_gain=float(args.min_ssim_gain),
        )
        sharp_ok = (sharp_after >= sharp_before) if args.require_sharpness_non_decrease else True
        diff_ok = mean_abs_diff >= float(args.min_abs_diff)
        passed = bool(metric_ok and sharp_ok and diff_ok)
        if not passed:
            all_passed = False

        _save_abs_delta(delta_dir / name, restored=rst, lq=lq, amplify=10.0)
        rows.append(
            {
                "view": name,
                "lpips_before": lp_b,
                "lpips_after": lp_a,
                "dists_before": ds_b,
                "dists_after": ds_a,
                "ssim_before": ssim_before,
                "ssim_after": ssim_after,
                "mean_abs_diff_restored_vs_lq": mean_abs_diff,
                "sharpness_before": sharp_before,
                "sharpness_after": sharp_after,
                "metric_name": metric_name,
                "metric_gain": gain,
                "metric_ok": metric_ok,
                "sharpness_ok": sharp_ok,
                "diff_ok": diff_ok,
                "passed": passed,
                "delta_path": str((delta_dir / name).resolve()),
            }
        )

    summary = {
        "demo_dir": str(demo_dir.resolve()),
        "num_views": len(rows),
        "all_passed": bool(all_passed),
        "lpips_available": any(r["lpips_before"] is not None for r in rows),
        "dists_available": any(r["dists_before"] is not None for r in rows),
        "perceptual_init_error": perceptual_error,
        "thresholds": {
            "min_lpips_margin": float(args.min_lpips_margin),
            "min_dists_margin": float(args.min_dists_margin),
            "min_ssim_gain": float(args.min_ssim_gain),
            "min_abs_diff": float(args.min_abs_diff),
            "require_sharpness_non_decrease": bool(args.require_sharpness_non_decrease),
            "garden_mode": bool(args.garden_mode),
        },
        "rows": rows,
    }

    out_path = Path(args.out) if args.out else (demo_dir / "quality_gate_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("view | metric | gain | diff | sharpness(before->after) | pass")
    for row in rows:
        print(
            f"{row['view']} | {row['metric_name']} | {row['metric_gain']:.6f} | "
            f"{row['mean_abs_diff_restored_vs_lq']:.6f} | "
            f"{row['sharpness_before']:.6f}->{row['sharpness_after']:.6f} | {row['passed']}"
        )

    print(f"[done] report: {out_path}")
    if not all_passed:
        print("[fail] quality gate failed")
        raise SystemExit(1)
    print("[pass] quality gate passed")


if __name__ == "__main__":
    main()
