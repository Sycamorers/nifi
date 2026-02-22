#!/usr/bin/env python3
"""Prove Garden restoration is active and meaningfully better than compressed input."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import torch
import torch.nn.functional as F
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.metrics.perceptual_metrics import PerceptualMetrics
from nifi.metrics.simple_metrics import ssim
from nifi.utils.triptych import save_labeled_triptych
from scripts.garden_utils import (
    build_restorer,
    ensure_garden_artifact_scene,
    ensure_rate_dir,
    laplacian_variance,
    resolve_view_names,
    restore_eq7,
    save_abs_diff_map,
    save_minus1_1,
    to_minus1_1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Garden no-op proof script")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    p.add_argument("--config", type=str, default="configs/garden_known_good.yaml")
    p.add_argument("--rate", type=float, default=0.1)
    p.add_argument("--views", type=str, default="0,96", help="Comma-separated view indices from sorted shared list")
    p.add_argument("--t0", type=int, default=None)
    p.add_argument("--adapter_scale", type=float, default=None)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--detail_boost", type=float, default=None)
    p.add_argument("--detail_sigma", type=float, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/garden/prove_not_noop")
    p.add_argument("--min_abs_diff", type=float, default=0.01)
    p.add_argument("--min_lpips_gain", type=float, default=0.002)
    p.add_argument("--min_ssim_gain", type=float, default=0.0)
    return p.parse_args()


def _parse_views(raw: str) -> List[int]:
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError("At least one view index is required")
    return vals


def _metric_row(
    *,
    hq: torch.Tensor,
    lq: torch.Tensor,
    restored: torch.Tensor,
    perceptual: PerceptualMetrics,
) -> Dict[str, float]:
    hq01 = ((hq + 1.0) * 0.5).clamp(0.0, 1.0).float()
    lq01 = ((lq + 1.0) * 0.5).clamp(0.0, 1.0).float()
    rst01 = ((restored + 1.0) * 0.5).clamp(0.0, 1.0).float()

    try:
        lpips_before = float(perceptual.lpips(lq, hq).mean().item())
        lpips_after = float(perceptual.lpips(restored, hq).mean().item())
    except Exception:
        lpips_before = float("nan")
        lpips_after = float("nan")

    return {
        "mean_abs_diff_restored_vs_lq": float(torch.abs(restored - lq).mean().item()),
        "ssim_before": float(ssim(lq01, hq01)),
        "ssim_after": float(ssim(rst01, hq01)),
        "lpips_before": lpips_before,
        "lpips_after": lpips_after,
        "sharpness_hq": float(laplacian_variance(hq)),
        "sharpness_lq": float(laplacian_variance(lq)),
        "sharpness_restored": float(laplacian_variance(restored)),
    }


def _is_finite(x: float) -> bool:
    return x == x and x != float("inf") and x != float("-inf")


def _gaussian_blur(x01: torch.Tensor, sigma: float = 1.2, kernel_size: int = 5) -> torch.Tensor:
    sigma = max(1e-3, float(sigma))
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    half = (k - 1) / 2.0
    coords = torch.arange(k, device=x01.device, dtype=x01.dtype) - half
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g).view(1, 1, k, k)
    c = x01.shape[1]
    kernel = kernel_2d.expand(c, 1, k, k).contiguous()
    return F.conv2d(x01, kernel, padding=k // 2, groups=c)


def _inject_lq_detail(restored_minus1_1: torch.Tensor, lq_minus1_1: torch.Tensor, amount: float, sigma: float) -> torch.Tensor:
    if amount <= 0.0:
        return restored_minus1_1
    rst01 = ((restored_minus1_1 + 1.0) * 0.5).clamp(0.0, 1.0)
    lq01 = ((lq_minus1_1 + 1.0) * 0.5).clamp(0.0, 1.0)
    hi = lq01 - _gaussian_blur(lq01, sigma=sigma, kernel_size=5)
    fused = (rst01 + float(amount) * hi).clamp(0.0, 1.0)
    return fused * 2.0 - 1.0


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise SystemExit("garden_prove_not_noop.py requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; cannot run Garden no-op proof")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_t0 = 199
    cfg_scale = 1.0
    cfg_detail_boost = 0.0
    cfg_detail_sigma = 1.2
    cfg_path = ROOT / args.config
    if cfg_path.exists():
        payload = yaml.safe_load(cfg_path.read_text()) or {}
        restoration = payload.get("restoration", {}) if isinstance(payload, dict) else {}
        if isinstance(restoration, dict):
            cfg_t0 = int(restoration.get("t0", cfg_t0))
            cfg_scale = float(restoration.get("adapter_scale", cfg_scale))
            cfg_detail_boost = float(restoration.get("detail_boost", cfg_detail_boost))
            cfg_detail_sigma = float(restoration.get("detail_sigma", cfg_detail_sigma))

    t0 = int(cfg_t0 if args.t0 is None else args.t0)
    adapter_scale = float(cfg_scale if args.adapter_scale is None else args.adapter_scale)
    detail_boost = float(cfg_detail_boost if args.detail_boost is None else args.detail_boost)
    detail_sigma = float(cfg_detail_sigma if args.detail_sigma is None else args.detail_sigma)

    artifact_scene = ensure_garden_artifact_scene(dataset="mipnerf360", scene="garden", rates=[args.rate])
    rate_dir = ensure_rate_dir(artifact_scene, rate=float(args.rate))
    view_indices = _parse_views(args.views)
    view_names = resolve_view_names(artifact_scene, rate_dir=rate_dir, requested_indices=view_indices)

    restorer = build_restorer(model_paths=ROOT / args.model_paths, device_flag="cuda")
    perceptual = PerceptualMetrics(device=restorer.device)

    rows: List[Dict[str, object]] = []
    all_passed = True
    for idx, view_name in zip(view_indices, view_names):
        hq = to_minus1_1(artifact_scene / "clean" / view_name, device=restorer.device)
        lq = to_minus1_1(artifact_scene / rate_dir / view_name, device=restorer.device)
        restored = restore_eq7(
            restorer=restorer,
            lq_minus1_1=lq,
            prompt=args.prompt,
            t0=t0,
            adapter_scale=adapter_scale,
        )
        restored = _inject_lq_detail(restored, lq, amount=detail_boost, sigma=detail_sigma)

        metrics = _metric_row(hq=hq, lq=lq, restored=restored, perceptual=perceptual)
        lpips_gain = metrics["lpips_before"] - metrics["lpips_after"] if _is_finite(metrics["lpips_before"]) and _is_finite(metrics["lpips_after"]) else float("nan")
        ssim_gain = metrics["ssim_after"] - metrics["ssim_before"]
        sharp_gain = metrics["sharpness_restored"] - metrics["sharpness_lq"]

        diff_ok = metrics["mean_abs_diff_restored_vs_lq"] >= float(args.min_abs_diff)
        if _is_finite(lpips_gain):
            metric_ok = lpips_gain >= float(args.min_lpips_gain)
            metric_name = "lpips"
            metric_gain = lpips_gain
        else:
            metric_ok = ssim_gain >= float(args.min_ssim_gain)
            metric_name = "ssim"
            metric_gain = ssim_gain
        sharp_ok = sharp_gain >= 0.0
        passed = bool(diff_ok and metric_ok and sharp_ok)
        if not passed:
            all_passed = False

        base = out_dir / f"view_{idx:03d}"
        hq_path = base / "hq.png"
        lq_path = base / "lq.png"
        restored_path = base / "restored.png"
        triptych_path = base / "triptych.png"
        diff_path = base / "diff_restored_minus_lq_x10.png"
        save_minus1_1(hq_path, hq)
        save_minus1_1(lq_path, lq)
        save_minus1_1(restored_path, restored)
        save_abs_diff_map(diff_path, restored, lq, amplify=10.0)
        save_labeled_triptych(
            hq_path=hq_path,
            lq_path=lq_path,
            restored_path=restored_path,
            out_path=triptych_path,
            title=f"Garden view {view_name} | HQ | Compressed | Restored",
        )

        row = {
            "view_index": int(idx),
            "view_name": view_name,
            "metric_name": metric_name,
            "metric_gain": float(metric_gain),
            "detail_boost": detail_boost,
            "detail_sigma": detail_sigma,
            "lpips_gain": None if not _is_finite(lpips_gain) else float(lpips_gain),
            "ssim_gain": float(ssim_gain),
            "sharpness_gain": float(sharp_gain),
            "diff_ok": bool(diff_ok),
            "metric_ok": bool(metric_ok),
            "sharpness_ok": bool(sharp_ok),
            "passed": bool(passed),
            "metrics": metrics,
            "paths": {
                "hq": str(hq_path.resolve()),
                "lq": str(lq_path.resolve()),
                "restored": str(restored_path.resolve()),
                "triptych": str(triptych_path.resolve()),
                "diff": str(diff_path.resolve()),
            },
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    summary = {
        "status": "pass" if all_passed else "fail",
        "artifact_scene": str(artifact_scene),
        "rate_dir": rate_dir,
        "checkpoint": str(restorer.checkpoint_path),
        "model_config": str(restorer.model_config_path),
        "config": str(cfg_path.resolve()) if cfg_path.exists() else None,
        "t0": t0,
        "adapter_scale": adapter_scale,
        "detail_boost": detail_boost,
        "detail_sigma": detail_sigma,
        "thresholds": {
            "min_abs_diff": float(args.min_abs_diff),
            "min_lpips_gain": float(args.min_lpips_gain),
            "min_ssim_gain": float(args.min_ssim_gain),
            "sharpness_restored_ge_lq": True,
        },
        "rows": rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": str(summary_path.resolve()), "status": summary["status"]}, indent=2))

    if not all_passed:
        raise SystemExit("Garden no-op proof failed: restoration is too close to LQ or metrics/sharpness did not improve.")


if __name__ == "__main__":
    main()
