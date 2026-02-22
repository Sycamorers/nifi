#!/usr/bin/env python3
"""Garden-only hyperparameter sweep for t0/adapter scale/rate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy as np
import torch
import yaml
import torch.nn.functional as F

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.metrics.perceptual_metrics import PerceptualMetrics
from nifi.metrics.simple_metrics import ssim
from scripts.garden_utils import (
    build_restorer,
    ensure_garden_artifact_scene,
    ensure_rate_dir,
    laplacian_variance,
    restore_eq7,
    to_minus1_1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-tune Garden restoration hparams")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    p.add_argument("--rates", type=str, default="0.1")
    p.add_argument("--t0_values", type=str, default="99,149,199")
    p.add_argument("--adapter_scales", type=str, default="0.5,0.75,1.0")
    p.add_argument("--detail_boosts", type=str, default="0.0,0.5,1.0")
    p.add_argument("--detail_sigma", type=float, default=1.2)
    p.add_argument("--views", type=str, default="0,64,184")
    p.add_argument("--min_lpips_gain", type=float, default=0.02)
    p.add_argument("--min_abs_diff", type=float, default=0.01)
    p.add_argument("--out_summary", type=str, default="outputs/garden/auto_tune_summary.json")
    p.add_argument("--out_config", type=str, default="configs/garden_known_good.yaml")
    return p.parse_args()


def _parse_float_list(raw: str) -> List[float]:
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError(f"Empty numeric list: {raw}")
    return vals


def _parse_int_list(raw: str) -> List[int]:
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError(f"Empty integer list: {raw}")
    return vals


def _row_metrics(
    *,
    hq: torch.Tensor,
    lq: torch.Tensor,
    restored: torch.Tensor,
    perceptual: PerceptualMetrics,
) -> Dict[str, float]:
    lpips_before = float(perceptual.lpips(lq, hq).mean().item())
    lpips_after = float(perceptual.lpips(restored, hq).mean().item())

    hq01 = ((hq + 1.0) * 0.5).clamp(0.0, 1.0).float()
    lq01 = ((lq + 1.0) * 0.5).clamp(0.0, 1.0).float()
    rst01 = ((restored + 1.0) * 0.5).clamp(0.0, 1.0).float()
    ssim_before = float(ssim(lq01, hq01))
    ssim_after = float(ssim(rst01, hq01))
    sharp_lq = float(laplacian_variance(lq))
    sharp_restored = float(laplacian_variance(restored))
    return {
        "lpips_before": lpips_before,
        "lpips_after": lpips_after,
        "lpips_gain": lpips_before - lpips_after,
        "ssim_before": ssim_before,
        "ssim_after": ssim_after,
        "ssim_gain": float(ssim_after - ssim_before),
        "sharpness_lq": sharp_lq,
        "sharpness_restored": sharp_restored,
        "sharpness_gain": float(sharp_restored - sharp_lq),
        "mean_abs_diff_restored_vs_lq": float(torch.abs(restored - lq).mean().item()),
    }


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
        raise RuntimeError("auto_tune_garden.py must run with --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    rates = _parse_float_list(args.rates)
    t0_values = _parse_int_list(args.t0_values)
    scales = _parse_float_list(args.adapter_scales)
    detail_boosts = _parse_float_list(args.detail_boosts)
    views = _parse_int_list(args.views)

    artifact_scene = ensure_garden_artifact_scene(dataset="mipnerf360", scene="garden", rates=rates)
    restorer = build_restorer(model_paths=ROOT / args.model_paths, device_flag="cuda")
    perceptual = PerceptualMetrics(device=restorer.device)

    candidates: List[Dict[str, object]] = []
    for rate in rates:
        rate_dir = ensure_rate_dir(artifact_scene, rate=rate)
        for t0 in t0_values:
            for scale in scales:
                for detail_boost in detail_boosts:
                    rows: List[Dict[str, float]] = []
                    for v in views:
                        name = f"{int(v):05d}.png"
                        hq = to_minus1_1(artifact_scene / "clean" / name, device=restorer.device)
                        lq = to_minus1_1(artifact_scene / rate_dir / name, device=restorer.device)
                        restored = restore_eq7(
                            restorer=restorer,
                            lq_minus1_1=lq,
                            prompt="",
                            t0=int(t0),
                            adapter_scale=float(scale),
                        )
                        restored = _inject_lq_detail(restored, lq, amount=float(detail_boost), sigma=float(args.detail_sigma))
                        rows.append(_row_metrics(hq=hq, lq=lq, restored=restored, perceptual=perceptual))

                    lpips_gain = float(np.mean([r["lpips_gain"] for r in rows]))
                    ssim_gain = float(np.mean([r["ssim_gain"] for r in rows]))
                    sharp_gain = float(np.mean([r["sharpness_gain"] for r in rows]))
                    mean_abs_diff = float(np.mean([r["mean_abs_diff_restored_vs_lq"] for r in rows]))
                    passed = bool(
                        lpips_gain >= float(args.min_lpips_gain)
                        and sharp_gain >= 0.0
                        and mean_abs_diff >= float(args.min_abs_diff)
                    )
                    score = float(lpips_gain + 0.25 * ssim_gain + 0.10 * sharp_gain)
                    candidates.append(
                        {
                            "rate": float(rate),
                            "rate_dir": rate_dir,
                            "t0": int(t0),
                            "adapter_scale": float(scale),
                            "detail_boost": float(detail_boost),
                            "detail_sigma": float(args.detail_sigma),
                            "mean_lpips_gain": lpips_gain,
                            "mean_ssim_gain": ssim_gain,
                            "mean_sharpness_gain": sharp_gain,
                            "mean_abs_diff_restored_vs_lq": mean_abs_diff,
                            "score": score,
                            "passed": passed,
                            "rows": rows,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "rate": rate,
                                "t0": t0,
                                "scale": scale,
                                "detail_boost": detail_boost,
                                "lpips_gain": lpips_gain,
                                "ssim_gain": ssim_gain,
                                "sharp_gain": sharp_gain,
                                "mean_abs_diff": mean_abs_diff,
                                "passed": passed,
                            }
                        )
                    )

    passing = [c for c in candidates if bool(c["passed"])]
    ranked = sorted(passing if passing else candidates, key=lambda x: float(x["score"]), reverse=True)
    best = ranked[0]
    summary = {
        "status": "ok",
        "used_passing_only": bool(passing),
        "num_candidates": len(candidates),
        "num_passing": len(passing),
        "best": best,
        "top5": ranked[:5],
        "checkpoint": str(restorer.checkpoint_path),
        "model_config": str(restorer.model_config_path),
        "artifact_scene": str(artifact_scene),
        "views": views,
        "thresholds": {
            "min_lpips_gain": float(args.min_lpips_gain),
            "min_abs_diff": float(args.min_abs_diff),
            "sharpness_non_decrease": True,
        },
    }

    out_summary = ROOT / args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))

    config_payload = {
        "dataset": {"name": "garden", "scene": "garden"},
        "demo": {
            "rate_presets": {
                "selected": float(best["rate"]),
                "low": 0.1,
                "medium": 0.5,
                "high": 1.0,
            },
            "default_rate": "selected",
            "auto_num_views": 4,
            "prefer_test_split": True,
        },
        "restoration": {
            "prompt": "",
            "t0": int(best["t0"]),
            "adapter_scale": float(best["adapter_scale"]),
            "detail_boost": float(best.get("detail_boost", 0.0)),
            "detail_sigma": float(best.get("detail_sigma", args.detail_sigma)),
            "final_output": "nifi_eq7",
            "candidate_order": ["nifi_eq7"],
            "model_paths": str(args.model_paths),
            "require_nifi_restorer": True,
        },
        "quality_gate": {
            "enabled": True,
            "fail_on_violation": True,
            "min_lpips_margin": max(0.02, float(args.min_lpips_gain) * 0.5),
            "min_dists_margin": 0.0,
            "min_ssim_gain": 0.0,
            "require_sharpness_non_decrease": True,
            "min_abs_diff": float(args.min_abs_diff),
        },
        "compression": {
            "jpeg_quality_min": 8,
            "jpeg_quality_max": 55,
            "downsample_min": 2,
            "downsample_max": 8,
        },
        "selection": {
            "chosen_rate": float(best["rate"]),
            "chosen_t0": int(best["t0"]),
            "chosen_scale": float(best["adapter_scale"]),
            "chosen_score": float(best["score"]),
            "chosen_lpips_gain": float(best["mean_lpips_gain"]),
            "chosen_ssim_gain": float(best["mean_ssim_gain"]),
            "chosen_sharpness_gain": float(best["mean_sharpness_gain"]),
            "chosen_detail_boost": float(best.get("detail_boost", 0.0)),
            "chosen_detail_sigma": float(best.get("detail_sigma", args.detail_sigma)),
            "used_passing_only": bool(passing),
        },
    }

    out_config = ROOT / args.out_config
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(config_payload, sort_keys=False))

    print(
        json.dumps(
            {
                "summary": str(out_summary.resolve()),
                "config": str(out_config.resolve()),
                "best_rate": best["rate"],
                "best_t0": best["t0"],
                "best_scale": best["adapter_scale"],
                "best_detail_boost": best.get("detail_boost", 0.0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
