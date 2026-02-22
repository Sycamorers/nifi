#!/usr/bin/env python3
"""Diagnose HQ render quality before restoration benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.metrics.simple_metrics import psnr, ssim


IMAGE_PATTERNS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
SUPPORTED_IMAGE_DIRS = ("images_8", "images_4", "images_2", "images", "rgb", "train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HQ quality diagnostic")
    parser.add_argument("--dataset", type=str, default="mipnerf360")
    parser.add_argument("--scene", type=str, required=True, help="Scene name, scene path, or artifact scene path")
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional GT image directory for PSNR/SSIM checks")
    parser.add_argument("--max_views", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--sharpness_warn_threshold", type=float, default=5e-4)
    return parser.parse_args()


def _laplacian_variance(img01: torch.Tensor) -> float:
    gray = img01.mean(dim=1, keepdim=True).float()
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return float(lap.var().item())


def _find_image_dir(scene_dir: Path) -> Path:
    if (scene_dir / "clean").exists():
        return scene_dir / "clean"
    for d in SUPPORTED_IMAGE_DIRS:
        p = scene_dir / d
        if p.exists() and p.is_dir():
            return p
    return scene_dir


def _resolve_scene_dir(dataset: str, scene_arg: str) -> Path:
    scene_path = Path(scene_arg)
    if scene_path.exists():
        return scene_path.resolve()
    artifact = ROOT / "artifacts" / scene_arg
    if artifact.exists():
        return artifact.resolve()
    dataset_scene = ROOT / "data" / dataset / scene_arg
    if dataset_scene.exists():
        return dataset_scene.resolve()
    raise FileNotFoundError(f"Could not resolve scene: {scene_arg}")


def _list_images(img_dir: Path) -> List[Path]:
    out: List[Path] = []
    for pattern in IMAGE_PATTERNS:
        out.extend(sorted(img_dir.glob(pattern)))
    return out


def _to_tensor01(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _match_gt(gt_dir: Optional[Path], name: str) -> Optional[Path]:
    if gt_dir is None:
        return None
    p = gt_dir / name
    if p.exists():
        return p
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg"):
        q = gt_dir / f"{stem}{ext}"
        if q.exists():
            return q
    return None


def main() -> None:
    args = parse_args()
    scene_dir = _resolve_scene_dir(args.dataset, args.scene)
    hq_dir = _find_image_dir(scene_dir)
    images = _list_images(hq_dir)
    if not images:
        raise RuntimeError(f"No images found under {hq_dir}")

    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir is not None else None
    selected = images[: max(1, int(args.max_views))]

    rows: List[Dict[str, object]] = []
    for src in selected:
        hq = _to_tensor01(src)
        sharp = _laplacian_variance(hq)
        row: Dict[str, object] = {
            "name": src.name,
            "resolution": [int(hq.shape[-2]), int(hq.shape[-1])],
            "sharpness_laplacian_var": sharp,
        }

        gt_path = _match_gt(gt_dir, src.name)
        if gt_path is not None:
            gt = _to_tensor01(gt_path)
            h = min(hq.shape[-2], gt.shape[-2])
            w = min(hq.shape[-1], gt.shape[-1])
            hq_c = hq[..., :h, :w]
            gt_c = gt[..., :h, :w]
            row["psnr_vs_gt"] = float(psnr(hq_c, gt_c))
            row["ssim_vs_gt"] = float(ssim(hq_c, gt_c))
            row["gt_path"] = str(gt_path)
        rows.append(row)

    mean_sharp = float(np.mean([r["sharpness_laplacian_var"] for r in rows]))
    payload: Dict[str, object] = {
        "scene_dir": str(scene_dir),
        "hq_dir": str(hq_dir),
        "num_images_total": len(images),
        "num_images_evaluated": len(rows),
        "mean_sharpness_laplacian_var": mean_sharp,
        "sharpness_warn_threshold": float(args.sharpness_warn_threshold),
        "hq_sharpness_ok": bool(mean_sharp >= float(args.sharpness_warn_threshold)),
        "gt_dir": None if gt_dir is None else str(gt_dir),
        "rows": rows,
        "notes": [
            "If hq_sharpness_ok is false, restoration quality is usually bottlenecked by HQ render/source quality.",
            "For 3DGS workflows, verify camera alignment, render resolution, and training quality of the 3DGS model.",
        ],
    }

    out_path = Path(args.out) if args.out else (ROOT / "outputs" / "diag_hq" / Path(args.scene).name / "hq_quality.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
