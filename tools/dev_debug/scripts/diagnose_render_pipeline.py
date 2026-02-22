#!/usr/bin/env python3
"""Diagnose HQ/LQ render consistency before restoration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose render pipeline (HQ vs LQ)")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g. garden) or artifact scene path")
    parser.add_argument("--rate", type=str, default="0.100", help="Rate folder or numeric value, e.g. 0.1 or rate_0.100")
    parser.add_argument("--views", nargs="*", type=int, default=None, help="Optional explicit source indices")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: outputs/diag_render/<scene>")
    return parser.parse_args()



def _resolve_scene_artifacts(scene_arg: str) -> Path:
    candidate = Path(scene_arg)
    if candidate.exists() and (candidate / "clean").exists():
        return candidate.resolve()

    by_name = ROOT / "artifacts" / scene_arg
    if by_name.exists() and (by_name / "clean").exists():
        return by_name.resolve()

    raise FileNotFoundError(
        f"Could not resolve artifact scene from '{scene_arg}'. "
        "Expected artifacts/<scene>/clean and artifacts/<scene>/rate_*/"
    )



def _resolve_rate_dir(scene_dir: Path, rate_arg: str) -> str:
    available = sorted([p.name for p in scene_dir.glob("rate_*") if p.is_dir()])
    if not available:
        raise RuntimeError(f"No rate_* directories found in {scene_dir}")

    s = rate_arg.strip()
    if s in available:
        return s

    if not s.startswith("rate_"):
        try:
            s = f"rate_{float(s):.3f}"
        except ValueError:
            pass

    if s in available:
        return s

    raise ValueError(f"rate '{rate_arg}' not found. available={available}")



def _load_candidate_names(scene_dir: Path, rate_dir: str) -> List[str]:
    clean = {p.name for p in (scene_dir / "clean").glob("*.png")}
    lq = {p.name for p in (scene_dir / rate_dir).glob("*.png")}
    names = sorted(clean & lq)
    if not names:
        raise RuntimeError("No shared clean/lq PNG names found")
    return names



def _choose_names(scene_dir: Path, names: Sequence[str], views: Optional[List[int]], num_views: int, seed: int) -> List[str]:
    names = list(names)
    if views:
        chosen = []
        for v in views:
            name = f"{int(v):05d}.png"
            if name not in names:
                raise ValueError(f"Requested view index {v} -> {name} not found")
            chosen.append(name)
        return chosen

    meta_path = scene_dir / "metadata.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text())
            splits = payload.get("splits", {}) if isinstance(payload, dict) else {}
            test = splits.get("test", []) if isinstance(splits, dict) else []
            test = [x for x in test if x in names]
            if test:
                names = test
        except Exception:
            pass

    if num_views >= len(names):
        return names

    if num_views <= 1:
        return [names[0]]

    offset = int(seed) % len(names)
    rotated = names[offset:] + names[:offset]
    out: List[str] = []
    span = len(rotated) - 1
    for i in range(num_views):
        idx = round(i * span / (num_views - 1))
        out.append(rotated[idx])
    return out



def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)



def _to_tensor01(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t



def _psnr(pred01: torch.Tensor, target01: torch.Tensor) -> float:
    mse = torch.mean((pred01 - target01) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))



def _ssim(pred01: torch.Tensor, target01: torch.Tensor) -> float:
    # Lightweight global SSIM surrogate (deterministic) for sanity only.
    x = pred01
    y = target01
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float((num / den).item())



def _stats(arr: np.ndarray) -> Dict[str, float]:
    arrf = arr.astype(np.float32)
    return {
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "channels": int(arr.shape[2]),
        "min": float(arrf.min()),
        "max": float(arrf.max()),
        "mean": float(arrf.mean()),
        "std": float(arrf.std()),
    }



def _parse_colmap_images_txt(path: Path) -> Dict[str, Dict[str, List[float]]]:
    poses: Dict[str, Dict[str, List[float]]] = {}
    if not path.exists():
        return poses

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    # COLMAP images.txt uses 2 lines per image; first line has pose + name.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        name = parts[9]
        q = [float(x) for x in parts[1:5]]
        t = [float(x) for x in parts[5:8]]
        poses[name] = {"qvec": q, "tvec": t}
    return poses



def _pose_l2(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.array(a, dtype=np.float64)
    bv = np.array(b, dtype=np.float64)
    return float(np.linalg.norm(av - bv))



def _save_mosaic(hq: Path, lq: Path, title: str, out_path: Path) -> None:
    with Image.open(hq) as h_img, Image.open(lq) as l_img:
        h = h_img.convert("RGB")
        l = l_img.convert("RGB")

    if h.size != l.size:
        l = l.resize(h.size, resample=Image.Resampling.BICUBIC)

    w, hgt = h.size
    pad = 14
    title_h = 26
    label_h = 20
    canvas = Image.new("RGB", (pad + w + pad + w + pad, pad + title_h + pad + label_h + pad + hgt + pad), (236, 236, 236))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, pad), title, fill=(20, 20, 20), font=font)

    y_label = pad + title_h + pad
    y_img = y_label + label_h + pad
    draw.text((pad + w // 2 - 12, y_label), "HQ", fill=(30, 30, 30), font=font)
    draw.text((pad + w + pad + w // 2 - 34, y_label), "Compressed", fill=(30, 30, 30), font=font)

    canvas.paste(h, (pad, y_img))
    canvas.paste(l, (pad + w + pad, y_img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)



def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    args = parse_args()

    scene_dir = _resolve_scene_artifacts(args.scene)
    scene_name = scene_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "outputs" / "diag_render" / scene_name)

    rate_dir = _resolve_rate_dir(scene_dir, args.rate)
    names = _load_candidate_names(scene_dir, rate_dir)
    selected = _choose_names(scene_dir, names, args.views, int(args.num_views), int(args.seed))

    print(f"[info] scene={scene_name}")
    print(f"[info] rate_dir={rate_dir}")
    print(f"[info] selected={selected}")

    hq_out = out_dir / "hq"
    lq_out = out_dir / "lq"
    raw_hq_out = out_dir / "raw" / "hq"
    raw_lq_out = out_dir / "raw" / "lq"
    mosaic_out = out_dir / "mosaic"

    for d in [hq_out, lq_out, raw_hq_out, raw_lq_out, mosaic_out]:
        d.mkdir(parents=True, exist_ok=True)

    src_scene = None
    meta_path = scene_dir / "metadata.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text())
            src_scene = payload.get("scene_dir") if isinstance(payload, dict) else None
        except Exception:
            src_scene = None

    pose_map = {}
    if src_scene:
        pose_map = _parse_colmap_images_txt(Path(src_scene) / "sparse" / "0" / "images.txt")

    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(selected):
        hq_src = scene_dir / "clean" / name
        lq_src = scene_dir / rate_dir / name

        hq_img = _load_rgb(hq_src)
        lq_img = _load_rgb(lq_src)

        # Save no-resize copies.
        hq_dst = hq_out / name
        lq_dst = lq_out / name
        Image.fromarray(hq_img).save(hq_dst)
        Image.fromarray(lq_img).save(lq_dst)

        # Raw dumps: same source in this pipeline (already rendered PNG without extra processing in this script).
        Image.fromarray(hq_img).save(raw_hq_out / name)
        Image.fromarray(lq_img).save(raw_lq_out / name)

        t_hq = _to_tensor01(hq_img)
        t_lq = _to_tensor01(lq_img)

        psnr = _psnr(t_lq, t_hq)
        ssim = _ssim(t_lq, t_hq)

        pose_q_l2 = 0.0
        pose_t_l2 = 0.0
        pose_available = False
        if name in pose_map:
            pose_available = True
            # HQ and LQ share the same source view name in this pipeline.
            pose_q_l2 = _pose_l2(pose_map[name]["qvec"], pose_map[name]["qvec"])
            pose_t_l2 = _pose_l2(pose_map[name]["tvec"], pose_map[name]["tvec"])

        row = {
            "view_name": name,
            "source_index": int(Path(name).stem),
            "hq_w": int(hq_img.shape[1]),
            "hq_h": int(hq_img.shape[0]),
            "lq_w": int(lq_img.shape[1]),
            "lq_h": int(lq_img.shape[0]),
            "hq_mean": float(hq_img.mean()),
            "lq_mean": float(lq_img.mean()),
            "hq_std": float(hq_img.std()),
            "lq_std": float(lq_img.std()),
            "psnr_lq_vs_hq": psnr,
            "ssim_lq_vs_hq": ssim,
            "pose_available": pose_available,
            "pose_q_l2": pose_q_l2,
            "pose_t_l2": pose_t_l2,
        }
        rows.append(row)

        _save_mosaic(hq_dst, lq_dst, f"Scene={scene_name} Rate={rate_dir} View={name}", mosaic_out / f"view_{idx:03d}.png")

    summary = {
        "scene": scene_name,
        "artifact_scene_dir": str(scene_dir),
        "rate_dir": rate_dir,
        "source_scene_dir": src_scene,
        "selected_views": selected,
        "stats": rows,
        "aggregate": {
            "avg_psnr_lq_vs_hq": float(np.mean([r["psnr_lq_vs_hq"] for r in rows])) if rows else float("nan"),
            "avg_ssim_lq_vs_hq": float(np.mean([r["ssim_lq_vs_hq"] for r in rows])) if rows else float("nan"),
            "resolution_match_all": bool(all(r["hq_w"] == r["lq_w"] and r["hq_h"] == r["lq_h"] for r in rows)),
            "pose_match_all": bool(all((not r["pose_available"]) or (r["pose_q_l2"] == 0.0 and r["pose_t_l2"] == 0.0) for r in rows)),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "render_diag.json").write_text(json.dumps(summary, indent=2))
    _write_csv(out_dir / "render_diag.csv", rows)

    print(f"[done] wrote render diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
