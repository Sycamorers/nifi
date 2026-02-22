#!/usr/bin/env python3
"""Sweep t0 and adapter strength to identify less-blurry restoration settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import torch
from torch import autocast
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config
from nifi.utils.runtime import configure_runtime, get_runtime_defaults, resolve_device
from nifi.metrics.simple_metrics import psnr, ssim



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep t0 and adapter scale for restoration quality")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--rate", type=str, default="0.100")
    parser.add_argument("--views", nargs="*", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    parser.add_argument("--t0_values", type=str, default="99,149,199,249,299")
    parser.add_argument("--adapter_scales", type=str, default="0.0,0.25,0.5,0.75,1.0")

    parser.add_argument("--out_dir", type=str, default=None, help="Default: outputs/sweep/<scene>")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()



def _resolve_scene_artifacts(scene_arg: str) -> Path:
    p = Path(scene_arg)
    if p.exists() and (p / "clean").exists():
        return p.resolve()
    p2 = ROOT / "artifacts" / scene_arg
    if p2.exists() and (p2 / "clean").exists():
        return p2.resolve()
    raise FileNotFoundError(f"Could not resolve artifact scene from '{scene_arg}'")



def _resolve_rate_dir(scene_dir: Path, rate_arg: str) -> str:
    available = sorted([p.name for p in scene_dir.glob("rate_*") if p.is_dir()])
    if not available:
        raise RuntimeError(f"No rate_* directories in {scene_dir}")

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



def _find_default_ckpt() -> Optional[Path]:
    runs = ROOT / "runs"
    if not runs.exists():
        return None
    candidates = list(runs.glob("**/best.pt")) + list(runs.glob("**/latest.pt"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]



def _pick_device(flag: str) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"



def _to_minus1_1(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return t * 2.0 - 1.0



def _save_minus1_1(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)



def _pad_to_multiple(x: torch.Tensor, m: int = 8) -> Tuple[torch.Tensor, Dict[str, int]]:
    _, _, h, w = x.shape
    pad_h = (m - (h % m)) % m
    pad_w = (m - (w % m)) % m
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    if pad_h == 0 and pad_w == 0:
        return x, {"left": 0, "right": 0, "top": 0, "bottom": 0}
    return torch.nn.functional.pad(x, (left, right, top, bottom), mode="reflect"), {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }



def _unpad(x: torch.Tensor, pads: Dict[str, int]) -> torch.Tensor:
    l, r, t, b = pads["left"], pads["right"], pads["top"], pads["bottom"]
    if l == r == t == b == 0:
        return x
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]



def _run_restore(
    model: FrozenBackboneArtifactRestorationModel,
    lq: torch.Tensor,
    t0: int,
    adapter_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    b = lq.shape[0]
    t = torch.full((b,), int(t0), device=lq.device, dtype=torch.long)
    with torch.no_grad(), autocast(device_type=lq.device.type, dtype=amp_dtype, enabled=use_amp):
        z = model.encode_images(lq)
        z_t0, _ = model.q_sample(z, t, noise=torch.zeros_like(z))
        eps_base = model.predict_eps(z_t0, t, [""], adapter_type=None, train_mode=False)
        eps_minus = model.predict_eps(z_t0, t, [""], adapter_type="minus", train_mode=False)
        eps_used = eps_base + float(adapter_scale) * (eps_minus - eps_base)
        sigma = model.sigma(t).view(-1, 1, 1, 1)
        z_hat = z_t0 - sigma * eps_used
        out = model.decode_latents(z_hat)
    return out



def _sharpness_laplacian_var(x_minus1_1: torch.Tensor) -> float:
    img = ((x_minus1_1.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).float().squeeze(0).permute(1, 2, 0).numpy()
    gray = img.mean(axis=2, dtype=np.float32).astype(np.float32, copy=False)
    lap = ndimage.laplace(gray)
    return float(np.var(lap))



def _build_lpips_model(device: torch.device):
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="vgg").to(device)
        model.eval()
        return model
    except Exception:
        return None


def _try_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> Optional[float]:
    if lpips_model is None:
        return None
    with torch.no_grad():
        return float(lpips_model(pred, target).mean().item())



def _parse_int_list(csv_values: str) -> List[int]:
    out = []
    for x in csv_values.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("no values parsed")
    return out



def _parse_float_list(csv_values: str) -> List[float]:
    out = []
    for x in csv_values.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise ValueError("no values parsed")
    return out



def _save_grid_montage(
    rows: List[Dict[str, object]],
    t0_values: List[int],
    scales: List[float],
    out_path: Path,
) -> None:
    # Build from first view only for concise debug visualization.
    first_view = rows[0]["view_name"] if rows else None
    if first_view is None:
        return

    lookup = {(r["view_name"], r["t0"], r["adapter_scale"]): Path(r["image_path"]) for r in rows}

    sample = None
    for key, path in lookup.items():
        if key[0] == first_view and path.exists():
            sample = path
            break
    if sample is None:
        return

    with Image.open(sample) as im:
        base = im.convert("RGB")

    w, h = base.size
    pad = 10
    label_h = 18
    title_h = 22
    grid_w = pad + len(t0_values) * (w + pad)
    grid_h = pad + title_h + pad + label_h + pad + len(scales) * (h + pad)

    canvas = Image.new("RGB", (grid_w, grid_h), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, pad), f"t0 / adapter_scale sweep (view={first_view})", fill=(25, 25, 25), font=font)

    y_headers = pad + title_h + pad
    for c, t0 in enumerate(t0_values):
        x = pad + c * (w + pad)
        draw.text((x + 4, y_headers), f"t0={t0}", fill=(35, 35, 35), font=font)

    y0 = y_headers + label_h + pad
    for r_idx, scale in enumerate(scales):
        y = y0 + r_idx * (h + pad)
        draw.text((2, y + 2), f"s={scale:.2f}", fill=(35, 35, 35), font=font)
        for c, t0 in enumerate(t0_values):
            x = pad + c * (w + pad)
            path = lookup.get((first_view, t0, scale))
            if path is None or not path.exists():
                continue
            with Image.open(path) as im:
                img = im.convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), resample=Image.Resampling.BICUBIC)
            canvas.paste(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)



def main() -> None:
    args = parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    scene_dir = _resolve_scene_artifacts(args.scene)
    scene_name = scene_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "outputs" / "sweep" / scene_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rate_dir = _resolve_rate_dir(scene_dir, args.rate)

    ckpt_path = Path(args.ckpt) if args.ckpt else _find_default_ckpt()
    if ckpt_path is None or not ckpt_path.exists():
        raise RuntimeError("No checkpoint found. Pass --ckpt runs/<exp>/best.pt")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    runtime_cfg = get_runtime_defaults()
    runtime_cfg.update(cfg.get("runtime", {}))
    runtime_cfg["device"] = _pick_device(args.device)
    configure_runtime(runtime_cfg)
    device = resolve_device(runtime_cfg)

    mp = cfg.get("train", {}).get("mixed_precision", "fp16")
    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    use_amp = device.type == "cuda" and mp in {"bf16", "fp16"}

    ckpt = load_checkpoint(str(ckpt_path), map_location="cpu")

    model_cfg = ArtifactRestorationDiffusionConfig(
        model_name_or_path=cfg["model"]["pretrained_model_name_or_path"],
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        lora_rank=int(cfg["model"]["lora_rank"]),
        guidance_scale=float(cfg["model"]["guidance_scale"]),
        prompt_dropout=float(cfg["model"]["prompt_dropout"]),
        max_token_length=int(cfg["model"]["max_token_length"]),
        vae_scaling_factor=float(cfg["model"]["vae_scaling_factor"]),
    )

    model = FrozenBackboneArtifactRestorationModel(model_cfg, device=device, dtype=amp_dtype if use_amp else torch.float32)
    model.phi_minus.load_state_dict(ckpt["phi_minus"])
    model.phi_plus.load_state_dict(ckpt["phi_plus"])
    model.freeze_backbone()
    model.eval()

    t0_values = _parse_int_list(args.t0_values)
    scales = _parse_float_list(args.adapter_scales)

    view_names = [f"{int(v):05d}.png" for v in args.views]
    lpips_model = _build_lpips_model(device)

    rows: List[Dict[str, object]] = []
    lpips_available = None

    for view_name in view_names:
        hq_path = scene_dir / "clean" / view_name
        lq_path = scene_dir / rate_dir / view_name
        if not hq_path.exists() or not lq_path.exists():
            raise FileNotFoundError(f"Missing view files for {view_name}")

        hq = _to_minus1_1(hq_path, device=device)
        lq = _to_minus1_1(lq_path, device=device)

        lq_pad, pads = _pad_to_multiple(lq, m=8)
        hq_pad, _ = _pad_to_multiple(hq, m=8)

        # Baseline metrics for compressed input.
        lq_01 = ((lq + 1.0) * 0.5).float()
        hq_01 = ((hq + 1.0) * 0.5).float()
        baseline_psnr = float(psnr(lq_01, hq_01))
        baseline_ssim = float(ssim(lq_01, hq_01))
        baseline_lpips = _try_lpips(lq, hq, lpips_model)
        baseline_sharp = _sharpness_laplacian_var(lq)
        if lpips_available is None:
            lpips_available = baseline_lpips is not None

        for t0 in t0_values:
            for scale in scales:
                restored_pad = _run_restore(model, lq_pad, t0=t0, adapter_scale=scale, use_amp=use_amp, amp_dtype=amp_dtype)
                restored = _unpad(restored_pad, pads)

                restored_01 = ((restored + 1.0) * 0.5).float()
                m_psnr = float(psnr(restored_01, hq_01))
                m_ssim = float(ssim(restored_01, hq_01))
                m_lpips = _try_lpips(restored, hq, lpips_model)
                sharp = _sharpness_laplacian_var(restored)

                img_path = out_dir / "images" / f"{Path(view_name).stem}_t{t0:04d}_s{scale:.2f}.png"
                _save_minus1_1(img_path, restored)

                row = {
                    "view_name": view_name,
                    "rate_dir": rate_dir,
                    "t0": int(t0),
                    "adapter_scale": float(scale),
                    "psnr_before": baseline_psnr,
                    "ssim_before": baseline_ssim,
                    "lpips_before": baseline_lpips,
                    "sharpness_before": baseline_sharp,
                    "psnr_after": m_psnr,
                    "ssim_after": m_ssim,
                    "lpips_after": m_lpips,
                    "sharpness_after": sharp,
                    "delta_psnr": m_psnr - baseline_psnr,
                    "delta_ssim": m_ssim - baseline_ssim,
                    "delta_lpips": (m_lpips - baseline_lpips) if (m_lpips is not None and baseline_lpips is not None) else None,
                    "delta_sharpness": sharp - baseline_sharp,
                    "improved_psnr": m_psnr > baseline_psnr,
                    "improved_ssim": m_ssim > baseline_ssim,
                    "improved_lpips": (m_lpips < baseline_lpips) if (m_lpips is not None and baseline_lpips is not None) else None,
                    "image_path": str(img_path),
                }
                rows.append(row)

    # Ranking: LPIPS (if available) then SSIM and sharpness.
    def _score(r: Dict[str, object]) -> Tuple[float, float, float]:
        if lpips_available and r["lpips_after"] is not None:
            return (float(r["lpips_after"]), -float(r["ssim_after"]), -float(r["sharpness_after"]))
        return (-float(r["ssim_after"]), -float(r["sharpness_after"]), -float(r["psnr_after"]))

    rows_sorted = sorted(rows, key=_score)
    top3 = rows_sorted[:3]

    any_improved = any(bool(r["improved_psnr"]) or bool(r["improved_ssim"]) for r in rows)

    csv_path = out_dir / "sweep_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    result_payload = {
        "scene": scene_name,
        "artifact_scene_dir": str(scene_dir),
        "rate_dir": rate_dir,
        "checkpoint": str(ckpt_path),
        "config": str(cfg_path),
        "backbone_model": cfg["model"]["pretrained_model_name_or_path"],
        "views": view_names,
        "t0_values": t0_values,
        "adapter_scales": scales,
        "lpips_available": bool(lpips_available),
        "any_improvement_psnr_or_ssim": bool(any_improved),
        "top3": top3,
        "num_runs": len(rows),
    }

    (out_dir / "sweep_summary.json").write_text(json.dumps(result_payload, indent=2))

    known_good_path = ROOT / "logs" / "nifi_known_good.yaml"
    known_good_path.parent.mkdir(parents=True, exist_ok=True)
    with known_good_path.open("w") as f:
        yaml.safe_dump(
            {
                "selected_from": str(out_dir / "sweep_summary.json"),
                "note": "Best config found by local sweep (may still underperform baseline if missing paper components).",
                "best": top3[0] if top3 else None,
            },
            f,
            sort_keys=False,
        )

    sweep_gallery = ROOT / "docs" / "assets" / "debug_sweeps" / f"{scene_name}_t0_scale_grid.png"
    _save_grid_montage(rows, t0_values=t0_values, scales=scales, out_path=sweep_gallery)

    print(f"[done] wrote sweep csv: {csv_path}")
    print(f"[done] wrote sweep json: {out_dir / 'sweep_summary.json'}")
    print(f"[done] wrote best config hint: {known_good_path}")
    print(f"[done] wrote sweep montage: {sweep_gallery}")


if __name__ == "__main__":
    main()
