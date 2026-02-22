#!/usr/bin/env python3
"""Garden-only adapter training with explicit GPU/runtime diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel
from nifi.data.builders import build_paired_dataloader
from nifi.gs.compressor import CompressionConfig, Proxy3DGSCompressor, list_scene_images
from nifi.losses.perceptual import ReconstructionLossBundle
from nifi.metrics.perceptual_metrics import PerceptualMetrics
from nifi.metrics.simple_metrics import ssim
from nifi.restoration_distribution_matching import (
    ground_truth_direction_surrogate_eq5,
    kl_divergence_surrogate_eq4,
    phi_minus_objective_eq6,
    phi_plus_diffusion_objective_eq8,
)
from nifi.utils.checkpoint import load_checkpoint, save_checkpoint
from nifi.utils.config import load_config
from nifi.utils.runtime import configure_runtime, get_runtime_defaults, resolve_device
from nifi.utils.seed import set_seed
from nifi.utils.triptych import save_labeled_triptych


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Garden adapters on GPU")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--model_config", type=str, default="configs/default_sd15.yaml")
    p.add_argument("--artifact_scene", type=str, default="artifacts/garden")
    p.add_argument("--pairs_root", type=str, default="pairs_real")
    p.add_argument("--scene", type=str, default="garden")
    p.add_argument("--rates", type=str, default="0.1")
    p.add_argument("--rate", type=float, default=None, help="Single compression rate alias for --rates")
    p.add_argument("--prompt", type=str, default="a detailed outdoor garden scene with trees, plants, stone paths, and natural daylight")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--allow_short_run", action="store_true", help="Allow max_steps < 10000 for quick debug runs")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=None, help="Deprecated alias for --grad_accum")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lr_phi_minus", type=float, default=None)
    p.add_argument("--lr_phi_plus", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    p.add_argument("--t0", type=int, default=149)
    p.add_argument("--adapter_rank", type=int, default=None, help="Override LoRA/adaptor rank from model config")
    p.add_argument("--train_phi_plus", dest="train_phi_plus", action="store_true")
    p.add_argument("--no_train_phi_plus", dest="train_phi_plus", action="store_false")
    p.set_defaults(train_phi_plus=True)
    p.add_argument("--adapter_scale", type=float, default=None, help="Alias for --adapter_scale_eval")
    p.add_argument("--adapter_scale_eval", type=float, default=0.5)
    p.add_argument("--detail_boost_eval", type=float, default=0.5)
    p.add_argument("--detail_sigma_eval", type=float, default=1.2)
    p.add_argument("--eval_views", type=str, default="0,10,20")
    p.add_argument("--val_batches", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=64)
    p.add_argument("--min_lpips_gain_gate", type=float, default=0.02)
    p.add_argument("--min_ssim_gain_gate", type=float, default=0.0)
    p.add_argument("--min_abs_diff_gate", type=float, default=0.01)

    p.add_argument("--out_dir", type=str, default="runs/garden_adapters")
    p.add_argument("--val_out_root", type=str, default="outputs/garden/val")
    p.add_argument("--metrics_csv", type=str, default="outputs/garden/metrics.csv")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/garden")
    p.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    p.add_argument("--update_model_paths", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    if args.rate is not None:
        args.rates = str(float(args.rate))
    if args.grad_accum_steps is not None:
        args.grad_accum = int(args.grad_accum_steps)
    if args.adapter_scale is not None:
        args.adapter_scale_eval = float(args.adapter_scale)
    return args


def _parse_rates(raw: str) -> Tuple[List[float], List[str]]:
    floats: List[float] = []
    tags: List[str] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("rate_"):
            val = float(token.split("rate_", 1)[1])
        else:
            val = float(token)
        floats.append(val)
        tags.append(f"rate_{val:.3f}")
    if not floats:
        raise ValueError("At least one rate is required")
    return floats, tags


def _parse_view_indices(raw: str) -> List[int]:
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError("At least one eval view index is required")
    return vals


def _pick_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def to_minus1_1(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device) * 2.0 - 1.0


def save_minus1_1(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img01 = ((tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_abs_diff_map(path: Path, x: torch.Tensor, y: torch.Tensor, amplify: float = 10.0) -> None:
    d = torch.abs(x - y).mean(dim=1).squeeze(0).detach().cpu().numpy()
    vis = np.clip(d * float(amplify), 0.0, 1.0)
    arr = np.clip(np.round(vis * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def laplacian_variance(x_minus1_1: torch.Tensor) -> float:
    gray = ((x_minus1_1.clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1, keepdim=True).float()
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    y = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return float(y.var().item())


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


def ensure_garden_artifact_scene(scene_name: str, rates: Sequence[float]) -> Path:
    artifact_scene = ROOT / "artifacts" / scene_name
    if artifact_scene.exists() and (artifact_scene / "clean").exists() and all(
        (artifact_scene / f"rate_{float(r):.3f}").exists() for r in rates
    ):
        return artifact_scene

    dataset_scene = ROOT / "data" / "mipnerf360" / scene_name
    if not dataset_scene.exists():
        raise FileNotFoundError(
            f"Missing Garden dataset source: {dataset_scene}. "
            "Expected Mip-NeRF360 images under data/mipnerf360/garden."
        )

    artifact_scene.mkdir(parents=True, exist_ok=True)
    clean_dir = artifact_scene / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    compressor = Proxy3DGSCompressor(
        CompressionConfig(
            jpeg_quality_min=8,
            jpeg_quality_max=55,
            downsample_min=2,
            downsample_max=8,
        )
    )

    names: List[str] = []
    for i, src in enumerate(list_scene_images(dataset_scene)):
        name = f"{i:05d}.png"
        names.append(name)
        with Image.open(src) as img:
            img.convert("RGB").save(clean_dir / name)

    for rate in rates:
        rate_dir = artifact_scene / f"rate_{float(rate):.3f}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            with Image.open(clean_dir / name) as clean:
                degraded = compressor.degrade_image(clean.convert("RGB"), float(rate))
                degraded.save(rate_dir / name)
    return artifact_scene


def ensure_rate_dir(artifact_scene: Path, rate: float) -> str:
    rate_dir = f"rate_{float(rate):.3f}"
    if (artifact_scene / rate_dir).exists():
        return rate_dir
    if not (artifact_scene / "clean").exists():
        raise FileNotFoundError(f"clean folder missing in artifact scene: {artifact_scene}")

    compressor = Proxy3DGSCompressor(
        CompressionConfig(
            jpeg_quality_min=8,
            jpeg_quality_max=55,
            downsample_min=2,
            downsample_max=8,
        )
    )
    dst = artifact_scene / rate_dir
    dst.mkdir(parents=True, exist_ok=True)
    for src in sorted((artifact_scene / "clean").glob("*.png")):
        with Image.open(src) as img:
            degraded = compressor.degrade_image(img.convert("RGB"), float(rate))
            degraded.save(dst / src.name)
    return rate_dir


def _infinite_loader(loader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _to_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _grad_norm(params: Iterable[torch.nn.Parameter]) -> Tuple[float, bool]:
    total = 0.0
    nonzero = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n = float(torch.linalg.vector_norm(g, ord=2).item())
        total += n * n
        if n > 0.0:
            nonzero = True
    return float(math.sqrt(total)), nonzero


def _load_split_names(artifact_scene: Path) -> Dict[str, List[str]]:
    meta_path = artifact_scene / "metadata.json"
    if meta_path.exists():
        payload = json.loads(meta_path.read_text())
        splits = payload.get("splits", {}) if isinstance(payload, dict) else {}
        if isinstance(splits, dict) and isinstance(splits.get("train"), list) and isinstance(splits.get("test"), list):
            return {"train": [str(x) for x in splits["train"]], "test": [str(x) for x in splits["test"]]}
    all_names = sorted([p.name for p in (artifact_scene / "clean").glob("*.png")])
    return {
        "train": [name for i, name in enumerate(all_names) if i % 8 != 0],
        "test": [name for i, name in enumerate(all_names) if i % 8 == 0],
    }


def ensure_garden_pairs(
    *,
    artifact_scene: Path,
    pairs_root: Path,
    scene_name: str,
    rates: Sequence[float],
    prompt: str,
) -> Path:
    splits = _load_split_names(artifact_scene)
    scene_out = pairs_root / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)
    (scene_out / "prompt.txt").write_text(prompt.strip() + "\n")

    stats: Dict[str, int] = {}
    for rate in rates:
        rate_dir = ensure_rate_dir(artifact_scene, rate=rate)
        for split in ("train", "test"):
            names = splits[split]
            clean_out = scene_out / rate_dir / split / "clean"
            degraded_out = scene_out / rate_dir / split / "degraded"
            clean_out.mkdir(parents=True, exist_ok=True)
            degraded_out.mkdir(parents=True, exist_ok=True)
            copied = 0
            for name in names:
                src_clean = artifact_scene / "clean" / name
                src_deg = artifact_scene / rate_dir / name
                dst_clean = clean_out / name
                dst_deg = degraded_out / name
                if not src_clean.exists() or not src_deg.exists():
                    continue
                if not dst_clean.exists():
                    shutil.copy2(src_clean, dst_clean)
                if not dst_deg.exists():
                    shutil.copy2(src_deg, dst_deg)
                copied += 1
            stats[f"{rate_dir}/{split}"] = copied

    (scene_out / "pairs_manifest.json").write_text(json.dumps({"scene": scene_name, "stats": stats}, indent=2))
    return scene_out


def _pad_to_multiple(x: torch.Tensor, m: int = 8) -> Tuple[torch.Tensor, Dict[str, int]]:
    _, _, h, w = x.shape
    ph = (m - (h % m)) % m
    pw = (m - (w % m)) % m
    if ph == 0 and pw == 0:
        return x, {"left": 0, "right": 0, "top": 0, "bottom": 0}
    left = pw // 2
    right = pw - left
    top = ph // 2
    bottom = ph - top
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


def restore_fullres_eq7(
    *,
    model: FrozenBackboneArtifactRestorationModel,
    lq_minus1_1: torch.Tensor,
    prompt: str,
    t0: int,
    adapter_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    lq_pad, pads = _pad_to_multiple(lq_minus1_1, m=8)
    b = lq_pad.shape[0]
    t = torch.full((b,), int(t0), device=lq_pad.device, dtype=torch.long)
    with torch.no_grad(), autocast(device_type=lq_pad.device.type, dtype=amp_dtype, enabled=use_amp):
        z_lq = model.encode_images(lq_pad)
        z_t0, _ = model.q_sample(z_lq, t, noise=torch.zeros_like(z_lq))
        eps_base = model.predict_eps(z_t0, t, [prompt] * b, adapter_type=None, train_mode=False)
        eps_minus = model.predict_eps(z_t0, t, [prompt] * b, adapter_type="minus", train_mode=False)
        eps = eps_base + float(adapter_scale) * (eps_minus - eps_base)
        sigma = model.sigma(t).view(-1, 1, 1, 1)
        z_hat = z_t0 - sigma * eps
        restored = model.decode_latents(z_hat)
    return _unpad(restored, pads)


def evaluate_dataset(
    *,
    model: FrozenBackboneArtifactRestorationModel,
    loader,
    perceptual: PerceptualMetrics,
    device: torch.device,
    t0: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> Dict[str, float]:
    lpips_before: List[float] = []
    lpips_after: List[float] = []
    ssim_before: List[float] = []
    ssim_after: List[float] = []
    sharp_before: List[float] = []
    sharp_after: List[float] = []

    model.eval()
    for i, batch in enumerate(loader):
        if i >= int(max_batches):
            break
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        prompts = list(batch["prompt"])
        with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z_deg = model.encode_images(degraded)
            restored_z = restore_fullres_eq7(
                model=model,
                lq_minus1_1=degraded,
                prompt=prompts[0] if prompts else "",
                t0=int(t0),
                adapter_scale=1.0,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        lpips_before.extend(perceptual.lpips(degraded, clean).detach().cpu().tolist())
        lpips_after.extend(perceptual.lpips(restored_z, clean).detach().cpu().tolist())

        clean01 = ((clean + 1.0) * 0.5).float()
        deg01 = ((degraded + 1.0) * 0.5).float()
        rst01 = ((restored_z + 1.0) * 0.5).float()
        ssim_before.append(float(ssim(deg01, clean01)))
        ssim_after.append(float(ssim(rst01, clean01)))
        sharp_before.append(float(laplacian_variance(degraded)))
        sharp_after.append(float(laplacian_variance(restored_z)))
    model.train()

    return {
        "lpips_before": float(np.mean(lpips_before)) if lpips_before else float("nan"),
        "lpips_after": float(np.mean(lpips_after)) if lpips_after else float("nan"),
        "ssim_before": float(np.mean(ssim_before)) if ssim_before else float("nan"),
        "ssim_after": float(np.mean(ssim_after)) if ssim_after else float("nan"),
        "sharpness_before": float(np.mean(sharp_before)) if sharp_before else float("nan"),
        "sharpness_after": float(np.mean(sharp_after)) if sharp_after else float("nan"),
    }


def run_fixed_view_validation(
    *,
    model: FrozenBackboneArtifactRestorationModel,
    artifact_scene: Path,
    rate_dir: str,
    view_indices: Sequence[int],
    step: int,
    val_out_root: Path,
    perceptual: PerceptualMetrics,
    prompt: str,
    t0: int,
    adapter_scale: float,
    detail_boost: float,
    detail_sigma: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    val_dir = val_out_root / f"step_{step:06d}"
    model.eval()
    for idx in view_indices:
        name = f"{int(idx):05d}.png"
        hq = to_minus1_1(artifact_scene / "clean" / name, device=device)
        lq = to_minus1_1(artifact_scene / rate_dir / name, device=device)
        restored = restore_fullres_eq7(
            model=model,
            lq_minus1_1=lq,
            prompt=prompt,
            t0=t0,
            adapter_scale=adapter_scale,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        restored = _inject_lq_detail(restored, lq, amount=float(detail_boost), sigma=float(detail_sigma))

        lpips_before = float(perceptual.lpips(lq, hq).mean().item())
        lpips_after = float(perceptual.lpips(restored, hq).mean().item())
        hq01 = ((hq + 1.0) * 0.5).float()
        lq01 = ((lq + 1.0) * 0.5).float()
        rst01 = ((restored + 1.0) * 0.5).float()
        ssim_before = float(ssim(lq01, hq01))
        ssim_after = float(ssim(rst01, hq01))
        sharp_lq = float(laplacian_variance(lq))
        sharp_restored = float(laplacian_variance(restored))
        mean_abs_diff = float(torch.abs(restored - lq).mean().item())

        base = val_dir / f"view_{idx:03d}"
        hq_path = base / "hq.png"
        lq_path = base / "lq.png"
        rst_path = base / "restored.png"
        triptych_path = base / "triptych.png"
        diff_path = base / "diff_restored_minus_lq_x10.png"
        save_minus1_1(hq_path, hq)
        save_minus1_1(lq_path, lq)
        save_minus1_1(rst_path, restored)
        save_abs_diff_map(diff_path, restored, lq, amplify=10.0)
        save_labeled_triptych(
            hq_path=hq_path,
            lq_path=lq_path,
            restored_path=rst_path,
            out_path=triptych_path,
            title=f"Garden step {step} view {name} | HQ | Compressed | Restored",
        )

        rows.append(
            {
                "view_index": int(idx),
                "view_name": name,
                "lpips_before": lpips_before,
                "lpips_after": lpips_after,
                "lpips_gain": lpips_before - lpips_after,
                "ssim_before": ssim_before,
                "ssim_after": ssim_after,
                "ssim_gain": ssim_after - ssim_before,
                "sharpness_lq": sharp_lq,
                "sharpness_restored": sharp_restored,
                "sharpness_gain": sharp_restored - sharp_lq,
                "mean_abs_diff_restored_vs_lq": mean_abs_diff,
                "triptych": str(triptych_path.resolve()),
            }
        )
    model.train()

    summary = {
        "step": int(step),
        "val_dir": str(val_dir.resolve()),
        "mean_lpips_gain": float(np.mean([r["lpips_gain"] for r in rows])) if rows else float("nan"),
        "mean_ssim_gain": float(np.mean([r["ssim_gain"] for r in rows])) if rows else float("nan"),
        "mean_sharpness_gain": float(np.mean([r["sharpness_gain"] for r in rows])) if rows else float("nan"),
        "mean_abs_diff_restored_vs_lq": float(np.mean([r["mean_abs_diff_restored_vs_lq"] for r in rows])) if rows else float("nan"),
        "rows": rows,
    }
    (val_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def _save_phi_minus_artifacts(phi_minus_state: Dict[str, torch.Tensor], dst_prefix: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    pt_path = dst_prefix.with_suffix(".pt")
    torch.save(phi_minus_state, pt_path)
    out["phi_minus_pt"] = str(pt_path)
    try:
        from safetensors.torch import save_file  # type: ignore

        sf_path = dst_prefix.with_suffix(".safetensors")
        payload = {k: v.detach().cpu() for k, v in phi_minus_state.items()}
        save_file(payload, str(sf_path))
        out["phi_minus_safetensors"] = str(sf_path)
    except Exception:
        pass
    return out


def _reload_checkpoint_sanity(path: Path, min_size_mb: float = 0.5) -> Dict[str, object]:
    size_mb = float(path.stat().st_size / (1024.0 * 1024.0)) if path.exists() else 0.0
    payload = load_checkpoint(path, map_location="cpu")
    ok = (
        bool(path.exists())
        and size_mb >= float(min_size_mb)
        and isinstance(payload, dict)
        and isinstance(payload.get("phi_minus"), dict)
        and isinstance(payload.get("phi_plus"), dict)
        and len(payload.get("phi_minus", {})) > 0
    )
    return {"checkpoint": str(path), "size_mb": size_mb, "ok": bool(ok)}


def _trainable_param_summary(model: torch.nn.Module) -> Dict[str, object]:
    rows = []
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            rows.append({"name": name, "numel": int(p.numel())})
            total += int(p.numel())
    return {"total_trainable_params": int(total), "trainable_tensors": rows}


def _maybe_update_model_paths(model_paths: Path, model_config: Path, checkpoint: Path) -> None:
    payload = json.loads(json.dumps({}))
    try:
        import yaml

        payload = yaml.safe_load(model_paths.read_text()) if model_paths.exists() else {}
        if not isinstance(payload, dict):
            payload = {}
        restoration = payload.get("restoration", {})
        if not isinstance(restoration, dict):
            restoration = {}
        restoration["model_config"] = str(model_config.relative_to(ROOT) if model_config.is_relative_to(ROOT) else model_config)
        restoration["adapter_checkpoint"] = str(checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint)
        payload["restoration"] = restoration
        model_paths.parent.mkdir(parents=True, exist_ok=True)
        model_paths.write_text(yaml.safe_dump(payload, sort_keys=False))
    except Exception as exc:
        print(f"[warn] failed to update model paths: {exc}")


def main() -> None:
    args = parse_args()

    if args.device != "cuda":
        raise RuntimeError("train_garden_adapters.py must run with --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable but --device cuda was requested")
    if int(args.max_steps) < 10000 and not bool(args.allow_short_run):
        raise RuntimeError(
            f"--max_steps {int(args.max_steps)} is below required minimum 10000. "
            "Use --allow_short_run only for explicit debug runs."
        )

    start_wall = time.time()
    seed = int(args.seed)
    set_seed(seed, deterministic=False)

    rates_float, rates_tags = _parse_rates(args.rates)
    eval_views = _parse_view_indices(args.eval_views)

    artifact_scene = _as_abs(args.artifact_scene)
    if not artifact_scene.exists() or not (artifact_scene / "clean").exists():
        artifact_scene = ensure_garden_artifact_scene(scene_name=args.scene, rates=rates_float)
    for r in rates_float:
        _ = ensure_rate_dir(artifact_scene, r)

    pairs_root = _as_abs(args.pairs_root)
    scene_pairs = ensure_garden_pairs(
        artifact_scene=artifact_scene,
        pairs_root=pairs_root,
        scene_name=args.scene,
        rates=rates_float,
        prompt=args.prompt,
    )

    out_dir = _as_abs(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_out_root = _as_abs(args.val_out_root)
    val_out_root.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = _as_abs(args.metrics_csv)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _as_abs(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_config_path = _as_abs(args.model_config)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config missing: {model_config_path}")
    cfg = load_config(str(model_config_path))

    runtime_cfg = get_runtime_defaults()
    runtime_cfg.update(cfg.get("runtime", {}))
    runtime_cfg["device"] = "cuda"
    runtime_cfg["device_id"] = 0
    configure_runtime(runtime_cfg)
    device = resolve_device(runtime_cfg)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    props = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "torch_version": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "device": str(device),
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "torch_cuda_version": str(torch.version.cuda),
                "gpu_mem_alloc_mb_start": float(torch.cuda.memory_allocated(device) / (1024.0 ** 2)),
                "gpu_mem_reserved_mb_start": float(torch.cuda.memory_reserved(device) / (1024.0 ** 2)),
            },
            indent=2,
        )
    )

    amp_dtype = _pick_dtype(args.mixed_precision)
    use_amp = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    model_cfg = ArtifactRestorationDiffusionConfig(
        model_name_or_path=str(cfg["model"]["pretrained_model_name_or_path"]),
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        lora_rank=int(args.adapter_rank if args.adapter_rank is not None else cfg["model"]["lora_rank"]),
        guidance_scale=float(cfg["model"]["guidance_scale"]),
        prompt_dropout=float(cfg["model"]["prompt_dropout"]),
        max_token_length=int(cfg["model"]["max_token_length"]),
        vae_scaling_factor=float(cfg["model"]["vae_scaling_factor"]),
    )
    model = FrozenBackboneArtifactRestorationModel(model_cfg, device=device, dtype=amp_dtype if use_amp else torch.float32)
    model.freeze_backbone()
    model.phi_minus.requires_grad_(True)
    model.phi_plus.requires_grad_(bool(args.train_phi_plus))
    model.train()

    trainable = _trainable_param_summary(model)
    print(json.dumps({"trainable": trainable}, indent=2))

    minus_params = [p for p in model.phi_minus.parameters() if p.requires_grad]
    plus_params = [p for p in model.phi_plus.parameters() if p.requires_grad]
    if not minus_params:
        raise RuntimeError("No trainable phi_minus params found")
    if args.train_phi_plus and not plus_params:
        raise RuntimeError("--train_phi_plus requested but no trainable phi_plus params found")

    lr_minus = float(args.lr_phi_minus if args.lr_phi_minus is not None else args.lr)
    lr_plus = float(args.lr_phi_plus if args.lr_phi_plus is not None else (lr_minus * 0.2))
    opt_minus = torch.optim.AdamW(minus_params, lr=lr_minus, weight_decay=float(args.weight_decay))
    opt_plus = torch.optim.AdamW(plus_params, lr=lr_plus, weight_decay=float(args.weight_decay)) if plus_params else None

    rec_losses = ReconstructionLossBundle().to(device)
    rec_losses.eval()
    for p in rec_losses.parameters():
        p.requires_grad_(False)
    perceptual = PerceptualMetrics(device=device)

    train_loader = build_paired_dataloader(
        data_root=str(pairs_root),
        split="train",
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=True,
        max_samples=args.max_train_samples,
        allowed_rates=rates_tags,
        pin_memory=True,
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=2,
    )
    val_loader = build_paired_dataloader(
        data_root=str(pairs_root),
        split="test",
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=max(0, int(args.num_workers) // 2),
        shuffle=False,
        max_samples=args.max_eval_samples,
        allowed_rates=rates_tags,
        pin_memory=True,
        persistent_workers=bool(args.num_workers > 1),
        prefetch_factor=2,
    )

    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    train_iter = _infinite_loader(train_loader)

    step_start = 1
    best_lpips_after = float("inf")
    best_fixed_lpips_gain = float("-inf")
    any_gate_pass = False
    saved_checkpoints: List[str] = []
    val_snapshots: List[str] = []

    if args.resume:
        resume = _as_abs(args.resume)
        payload = torch.load(resume, map_location="cpu")
        model.phi_minus.load_state_dict(payload["phi_minus"])
        model.phi_plus.load_state_dict(payload["phi_plus"])
        if payload.get("opt_minus") is not None:
            opt_minus.load_state_dict(payload["opt_minus"])
        if opt_plus is not None and payload.get("opt_plus") is not None:
            opt_plus.load_state_dict(payload["opt_plus"])
        if scaler.is_enabled() and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        step_start = int(payload.get("step", 0)) + 1
        best_lpips_after = float(payload.get("best_metric", best_lpips_after))
        print(f"[info] resumed from {resume} at step {step_start}")

    grad_accum = max(1, int(args.grad_accum))
    log_every = max(1, int(args.log_every))
    save_every = max(1, int(args.save_every))
    val_every = max(1, int(args.val_every))
    max_steps = int(args.max_steps)
    t0 = int(args.t0)
    prompt = str(args.prompt)
    rate_dir_eval = f"rate_{float(rates_float[0]):.3f}"

    csv_path = out_dir / "train_log.csv"
    val_csv_path = out_dir / "val_log.csv"
    if args.resume is None:
        for path in (csv_path, val_csv_path, metrics_csv_path):
            if path.exists():
                path.unlink()
    opt_minus.zero_grad(set_to_none=True)
    if opt_plus is not None:
        opt_plus.zero_grad(set_to_none=True)

    loop_t_prev = time.perf_counter()
    history: Dict[str, object] = {
        "loss_start": None,
        "loss_end": None,
        "steps": max_steps,
        "device": str(device),
        "gpu": props.name,
    }

    for step in range(step_start, max_steps + 1):
        batch = next(train_iter)
        clean = batch["clean"].to(device, non_blocking=True)
        degraded = batch["degraded"].to(device, non_blocking=True)
        prompts = list(batch["prompt"])
        prompts_train = [p if p.strip() else prompt for p in prompts]
        bsz = clean.shape[0]
        t0_t = torch.full((bsz,), t0, device=device, dtype=torch.long)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z_clean = model.encode_images(clean)
            z_deg = model.encode_images(degraded)

            z_tilde_t0, _ = model.q_sample(z_deg, t0_t)
            eps_minus = model.predict_eps(z_tilde_t0, t0_t, prompts_train, adapter_type="minus", train_mode=False)
            sigma0 = model.sigma(t0_t).view(-1, 1, 1, 1)
            z_hat = z_tilde_t0 - sigma0 * eps_minus

            t_dist = torch.randint(0, model_cfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            z_hat_t, noise_dist = model.q_sample(z_hat, t_dist)
            sigma_dist = model.sigma(t_dist).view(-1, 1, 1, 1)
            z_clean_t = (1.0 - sigma_dist) * z_clean + sigma_dist * noise_dist

            eps_real_hat = model.predict_eps(z_hat_t, t_dist, prompts_train, adapter_type=None, train_mode=False)
            eps_restore = model.predict_eps(
                z_hat_t,
                t_dist,
                prompts_train,
                adapter_type="plus" if opt_plus is not None else None,
                train_mode=False,
            )
            eps_real_clean = model.predict_eps(z_clean_t, t_dist, prompts_train, adapter_type=None, train_mode=False)

            s_real_hat = model.score_from_eps(z_hat_t, eps_real_hat, sigma_dist)
            s_restore = model.score_from_eps(z_hat_t, eps_restore, sigma_dist)
            s_real_clean = model.score_from_eps(z_clean_t, eps_real_clean, sigma_dist)

            loss_kl = kl_divergence_surrogate_eq4(z_hat, s_real_hat, s_restore)
            loss_gt = ground_truth_direction_surrogate_eq5(z_hat, s_real_clean, s_real_hat)
            restored = model.decode_latents(z_hat)

        rec = rec_losses(clean.float(), restored.float())
        total_minus = phi_minus_objective_eq6(
            alpha=0.7,
            kl_term=loss_kl,
            gt_term=loss_gt,
            l2_term=rec["l2"],
            lpips_term=rec["lpips"],
            dists_term=rec["dists"],
            weight_kl=1.0,
            weight_gt=1.0,
            weight_l2=1.0,
            weight_lpips=1.0,
            weight_dists=0.0,
        )
        scaled_minus = total_minus / grad_accum
        if scaler.is_enabled():
            scaler.scale(scaled_minus).backward()
        else:
            scaled_minus.backward()

        if opt_plus is not None:
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                z_hat_det = z_hat.detach()
                t_plus = torch.randint(0, model_cfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
                z_plus_t, noise_plus = model.q_sample(z_hat_det, t_plus)
                eps_plus = model.predict_eps(z_plus_t, t_plus, prompts_train, adapter_type="plus", train_mode=False)
                loss_plus = phi_plus_diffusion_objective_eq8(eps_plus, noise_plus)
            scaled_plus = loss_plus / grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled_plus).backward()
            else:
                scaled_plus.backward()
        else:
            loss_plus = torch.tensor(0.0, device=device)

        step_ready = (step % grad_accum) == 0
        grad_norm_minus = float("nan")
        grad_norm_plus = float("nan")
        grad_minus_nonzero = False
        grad_plus_nonzero = False
        if step_ready:
            if scaler.is_enabled():
                scaler.unscale_(opt_minus)
                if opt_plus is not None:
                    scaler.unscale_(opt_plus)
            grad_norm_minus, grad_minus_nonzero = _grad_norm(minus_params)
            if opt_plus is not None:
                grad_norm_plus, grad_plus_nonzero = _grad_norm(plus_params)
            torch.nn.utils.clip_grad_norm_(minus_params, float(args.grad_clip))
            if opt_plus is not None:
                torch.nn.utils.clip_grad_norm_(plus_params, float(args.grad_clip))

            if scaler.is_enabled():
                scaler.step(opt_minus)
                if opt_plus is not None:
                    scaler.step(opt_plus)
                scaler.update()
            else:
                opt_minus.step()
                if opt_plus is not None:
                    opt_plus.step()

            opt_minus.zero_grad(set_to_none=True)
            if opt_plus is not None:
                opt_plus.zero_grad(set_to_none=True)

        if history["loss_start"] is None:
            history["loss_start"] = float(total_minus.item())
        history["loss_end"] = float(total_minus.item())

        now = time.perf_counter()
        step_time = now - loop_t_prev
        loop_t_prev = now
        steps_per_sec = float(1.0 / max(step_time, 1e-6))
        mem_alloc = int(torch.cuda.memory_allocated(device))
        mem_peak = int(torch.cuda.max_memory_allocated(device))

        row = {
            "step": int(step),
            "loss_minus": float(total_minus.item()),
            "loss_plus": float(loss_plus.item()),
            "loss_kl": float(loss_kl.item()),
            "loss_gt": float(loss_gt.item()),
            "loss_l2": float(rec["l2"].item()),
            "loss_lpips": float(rec["lpips"].item()),
            "steps_per_sec": steps_per_sec,
            "gpu_mem_alloc_mb": float(mem_alloc / (1024 ** 2)),
            "gpu_mem_peak_mb": float(mem_peak / (1024 ** 2)),
            "grad_norm_phi_minus": float(grad_norm_minus),
            "grad_norm_phi_plus": float(grad_norm_plus),
            "grad_phi_minus_nonzero": bool(grad_minus_nonzero),
            "grad_phi_plus_nonzero": bool(grad_plus_nonzero if opt_plus is not None else False),
        }
        _to_csv(csv_path, row)

        if step % log_every == 0 or step == step_start:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss_minus": row["loss_minus"],
                        "loss_plus": row["loss_plus"],
                        "steps_per_sec": row["steps_per_sec"],
                        "gpu_mem_alloc_mb": row["gpu_mem_alloc_mb"],
                        "grad_norm_phi_minus": row["grad_norm_phi_minus"],
                        "grad_norm_phi_plus": row["grad_norm_phi_plus"],
                    },
                    indent=2,
                )
            )

        if step % val_every == 0 or step == max_steps:
            eval_row = evaluate_dataset(
                model=model,
                loader=val_loader,
                perceptual=perceptual,
                device=device,
                t0=t0,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_batches=int(args.val_batches),
            )
            fixed = run_fixed_view_validation(
                model=model,
                artifact_scene=artifact_scene,
                rate_dir=rate_dir_eval,
                view_indices=eval_views,
                step=step,
                val_out_root=val_out_root,
                perceptual=perceptual,
                prompt=prompt,
                t0=t0,
                adapter_scale=float(args.adapter_scale_eval),
                detail_boost=float(args.detail_boost_eval),
                detail_sigma=float(args.detail_sigma_eval),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                device=device,
            )
            val_row = {
                "step": int(step),
                "lpips_before": float(eval_row["lpips_before"]),
                "lpips_after": float(eval_row["lpips_after"]),
                "lpips_gain": float(eval_row["lpips_before"] - eval_row["lpips_after"]),
                "ssim_before": float(eval_row["ssim_before"]),
                "ssim_after": float(eval_row["ssim_after"]),
                "ssim_gain": float(eval_row["ssim_after"] - eval_row["ssim_before"]),
                "sharpness_before": float(eval_row["sharpness_before"]),
                "sharpness_after": float(eval_row["sharpness_after"]),
                "sharpness_gain": float(eval_row["sharpness_after"] - eval_row["sharpness_before"]),
                "fixed_mean_lpips_gain": float(fixed["mean_lpips_gain"]),
                "fixed_mean_ssim_gain": float(fixed["mean_ssim_gain"]),
                "fixed_mean_sharpness_gain": float(fixed["mean_sharpness_gain"]),
                "fixed_mean_abs_diff": float(fixed["mean_abs_diff_restored_vs_lq"]),
                "fixed_val_dir": str(fixed["val_dir"]),
            }
            lpips_gate_ok = bool(val_row["fixed_mean_lpips_gain"] >= float(args.min_lpips_gain_gate))
            ssim_gate_ok = bool(val_row["fixed_mean_ssim_gain"] >= float(args.min_ssim_gain_gate))
            diff_gate_ok = bool(val_row["fixed_mean_abs_diff"] >= float(args.min_abs_diff_gate))
            sharp_gate_ok = bool(val_row["fixed_mean_sharpness_gain"] >= 0.0)
            val_row["gate_pass"] = bool((lpips_gate_ok or ssim_gate_ok) and diff_gate_ok and sharp_gate_ok)
            val_row["gate_lpips_ok"] = lpips_gate_ok
            val_row["gate_ssim_ok"] = ssim_gate_ok
            val_row["gate_diff_ok"] = diff_gate_ok
            val_row["gate_sharpness_ok"] = sharp_gate_ok
            _to_csv(val_csv_path, val_row)
            _to_csv(metrics_csv_path, val_row)
            val_snapshots.append(str(fixed["val_dir"]))
            print(json.dumps({"validation": val_row}, indent=2))
            if bool(val_row["gate_pass"]):
                any_gate_pass = True

            if val_row["lpips_after"] < best_lpips_after:
                best_lpips_after = float(val_row["lpips_after"])
            if val_row["fixed_mean_lpips_gain"] > best_fixed_lpips_gain:
                best_fixed_lpips_gain = float(val_row["fixed_mean_lpips_gain"])
                best_path = ckpt_dir / "adapter_best.pt"
                save_checkpoint(
                    best_path,
                    step=step,
                    phi_minus_state=model.phi_minus.state_dict(),
                    phi_plus_state=model.phi_plus.state_dict(),
                    opt_minus_state=opt_minus.state_dict(),
                    opt_plus_state=opt_plus.state_dict() if opt_plus is not None else None,
                    scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                    best_metric=best_fixed_lpips_gain,
                    extra={
                        "scene": args.scene,
                        "rates": rates_float,
                        "t0": t0,
                        "prompt": prompt,
                        "criterion": "fixed_mean_lpips_gain",
                    },
                )
                saved_checkpoints.append(str(best_path))
                sanity = _reload_checkpoint_sanity(best_path, min_size_mb=0.5)
                if not bool(sanity["ok"]):
                    raise RuntimeError(f"Best checkpoint reload sanity failed: {sanity}")

            if step == 10000 and not any_gate_pass:
                best_path = ckpt_dir / "adapter_best.pt"
                print(
                    json.dumps(
                        {
                            "warning": "quality gate has not passed by step 10000",
                            "recommended_next": "conda run -n nifi python scripts/garden_debug_adjust.py --device cuda",
                            "best_checkpoint_so_far": str(best_path),
                        },
                        indent=2,
                    )
                )

        if step % save_every == 0 or step == max_steps:
            step_ckpt = ckpt_dir / f"adapter_step{step:06d}.pt"
            save_checkpoint(
                step_ckpt,
                step=step,
                phi_minus_state=model.phi_minus.state_dict(),
                phi_plus_state=model.phi_plus.state_dict(),
                opt_minus_state=opt_minus.state_dict(),
                opt_plus_state=opt_plus.state_dict() if opt_plus is not None else None,
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                best_metric=best_fixed_lpips_gain,
                extra={
                    "scene": args.scene,
                    "rates": rates_float,
                    "t0": t0,
                    "prompt": prompt,
                    "criterion": "fixed_mean_lpips_gain",
                },
            )
            saved_checkpoints.append(str(step_ckpt))
            sanity_step = _reload_checkpoint_sanity(step_ckpt, min_size_mb=0.5)
            if not bool(sanity_step["ok"]):
                raise RuntimeError(f"Step checkpoint reload sanity failed: {sanity_step}")
            phi_exports = _save_phi_minus_artifacts(
                model.phi_minus.state_dict(),
                ckpt_dir / f"phi_minus_step{step:06d}",
            )
            print(json.dumps({"checkpoint": str(step_ckpt), "phi_minus_exports": phi_exports}, indent=2))

    latest_path = ckpt_dir / "adapter_latest.pt"
    save_checkpoint(
        latest_path,
        step=max_steps,
        phi_minus_state=model.phi_minus.state_dict(),
        phi_plus_state=model.phi_plus.state_dict(),
        opt_minus_state=opt_minus.state_dict(),
        opt_plus_state=opt_plus.state_dict() if opt_plus is not None else None,
        scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
        best_metric=best_fixed_lpips_gain,
        extra={
            "scene": args.scene,
            "rates": rates_float,
            "t0": t0,
            "prompt": prompt,
            "criterion": "fixed_mean_lpips_gain",
        },
    )
    saved_checkpoints.append(str(latest_path))
    sanity_latest = _reload_checkpoint_sanity(latest_path, min_size_mb=0.5)
    if not bool(sanity_latest["ok"]):
        raise RuntimeError(f"Latest checkpoint reload sanity failed: {sanity_latest}")

    if args.update_model_paths:
        best_path = ckpt_dir / "adapter_best.pt"
        target_ckpt = best_path if best_path.exists() else latest_path
        _maybe_update_model_paths(_as_abs(args.model_paths), model_config_path, target_ckpt)

    elapsed = time.time() - start_wall
    status = "ok"
    if max_steps >= 10000 and not any_gate_pass:
        status = "needs_debug_adjust"
    summary = {
        "status": status,
        "scene": args.scene,
        "rates": rates_float,
        "pairs_root": str(pairs_root),
        "scene_pairs": str(scene_pairs),
        "artifact_scene": str(artifact_scene),
        "model_config": str(model_config_path),
        "max_steps": max_steps,
        "elapsed_sec": float(elapsed),
        "device": str(device),
        "gpu_name": props.name,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_mem_alloc_mb_end": float(torch.cuda.memory_allocated(device) / (1024.0 ** 2)),
        "gpu_mem_peak_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)),
        "best_lpips_after": float(best_lpips_after),
        "best_fixed_lpips_gain": float(best_fixed_lpips_gain),
        "quality_gate_passed_during_training": bool(any_gate_pass),
        "quality_gate_thresholds": {
            "min_lpips_gain_gate": float(args.min_lpips_gain_gate),
            "min_ssim_gain_gate": float(args.min_ssim_gain_gate),
            "min_abs_diff_gate": float(args.min_abs_diff_gate),
            "sharpness_non_decrease": True,
        },
        "validation_restore_params": {
            "t0": int(args.t0),
            "adapter_scale_eval": float(args.adapter_scale_eval),
            "detail_boost_eval": float(args.detail_boost_eval),
            "detail_sigma_eval": float(args.detail_sigma_eval),
        },
        "loss_start": history["loss_start"],
        "loss_end": history["loss_end"],
        "checkpoints": saved_checkpoints,
        "val_snapshots": val_snapshots,
        "val_out_root": str(val_out_root),
        "train_csv": str(csv_path),
        "val_csv": str(val_csv_path),
        "metrics_csv": str(metrics_csv_path),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    md_path = ROOT / "logs" / "garden_train_run.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Garden Training Run",
        "",
        f"- Start: {datetime.fromtimestamp(start_wall).isoformat(timespec='seconds')}",
        f"- End: {datetime.now().isoformat(timespec='seconds')}",
        f"- Total steps: {max_steps}",
        f"- Wall-clock seconds: {elapsed:.2f}",
        f"- Device: {device}",
        f"- GPU: {props.name}",
        f"- Model config: `{model_config_path}`",
        f"- Artifact scene: `{artifact_scene}`",
        f"- Pairs root: `{pairs_root}`",
        f"- Best LPIPS(after): {best_lpips_after:.6f}",
        f"- Best fixed-view LPIPS gain: {best_fixed_lpips_gain:.6f}",
        f"- Quality gate passed during training: {any_gate_pass}",
        f"- Validation restore params: t0={int(args.t0)}, scale={float(args.adapter_scale_eval)}, "
        f"detail_boost={float(args.detail_boost_eval)}, detail_sigma={float(args.detail_sigma_eval)}",
        f"- Loss start -> end: {history['loss_start']} -> {history['loss_end']}",
        "",
        "## Checkpoints",
    ]
    for ck in saved_checkpoints[-10:]:
        md_lines.append(f"- `{ck}`")
    md_lines.append("")
    md_lines.append("## Validation Snapshots")
    if val_snapshots:
        for snap in val_snapshots[-10:]:
            md_lines.append(f"- `{snap}`")
    else:
        md_lines.append("- none")
    md_lines.append("")
    md_lines.append("## Logs")
    md_lines.append(f"- `{csv_path}`")
    md_lines.append(f"- `{val_csv_path}`")
    md_lines.append(f"- `{metrics_csv_path}`")
    md_lines.append(f"- `{out_dir / 'train_summary.json'}`")
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[done] wrote {md_path}")

    if status != "ok":
        raise SystemExit(
            "Garden quality gate did not pass by 10k+ steps. "
            "Run: conda run -n nifi python scripts/garden_debug_adjust.py --device cuda"
        )


if __name__ == "__main__":
    main()
