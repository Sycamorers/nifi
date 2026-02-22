#!/usr/bin/env python3
"""Diagnose restoration wiring, checkpoints, and latent/timestep behavior."""

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
    parser = argparse.ArgumentParser(description="Diagnose restoration pipeline end-to-end")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g. garden) or artifact scene path")
    parser.add_argument("--rate", type=str, default="0.100")
    parser.add_argument("--views", nargs="*", type=int, default=None)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    parser.add_argument("--t0", type=int, default=None, help="Override t0")
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--disable_restore", action="store_true")

    parser.add_argument("--out_dir", type=str, default=None, help="Default: outputs/diag_restore/<scene>")
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
        raise RuntimeError("No shared clean/lq names")
    return names



def _choose_names(scene_dir: Path, names: Sequence[str], views: Optional[List[int]], num_views: int, seed: int) -> List[str]:
    names = list(names)
    if views:
        chosen = []
        for v in views:
            name = f"{int(v):05d}.png"
            if name not in names:
                raise ValueError(f"view {v} ({name}) missing")
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
    out = []
    span = len(rotated) - 1
    for i in range(num_views):
        out.append(rotated[round(i * span / (num_views - 1))])
    return out



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



def _pick_device(device_flag: str) -> str:
    if device_flag == "cpu":
        return "cpu"
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"



def _to_minus1_1_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return t * 2.0 - 1.0



def _save_minus1_1(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img01 = ((tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)



def _pad_to_multiple(x: torch.Tensor, m: int = 8) -> Tuple[torch.Tensor, Dict[str, int]]:
    _, _, h, w = x.shape
    pad_h = (m - (h % m)) % m
    pad_w = (m - (w % m)) % m
    if pad_h == 0 and pad_w == 0:
        return x, {"left": 0, "right": 0, "top": 0, "bottom": 0}
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = torch.nn.functional.pad(x, (left, right, top, bottom), mode="reflect")
    return padded, {"left": left, "right": right, "top": top, "bottom": bottom}



def _unpad(x: torch.Tensor, pads: Dict[str, int]) -> torch.Tensor:
    l, r, t, b = pads["left"], pads["right"], pads["top"], pads["bottom"]
    if l == r == t == b == 0:
        return x
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]



def _tensor_stats(x: torch.Tensor) -> Dict[str, object]:
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }



def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def _save_triplet_mosaic(hq: Path, lq: Path, restored: Path, title: str, out_path: Path) -> None:
    with Image.open(hq) as img_hq, Image.open(lq) as img_lq, Image.open(restored) as img_r:
        hq_i = img_hq.convert("RGB")
        lq_i = img_lq.convert("RGB")
        r_i = img_r.convert("RGB")

    base = hq_i.size
    if lq_i.size != base:
        lq_i = lq_i.resize(base, resample=Image.Resampling.BICUBIC)
    if r_i.size != base:
        r_i = r_i.resize(base, resample=Image.Resampling.BICUBIC)

    w, h = base
    pad = 12
    title_h = 24
    label_h = 20
    canvas = Image.new("RGB", (pad + w + pad + w + pad + w + pad, pad + title_h + pad + label_h + pad + h + pad), (236, 236, 236))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, pad), title, fill=(20, 20, 20), font=font)
    y_lab = pad + title_h + pad
    y_img = y_lab + label_h + pad

    draw.text((pad + w // 2 - 12, y_lab), "HQ", fill=(30, 30, 30), font=font)
    draw.text((pad + w + pad + w // 2 - 34, y_lab), "Compressed", fill=(30, 30, 30), font=font)
    draw.text((pad + 2 * (w + pad) + w // 2 - 24, y_lab), "Restored", fill=(30, 30, 30), font=font)

    canvas.paste(hq_i, (pad, y_img))
    canvas.paste(lq_i, (pad + w + pad, y_img))
    canvas.paste(r_i, (pad + 2 * (w + pad), y_img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)



def _param_abs_sum(state: Dict[str, torch.Tensor]) -> float:
    return float(sum(v.abs().sum().item() for v in state.values()))



def _count_trainable(module: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable



def run_restore_once(
    model: FrozenBackboneArtifactRestorationModel,
    lq: torch.Tensor,
    prompts: List[str],
    t0: int,
    adapter_scale: float,
    disable_restore: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    if disable_restore:
        return {
            "z_lq": torch.empty(0, device=lq.device),
            "z_t0": torch.empty(0, device=lq.device),
            "eps_base": torch.empty(0, device=lq.device),
            "eps_minus": torch.empty(0, device=lq.device),
            "eps_used": torch.empty(0, device=lq.device),
            "z_hat": torch.empty(0, device=lq.device),
            "restored": lq,
        }

    b = lq.shape[0]
    t = torch.full((b,), int(t0), device=lq.device, dtype=torch.long)

    with torch.no_grad(), autocast(device_type=lq.device.type, dtype=amp_dtype, enabled=use_amp):
        z_lq = model.encode_images(lq)
        z_t0, _ = model.q_sample(z_lq, t, noise=torch.zeros_like(z_lq))

        eps_base = model.predict_eps(z_t0, t, prompts, adapter_type=None, train_mode=False)
        eps_minus = model.predict_eps(z_t0, t, prompts, adapter_type="minus", train_mode=False)

        eps_used = eps_base + float(adapter_scale) * (eps_minus - eps_base)
        sigma = model.sigma(t).view(-1, 1, 1, 1)
        z_hat = z_t0 - sigma * eps_used
        restored = model.decode_latents(z_hat)

    return {
        "z_lq": z_lq,
        "z_t0": z_t0,
        "eps_base": eps_base,
        "eps_minus": eps_minus,
        "eps_used": eps_used,
        "z_hat": z_hat,
        "restored": restored,
    }



def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    scene_dir = _resolve_scene_artifacts(args.scene)
    scene_name = scene_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "outputs" / "diag_restore" / scene_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rate_dir = _resolve_rate_dir(scene_dir, args.rate)
    all_names = _load_candidate_names(scene_dir, rate_dir)
    selected = _choose_names(scene_dir, all_names, args.views, int(args.num_views), int(args.seed))

    ckpt_path = Path(args.ckpt) if args.ckpt else _find_default_ckpt()
    if ckpt_path is None or not ckpt_path.exists():
        raise RuntimeError(
            "No checkpoint found. Provide --ckpt runs/<exp>/best.pt. "
            "Without checkpoint, restoration cannot be diagnosed."
        )

    if args.config is None:
        raise ValueError("--config is required to construct the model")
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"--config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    runtime_cfg = get_runtime_defaults()
    runtime_cfg.update(cfg.get("runtime", {}))

    chosen_device = _pick_device(args.device)
    runtime_cfg["device"] = chosen_device
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

    t0 = int(args.t0) if args.t0 is not None else int(cfg["diffusion"].get("t0", 199))

    total_params, trainable_params = _count_trainable(model)
    phi_minus_abs_sum = _param_abs_sum(ckpt["phi_minus"])
    phi_plus_abs_sum = _param_abs_sum(ckpt["phi_plus"])

    print(f"[info] scene={scene_name} rate={rate_dir}")
    print(f"[info] checkpoint={ckpt_path}")
    print(f"[info] backbone={cfg['model']['pretrained_model_name_or_path']}")
    print(f"[info] t0={t0} adapter_scale={args.adapter_scale} disable_restore={args.disable_restore}")
    print(f"[info] selected={selected}")

    rows: List[Dict[str, object]] = []
    per_view_debug: Dict[str, object] = {}

    for idx, name in enumerate(selected):
        prompt = ""
        lq = _to_minus1_1_tensor(scene_dir / rate_dir / name, device=device)
        hq = _to_minus1_1_tensor(scene_dir / "clean" / name, device=device)

        lq_padded, pad_info = _pad_to_multiple(lq, m=8)
        hq_padded, _ = _pad_to_multiple(hq, m=8)

        result = run_restore_once(
            model=model,
            lq=lq_padded,
            prompts=[prompt],
            t0=t0,
            adapter_scale=float(args.adapter_scale),
            disable_restore=bool(args.disable_restore),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        restored = _unpad(result["restored"], pad_info)
        lq_for_metrics = lq
        hq_for_metrics = hq
        with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            lq_roundtrip_pad = model.decode_latents(model.encode_images(lq_padded))
        lq_roundtrip = _unpad(lq_roundtrip_pad, pad_info)

        diff = torch.mean(torch.abs(restored - lq_for_metrics)).item()
        diff_roundtrip = torch.mean(torch.abs(lq_roundtrip - lq_for_metrics)).item()
        lq_01 = ((lq_for_metrics + 1.0) * 0.5).float()
        hq_01 = ((hq_for_metrics + 1.0) * 0.5).float()
        restored_01 = ((restored + 1.0) * 0.5).float()
        roundtrip_01 = ((lq_roundtrip + 1.0) * 0.5).float()

        psnr_before = psnr(lq_01, hq_01)
        ssim_before = ssim(lq_01, hq_01)
        psnr_roundtrip = psnr(roundtrip_01, hq_01)
        ssim_roundtrip = ssim(roundtrip_01, hq_01)
        psnr_after = psnr(restored_01, hq_01)
        ssim_after = ssim(restored_01, hq_01)

        name_tag = f"view_{idx:03d}"
        hq_out = out_dir / "hq" / f"{name_tag}.png"
        lq_out = out_dir / "lq" / f"{name_tag}.png"
        rst_out = out_dir / "restored" / f"{name_tag}.png"
        _save_minus1_1(hq_out, hq_for_metrics)
        _save_minus1_1(lq_out, lq_for_metrics)
        _save_minus1_1(rst_out, restored)
        _save_triplet_mosaic(hq_out, lq_out, rst_out, f"Scene={scene_name} Rate={rate_dir} src={name}", out_dir / "mosaic" / f"{name_tag}.png")

        row = {
            "view_name": name,
            "source_index": int(Path(name).stem),
            "pad_left": pad_info["left"],
            "pad_right": pad_info["right"],
            "pad_top": pad_info["top"],
            "pad_bottom": pad_info["bottom"],
            "mean_abs_diff_restored_vs_lq": float(diff),
            "mean_abs_diff_vae_roundtrip_vs_lq": float(diff_roundtrip),
            "psnr_before": float(psnr_before),
            "ssim_before": float(ssim_before),
            "psnr_vae_roundtrip": float(psnr_roundtrip),
            "ssim_vae_roundtrip": float(ssim_roundtrip),
            "psnr_after": float(psnr_after),
            "ssim_after": float(ssim_after),
            "improved_psnr": bool(psnr_after > psnr_before),
            "improved_ssim": bool(ssim_after > ssim_before),
        }
        rows.append(row)

        per_view_debug[name] = {
            "lq_stats": _tensor_stats(lq_padded),
            "hq_stats": _tensor_stats(hq_padded),
            "lq_roundtrip_stats": _tensor_stats(_pad_to_multiple(lq_roundtrip, 8)[0]),
            "restored_stats": _tensor_stats(_pad_to_multiple(restored, 8)[0]),
            "z_lq_stats": _tensor_stats(result["z_lq"]) if result["z_lq"].numel() else None,
            "z_t0_stats": _tensor_stats(result["z_t0"]) if result["z_t0"].numel() else None,
            "eps_base_stats": _tensor_stats(result["eps_base"]) if result["eps_base"].numel() else None,
            "eps_minus_stats": _tensor_stats(result["eps_minus"]) if result["eps_minus"].numel() else None,
            "eps_used_stats": _tensor_stats(result["eps_used"]) if result["eps_used"].numel() else None,
            "z_hat_stats": _tensor_stats(result["z_hat"]) if result["z_hat"].numel() else None,
            "sigma_t0": float(model.sigma(torch.tensor([t0], device=device, dtype=torch.long))[0].item()),
        }

    summary = {
        "scene": scene_name,
        "artifact_scene_dir": str(scene_dir),
        "rate_dir": rate_dir,
        "checkpoint": str(ckpt_path),
        "config": str(cfg_path),
        "backbone_model": cfg["model"]["pretrained_model_name_or_path"],
        "t0": t0,
        "num_train_timesteps": int(cfg["diffusion"]["num_train_timesteps"]),
        "adapter_scale": float(args.adapter_scale),
        "disable_restore": bool(args.disable_restore),
        "selected_views": selected,
        "device": str(device),
        "mixed_precision": mp,
        "parameter_summary": {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "phi_minus_abs_sum": float(phi_minus_abs_sum),
            "phi_plus_abs_sum": float(phi_plus_abs_sum),
        },
        "rows": rows,
        "aggregate": {
            "avg_diff_restored_vs_lq": float(np.mean([r["mean_abs_diff_restored_vs_lq"] for r in rows])) if rows else float("nan"),
            "avg_diff_vae_roundtrip_vs_lq": float(np.mean([r["mean_abs_diff_vae_roundtrip_vs_lq"] for r in rows])) if rows else float("nan"),
            "avg_psnr_before": float(np.mean([r["psnr_before"] for r in rows])) if rows else float("nan"),
            "avg_psnr_vae_roundtrip": float(np.mean([r["psnr_vae_roundtrip"] for r in rows])) if rows else float("nan"),
            "avg_psnr_after": float(np.mean([r["psnr_after"] for r in rows])) if rows else float("nan"),
            "avg_ssim_before": float(np.mean([r["ssim_before"] for r in rows])) if rows else float("nan"),
            "avg_ssim_vae_roundtrip": float(np.mean([r["ssim_vae_roundtrip"] for r in rows])) if rows else float("nan"),
            "avg_ssim_after": float(np.mean([r["ssim_after"] for r in rows])) if rows else float("nan"),
            "num_psnr_improved": int(sum(1 for r in rows if r["improved_psnr"])),
            "num_ssim_improved": int(sum(1 for r in rows if r["improved_ssim"])),
        },
        "per_view_debug": per_view_debug,
    }

    (out_dir / "restoration_diag.json").write_text(json.dumps(summary, indent=2))
    _write_csv(out_dir / "restoration_diag.csv", rows)

    with (out_dir / "run_config.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "scene": args.scene,
                "rate": args.rate,
                "views": args.views,
                "num_views": args.num_views,
                "seed": args.seed,
                "ckpt": str(ckpt_path),
                "config": str(cfg_path),
                "t0": t0,
                "adapter_scale": args.adapter_scale,
                "disable_restore": args.disable_restore,
                "device": args.device,
            },
            f,
            sort_keys=False,
        )

    print(f"[done] wrote restoration diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
