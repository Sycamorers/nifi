#!/usr/bin/env python3
"""Prove restoration path is active and numerically changes outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel
from nifi.artifact_synthesis import ArtifactSynthesisCompressionConfig, ProxyArtifactSynthesisCompressor
from nifi.metrics.simple_metrics import psnr, ssim
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether Eq.7 restoration is applied")
    parser.add_argument("--dataset", type=str, default="mipnerf360")
    parser.add_argument("--scene", type=str, required=True, help="Scene name or artifact scene path")
    parser.add_argument("--view", type=int, default=0, help="View index in sorted PNG list")
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--model_paths", type=str, default="configs/model_paths.yaml")
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--t0", type=int, default=0)
    parser.add_argument("--t0_variant", type=str, default="custom", choices=["custom", "small", "large"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--disable_restore", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--min_abs_diff", type=float, default=1e-4)
    return parser.parse_args()


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _stats(x: torch.Tensor) -> Dict[str, object]:
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _to_minus1_1(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device) * 2.0 - 1.0


def _save_minus1_1(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img01 = ((x.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_abs_diff(path: Path, restored: torch.Tensor, lq: torch.Tensor, amplify: float = 10.0) -> None:
    # Visualized absolute difference map to prove changed output.
    diff = torch.abs(restored - lq).mean(dim=1, keepdim=False).squeeze(0).detach().cpu().numpy()
    vis = np.clip(diff * amplify, 0.0, 1.0)
    arr = np.clip(np.round(vis * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


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
    y = torch.nn.functional.pad(x, (left, right, top, bottom), mode="reflect")
    return y, {"left": left, "right": right, "top": top, "bottom": bottom}


def _unpad(x: torch.Tensor, pads: Dict[str, int]) -> torch.Tensor:
    l = int(pads["left"])
    r = int(pads["right"])
    t = int(pads["top"])
    b = int(pads["bottom"])
    if l == r == t == b == 0:
        return x
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]


def _resolve_t0(args: argparse.Namespace) -> int:
    if args.t0_variant == "small":
        return 0
    if args.t0_variant == "large":
        return 199
    return int(args.t0)


def _resolve_artifact_scene(dataset: str, scene: str, rate: float) -> Tuple[Path, str]:
    scene_path = Path(scene)
    if scene_path.exists() and (scene_path / "clean").exists():
        rate_dir = f"rate_{float(rate):.3f}"
        return scene_path.resolve(), rate_dir

    artifact_scene = ROOT / "artifacts" / scene
    rate_dir = f"rate_{float(rate):.3f}"
    if (artifact_scene / "clean").exists() and (artifact_scene / rate_dir).exists():
        return artifact_scene.resolve(), rate_dir

    dataset_scene = ROOT / "data" / dataset / scene
    if not dataset_scene.exists():
        raise FileNotFoundError(
            f"Could not resolve scene={scene}. Expected artifact scene or dataset at {dataset_scene}."
        )

    out_scene = ROOT / "artifacts" / scene
    out_scene.mkdir(parents=True, exist_ok=True)
    cfg = ArtifactSynthesisCompressionConfig(jpeg_quality_min=8, jpeg_quality_max=55, downsample_min=2, downsample_max=8)
    compressor = ProxyArtifactSynthesisCompressor(cfg)
    compressor.synthesize_scene_artifacts(
        scene_dir=dataset_scene,
        rates_lambda=[float(rate)],
        out_dir=out_scene,
        holdout_every=8,
        max_images=None,
    )
    return out_scene.resolve(), rate_dir


def _load_model_paths(path: Path) -> Tuple[Path, Path]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    restoration = payload.get("restoration", {})
    if not isinstance(restoration, dict):
        raise ValueError(f"`restoration` block missing in {path}")
    cfg = restoration.get("model_config")
    ckpt = restoration.get("adapter_checkpoint")
    if not isinstance(cfg, str) or not isinstance(ckpt, str):
        raise ValueError(f"`model_config` and `adapter_checkpoint` are required in {path}")
    return _as_abs(cfg), _as_abs(ckpt)


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("This diagnostic must run with --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable but --device cuda was requested")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    t0 = _resolve_t0(args)
    artifact_scene, rate_dir = _resolve_artifact_scene(dataset=args.dataset, scene=args.scene, rate=args.rate)
    clean_paths = sorted((artifact_scene / "clean").glob("*.png"))
    lq_paths = sorted((artifact_scene / rate_dir).glob("*.png"))
    shared = sorted({p.name for p in clean_paths} & {p.name for p in lq_paths})
    if not shared:
        raise RuntimeError(f"No shared clean/LQ views under {artifact_scene}")
    if args.view < 0 or args.view >= len(shared):
        raise ValueError(f"--view {args.view} is out of range [0, {len(shared)-1}]")

    name = shared[args.view]
    hq = _to_minus1_1(artifact_scene / "clean" / name, device=device)
    lq = _to_minus1_1(artifact_scene / rate_dir / name, device=device)

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "outputs" / "diag_restore" / f"{Path(args.scene).name}_view{args.view:03d}_t{t0}_s{args.adapter_scale:g}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_restore:
        restored = lq.clone()
        debug = {
            "z_lq_stats": None,
            "z_t0_stats": None,
            "eps_base_stats": None,
            "eps_minus_stats": None,
            "eps_used_stats": None,
            "z_hat_stats": None,
            "sigma_t0": float(t0),
            "timestep_index": int(t0),
        }
        checkpoint_path = None
        model_config_path = None
        model_name_or_path = None
    else:
        model_paths_file = _as_abs(args.model_paths)
        if not model_paths_file.exists():
            raise FileNotFoundError(f"model paths file not found: {model_paths_file}")
        model_config_path, checkpoint_path = _load_model_paths(model_paths_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"adapter checkpoint missing: {checkpoint_path}")
        if not model_config_path.exists():
            raise FileNotFoundError(f"model config missing: {model_config_path}")

        cfg = load_config(str(model_config_path))
        model_cfg = ArtifactRestorationDiffusionConfig(
            model_name_or_path=str(cfg["model"]["pretrained_model_name_or_path"]),
            num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
            lora_rank=int(cfg["model"]["lora_rank"]),
            guidance_scale=float(cfg["model"]["guidance_scale"]),
            prompt_dropout=float(cfg["model"]["prompt_dropout"]),
            max_token_length=int(cfg["model"]["max_token_length"]),
            vae_scaling_factor=float(cfg["model"]["vae_scaling_factor"]),
        )
        model_name_or_path = model_cfg.model_name_or_path

        mp = str(cfg.get("train", {}).get("mixed_precision", "fp16"))
        amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
        use_amp = mp in {"fp16", "bf16"}

        model = FrozenBackboneArtifactRestorationModel(model_cfg, device=device, dtype=amp_dtype if use_amp else torch.float32)
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        model.phi_minus.load_state_dict(ckpt["phi_minus"])
        model.phi_plus.load_state_dict(ckpt["phi_plus"])
        model.freeze_backbone()
        model.eval()

        lq_pad, pads = _pad_to_multiple(lq, m=8)
        b = lq_pad.shape[0]
        t = torch.full((b,), int(t0), device=device, dtype=torch.long)

        with torch.no_grad():
            z_lq = model.encode_images(lq_pad)
            z_t0, _ = model.q_sample(z_lq, t, noise=torch.zeros_like(z_lq))
            eps_base = model.predict_eps(z_t0, t, [args.prompt] * b, adapter_type=None, train_mode=False)
            eps_minus = model.predict_eps(z_t0, t, [args.prompt] * b, adapter_type="minus", train_mode=False)
            eps = eps_base + float(args.adapter_scale) * (eps_minus - eps_base)
            sigma_t0 = model.sigma(t).view(-1, 1, 1, 1)
            z_hat = z_t0 - sigma_t0 * eps
            restored = _unpad(model.decode_latents(z_hat), pads)

        debug = {
            "z_lq_stats": _stats(z_lq),
            "z_t0_stats": _stats(z_t0),
            "eps_base_stats": _stats(eps_base),
            "eps_minus_stats": _stats(eps_minus),
            "eps_used_stats": _stats(eps),
            "z_hat_stats": _stats(z_hat),
            "sigma_t0": float(sigma_t0.flatten()[0].item()),
            "timestep_index": int(t0),
        }

    mean_abs_diff = float(torch.mean(torch.abs(restored - lq)).item())
    mean_l2_diff = float(torch.mean((restored - lq) ** 2).item())

    hq01 = ((hq + 1.0) * 0.5).clamp(0.0, 1.0).float()
    lq01 = ((lq + 1.0) * 0.5).clamp(0.0, 1.0).float()
    rst01 = ((restored + 1.0) * 0.5).clamp(0.0, 1.0).float()
    summary = {
        "dataset": args.dataset,
        "scene": str(args.scene),
        "artifact_scene": str(artifact_scene),
        "view_name": name,
        "rate_dir": rate_dir,
        "device": str(device),
        "gpu_name": str(torch.cuda.get_device_name(device)),
        "disable_restore": bool(args.disable_restore),
        "t0": int(t0),
        "adapter_scale": float(args.adapter_scale),
        "model_paths": str(_as_abs(args.model_paths)),
        "model_config": None if model_config_path is None else str(model_config_path),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "backbone_model": model_name_or_path,
        "mean_abs_diff_restored_vs_lq": mean_abs_diff,
        "mean_l2_diff_restored_vs_lq": mean_l2_diff,
        "psnr_before": float(psnr(lq01, hq01)),
        "psnr_after": float(psnr(rst01, hq01)),
        "ssim_before": float(ssim(lq01, hq01)),
        "ssim_after": float(ssim(rst01, hq01)),
        "lq_stats": _stats(lq),
        "hq_stats": _stats(hq),
        "restored_stats": _stats(restored),
        "latent_debug": debug,
        "paths": {
            "hq": str((out_dir / "hq.png").resolve()),
            "lq": str((out_dir / "lq.png").resolve()),
            "restored": str((out_dir / "restored.png").resolve()),
            "abs_diff": str((out_dir / "abs_diff_amplified.png").resolve()),
        },
    }

    _save_minus1_1(out_dir / "hq.png", hq)
    _save_minus1_1(out_dir / "lq.png", lq)
    _save_minus1_1(out_dir / "restored.png", restored)
    _save_abs_diff(out_dir / "abs_diff_amplified.png", restored, lq, amplify=10.0)
    (out_dir / "restoration_diag.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"[done] wrote diagnostic bundle to {out_dir}")

    if not args.disable_restore and mean_abs_diff < float(args.min_abs_diff):
        raise RuntimeError(
            f"Restored output is too close to LQ (mean_abs_diff={mean_abs_diff:.8f} < min_abs_diff={args.min_abs_diff})"
        )


if __name__ == "__main__":
    main()
