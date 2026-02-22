#!/usr/bin/env python3
"""Shared Garden-only helpers for diagnostics, training, and demo scripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy as np
import torch
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel
from nifi.artifact_synthesis import ArtifactSynthesisCompressionConfig, ProxyArtifactSynthesisCompressor
from nifi.metrics.simple_metrics import ssim
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def read_model_paths(model_paths: Path) -> Tuple[Path, Path]:
    payload = yaml.safe_load(model_paths.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML in {model_paths}")
    restoration = payload.get("restoration", {})
    if not isinstance(restoration, dict):
        raise ValueError(f"`restoration` mapping missing in {model_paths}")
    cfg = restoration.get("model_config")
    ckpt = restoration.get("adapter_checkpoint")
    if not isinstance(cfg, str) or not isinstance(ckpt, str):
        raise ValueError(f"`model_config` and `adapter_checkpoint` are required in {model_paths}")
    return _as_abs(cfg), _as_abs(ckpt)


def _is_artifact_scene(path: Path) -> bool:
    return path.is_dir() and (path / "clean").exists()


def ensure_garden_artifact_scene(
    *,
    dataset: str = "mipnerf360",
    scene: str = "garden",
    rates: Sequence[float] = (0.1, 0.5, 1.0),
    out_root: Optional[Path] = None,
) -> Path:
    artifact_scene = ROOT / "artifacts" / scene if out_root is None else out_root / scene
    have_all = _is_artifact_scene(artifact_scene) and all((artifact_scene / f"rate_{float(r):.3f}").exists() for r in rates)
    if have_all:
        return artifact_scene.resolve()

    dataset_scene = ROOT / "data" / dataset / scene
    if not dataset_scene.exists():
        raise FileNotFoundError(
            f"Garden dataset scene missing: {dataset_scene}. "
            "Expected source images under data/mipnerf360/garden."
        )

    artifact_scene.mkdir(parents=True, exist_ok=True)
    compression_cfg = ArtifactSynthesisCompressionConfig(
        jpeg_quality_min=8,
        jpeg_quality_max=55,
        downsample_min=2,
        downsample_max=8,
    )
    compressor = ProxyArtifactSynthesisCompressor(compression_cfg)
    compressor.synthesize_scene_artifacts(
        scene_dir=dataset_scene,
        rates_lambda=[float(r) for r in rates],
        out_dir=artifact_scene,
        holdout_every=8,
        max_images=None,
    )
    return artifact_scene.resolve()


def ensure_rate_dir(artifact_scene: Path, rate: float) -> str:
    rate_dir = f"rate_{float(rate):.3f}"
    full = artifact_scene / rate_dir
    if full.exists():
        return rate_dir

    compressor = ProxyArtifactSynthesisCompressor(
        ArtifactSynthesisCompressionConfig(
            jpeg_quality_min=8,
            jpeg_quality_max=55,
            downsample_min=2,
            downsample_max=8,
        )
    )
    full.mkdir(parents=True, exist_ok=True)
    for src in sorted((artifact_scene / "clean").glob("*.png")):
        with Image.open(src) as img:
            degraded = compressor.synthesize_view_artifact(img.convert("RGB"), rate_lambda=float(rate))
            degraded.save(full / src.name)
    return rate_dir


def load_metadata(artifact_scene: Path) -> Dict[str, object]:
    meta = artifact_scene / "metadata.json"
    if not meta.exists():
        return {}
    payload = json.loads(meta.read_text())
    if not isinstance(payload, dict):
        return {}
    return payload


def resolve_view_names(artifact_scene: Path, rate_dir: str, requested_indices: Optional[Iterable[int]] = None) -> List[str]:
    clean = sorted((artifact_scene / "clean").glob("*.png"))
    degraded = sorted((artifact_scene / rate_dir).glob("*.png"))
    shared = sorted({p.name for p in clean} & {p.name for p in degraded})
    if not shared:
        raise RuntimeError(f"No clean/LQ overlap in {artifact_scene} and {rate_dir}")

    if requested_indices is None:
        meta = load_metadata(artifact_scene)
        splits = meta.get("splits", {}) if isinstance(meta, dict) else {}
        test = splits.get("test", []) if isinstance(splits, dict) else []
        if isinstance(test, list) and test:
            keep = [str(x) for x in test if str(x) in shared]
            if keep:
                return keep
        return shared

    selected: List[str] = []
    for i in requested_indices:
        name = f"{int(i):05d}.png"
        if name not in shared:
            raise ValueError(f"Requested view index {i} not available in scene")
        selected.append(name)
    return selected


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


def pad_to_multiple(x: torch.Tensor, m: int = 8) -> Tuple[torch.Tensor, Dict[str, int]]:
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


def unpad(x: torch.Tensor, pads: Dict[str, int]) -> torch.Tensor:
    l, r, t, b = pads["left"], pads["right"], pads["top"], pads["bottom"]
    if l == r == t == b == 0:
        return x
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]


def laplacian_variance(x_minus1_1: torch.Tensor) -> float:
    gray = ((x_minus1_1.clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1, keepdim=True).float()
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    y = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return float(y.var().item())


@dataclass
class RestorerBundle:
    model: FrozenBackboneArtifactRestorationModel
    model_cfg: ArtifactRestorationDiffusionConfig
    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    checkpoint_path: Path
    model_config_path: Path


def build_restorer(
    *,
    model_paths: Path,
    device_flag: str = "cuda",
) -> RestorerBundle:
    model_config_path, checkpoint_path = read_model_paths(model_paths)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config missing: {model_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint missing: {checkpoint_path}")

    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA unavailable")
        device = torch.device("cuda:0")
    elif device_flag == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    mp = str(cfg.get("train", {}).get("mixed_precision", "fp16"))
    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    use_amp = device.type == "cuda" and mp in {"fp16", "bf16"}

    model = FrozenBackboneArtifactRestorationModel(model_cfg, device=device, dtype=amp_dtype if use_amp else torch.float32)
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model.phi_minus.load_state_dict(ckpt["phi_minus"])
    model.phi_plus.load_state_dict(ckpt["phi_plus"])
    model.freeze_backbone()
    model.eval()
    return RestorerBundle(
        model=model,
        model_cfg=model_cfg,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path,
    )


def restore_eq7(
    *,
    restorer: RestorerBundle,
    lq_minus1_1: torch.Tensor,
    prompt: str = "",
    t0: int = 199,
    adapter_scale: float = 1.0,
) -> torch.Tensor:
    x_pad, pads = pad_to_multiple(lq_minus1_1, m=8)
    b = x_pad.shape[0]
    t = torch.full((b,), int(t0), device=restorer.device, dtype=torch.long)
    with torch.no_grad(), torch.autocast(device_type=restorer.device.type, dtype=restorer.amp_dtype, enabled=restorer.use_amp):
        z = restorer.model.encode_images(x_pad)
        z_t, _ = restorer.model.q_sample(z, t, noise=torch.zeros_like(z))
        eps_base = restorer.model.predict_eps(z_t, t, [prompt] * b, adapter_type=None, train_mode=False)
        eps_minus = restorer.model.predict_eps(z_t, t, [prompt] * b, adapter_type="minus", train_mode=False)
        eps = eps_base + float(adapter_scale) * (eps_minus - eps_base)
        sigma = restorer.model.sigma(t).view(-1, 1, 1, 1)
        z_hat = z_t - sigma * eps
        restored = restorer.model.decode_latents(z_hat)
    return unpad(restored, pads)


def ssim_gain(hq_minus1_1: torch.Tensor, lq_minus1_1: torch.Tensor, restored_minus1_1: torch.Tensor) -> float:
    hq01 = ((hq_minus1_1 + 1.0) * 0.5).clamp(0.0, 1.0).float()
    lq01 = ((lq_minus1_1 + 1.0) * 0.5).clamp(0.0, 1.0).float()
    rst01 = ((restored_minus1_1 + 1.0) * 0.5).clamp(0.0, 1.0).float()
    return float(ssim(rst01, hq01) - ssim(lq01, hq01))
