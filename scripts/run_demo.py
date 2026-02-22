#!/usr/bin/env python3
"""Canonical demo runner for real-data qualitative results.

Pipeline order:
1) HQ render/export
2) LQ render/export
3) restoration
4) per-view triptych export
"""

from __future__ import annotations

import argparse
import json
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import autocast
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy as np
from PIL import Image, ImageFilter
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from nifi.artifact_synthesis import ArtifactSynthesisCompressionConfig, ProxyArtifactSynthesisCompressor
from nifi.metrics.perceptual_metrics import PerceptualMetrics
from nifi.metrics.simple_metrics import ssim
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config
from nifi.utils.runtime import configure_runtime, get_runtime_defaults, resolve_device
from nifi.utils.seed import set_seed
from nifi.utils.triptych import save_labeled_triptych


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic real-data NiFi demo")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key, e.g. mipnerf360")
    parser.add_argument("--scene", type=str, default=None, help="Scene name or path (defaults to 'garden' when --dataset garden)")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--views", nargs="*", default=["auto"], help='View indices list or "auto"')
    parser.add_argument("--rate", type=str, default=None, help='Rate preset/float, e.g. "low" or "0.1"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--config", type=str, default="configs/demo_realdata_known_good.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override restoration adapter checkpoint path")
    parser.add_argument("--model_config", type=str, default=None, help="Override restoration model config path")
    parser.add_argument("--allow_gate_fail", action="store_true", help="Do not fail hard when quality gate fails")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return payload


def _count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _abs_weight_sum(module: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for p in module.parameters():
            total += float(p.detach().abs().sum().cpu().item())
    return total


def _is_artifact_scene(path: Path) -> bool:
    return path.is_dir() and (path / "clean").exists()


def _to_rate_dir(rate_value: float) -> str:
    return f"rate_{float(rate_value):.3f}"


def _resolve_rate_value(rate_arg: Optional[str], demo_cfg: Dict[str, object]) -> float:
    presets = demo_cfg.get("rate_presets", {})
    if not isinstance(presets, dict):
        presets = {}

    raw = str(rate_arg) if rate_arg is not None else str(demo_cfg.get("default_rate", "0.1"))
    raw = raw.strip()

    if raw in presets:
        return float(presets[raw])

    if raw.startswith("rate_"):
        return float(raw.split("rate_", 1)[1])

    return float(raw)


def _parse_views(view_tokens: Sequence[str]) -> Optional[List[int]]:
    tokens = [str(v).strip() for v in view_tokens if str(v).strip()]
    if not tokens or (len(tokens) == 1 and tokens[0].lower() == "auto"):
        return None

    out: List[int] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def _select_evenly(items: Sequence[str], n: int, seed: int) -> List[str]:
    vals = list(items)
    if n <= 0:
        raise ValueError("auto_num_views must be > 0")
    if not vals:
        return []
    if n >= len(vals):
        return vals

    offset = int(seed) % len(vals)
    vals = vals[offset:] + vals[:offset]

    if n == 1:
        return [vals[0]]

    out: List[str] = []
    used = set()
    span = len(vals) - 1
    for i in range(n):
        idx = round(i * span / (n - 1))
        name = vals[idx]
        if name not in used:
            out.append(name)
            used.add(name)

    if len(out) < n:
        for name in vals:
            if name not in used:
                out.append(name)
                used.add(name)
            if len(out) == n:
                break
    return out


def _load_test_split(artifact_scene: Path) -> List[str]:
    meta_path = artifact_scene / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        payload = json.loads(meta_path.read_text())
        splits = payload.get("splits", {}) if isinstance(payload, dict) else {}
        test = splits.get("test", []) if isinstance(splits, dict) else []
        if isinstance(test, list):
            return [str(x) for x in test]
    except Exception:
        pass
    return []


def _resolve_dataset_scene_dir(dataset_key: str, scene_arg: str) -> Optional[Path]:
    candidate = Path(scene_arg)
    if candidate.exists() and candidate.is_dir() and not _is_artifact_scene(candidate):
        return candidate.resolve()

    checks = [
        ROOT / "data" / dataset_key / scene_arg,
        ROOT / "data" / "benchmarks" / dataset_key / scene_arg,
        ROOT / "data" / "mipnerf360" / scene_arg,
    ]

    if dataset_key in {"deepblending", "tanks_temples"}:
        checks.extend(
            [
                ROOT / "data" / "tanks_temples_deepblending" / scene_arg,
                ROOT / "data" / "benchmarks" / "tanks_temples_deepblending" / scene_arg,
            ]
        )

    for p in checks:
        if p.exists() and p.is_dir():
            return p.resolve()

    # Last fallback: scan data roots for matching leaf folder name.
    scan_roots = [ROOT / "data", ROOT / "data" / "benchmarks"]
    for root in scan_roots:
        if not root.exists():
            continue
        for p in root.rglob(scene_arg):
            if p.is_dir() and any((p / d).exists() for d in ("images", "images_4", "images_2", "rgb", "train")):
                return p.resolve()

    return None


def _resolve_artifact_scene(
    *,
    dataset_key: str,
    scene_arg: str,
    out_dir: Path,
    rate_value: float,
    compression_cfg: ArtifactSynthesisCompressionConfig,
) -> Path:
    candidate = Path(scene_arg)
    if _is_artifact_scene(candidate):
        return candidate.resolve()

    by_name = ROOT / "artifacts" / scene_arg
    if _is_artifact_scene(by_name):
        return by_name.resolve()

    dataset_scene = _resolve_dataset_scene_dir(dataset_key=dataset_key, scene_arg=scene_arg)
    if dataset_scene is None:
        raise FileNotFoundError(
            f"Could not resolve scene '{scene_arg}' for dataset '{dataset_key}'. "
            "Expected artifact dir or data/<dataset>/<scene>."
        )

    artifact_scene = out_dir / "_artifacts" / dataset_scene.name
    rate_dir = artifact_scene / _to_rate_dir(rate_value)
    if _is_artifact_scene(artifact_scene) and rate_dir.exists():
        return artifact_scene.resolve()

    artifact_scene.mkdir(parents=True, exist_ok=True)
    compressor = ProxyArtifactSynthesisCompressor(compression_cfg)
    meta = compressor.synthesize_scene_artifacts(
        scene_dir=dataset_scene,
        rates_lambda=[rate_value],
        out_dir=artifact_scene,
        holdout_every=8,
        max_images=None,
    )
    print(
        "[info] generated artifacts from dataset scene:",
        json.dumps(
            {
                "dataset_scene": str(dataset_scene),
                "artifact_scene": str(artifact_scene),
                "num_images": int(meta.get("num_images", 0)),
                "rates": meta.get("rates", []),
            },
            indent=2,
        ),
    )
    return artifact_scene.resolve()


def _ensure_rate_dir(
    artifact_scene: Path,
    rate_value: float,
    compression_cfg: ArtifactSynthesisCompressionConfig,
) -> str:
    rate_dir = _to_rate_dir(rate_value)
    full = artifact_scene / rate_dir
    if full.exists():
        return rate_dir

    clean_paths = sorted((artifact_scene / "clean").glob("*.png"))
    if not clean_paths:
        raise RuntimeError(f"No clean images found under {artifact_scene / 'clean'}")

    compressor = ProxyArtifactSynthesisCompressor(compression_cfg)
    full.mkdir(parents=True, exist_ok=True)
    for src in clean_paths:
        with Image.open(src) as img:
            degraded = compressor.synthesize_view_artifact(img.convert("RGB"), rate_lambda=rate_value)
            degraded.save(full / src.name)

    return rate_dir


def _to_minus1_1(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device) * 2.0 - 1.0


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
    out = torch.nn.functional.pad(x, (left, right, top, bottom), mode="reflect")
    return out, {"left": left, "right": right, "top": top, "bottom": bottom}


def _unpad(x: torch.Tensor, pads: Dict[str, int]) -> torch.Tensor:
    l, r, t, b = pads["left"], pads["right"], pads["top"], pads["bottom"]
    if l == r == t == b == 0:
        return x
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]


def _laplacian_variance(x_minus1_1: torch.Tensor) -> float:
    gray = ((x_minus1_1.clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1, keepdim=True).float()
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1)
    return float(lap.var().item())


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


def _pick_device(device_flag: str) -> str:
    if device_flag == "cpu":
        return "cpu"
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DemoRestorer:
    model: torch.nn.Module
    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype

    def restore_eq7(self, lq: torch.Tensor, prompt: str, t0: int, adapter_scale: float) -> torch.Tensor:
        lq_pad, pads = _pad_to_multiple(lq, m=8)
        b = lq_pad.shape[0]
        t = torch.full((b,), int(t0), device=self.device, dtype=torch.long)
        with torch.no_grad(), autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            z_lq = self.model.encode_images(lq_pad)
            z_t0, _ = self.model.q_sample(z_lq, t, noise=torch.zeros_like(z_lq))
            eps_base = self.model.predict_eps(z_t0, t, [prompt] * b, adapter_type=None, train_mode=False)
            eps_minus = self.model.predict_eps(z_t0, t, [prompt] * b, adapter_type="minus", train_mode=False)
            eps = eps_base + float(adapter_scale) * (eps_minus - eps_base)
            sigma = self.model.sigma(t).view(-1, 1, 1, 1)
            z_hat = z_t0 - sigma * eps
            restored = self.model.decode_latents(z_hat)
        return _unpad(restored, pads)


def _build_restorer(
    checkpoint_path: Path,
    model_config_path: Path,
    device_flag: str,
    require_checkpoint: bool = False,
) -> Optional[DemoRestorer]:
    from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel

    if not checkpoint_path.exists():
        msg = f"checkpoint not found: {checkpoint_path}"
        if require_checkpoint:
            raise FileNotFoundError(msg)
        print(f"[warn] {msg}; disabling nifi_eq7")
        return None
    if not model_config_path.exists():
        msg = f"model config not found: {model_config_path}"
        if require_checkpoint:
            raise FileNotFoundError(msg)
        print(f"[warn] {msg}; disabling nifi_eq7")
        return None

    cfg = load_config(str(model_config_path))
    runtime_cfg = get_runtime_defaults()
    runtime_cfg.update(cfg.get("runtime", {}))
    runtime_cfg["deterministic"] = True
    runtime_cfg["cudnn_benchmark"] = False
    runtime_cfg["device"] = _pick_device(device_flag)
    configure_runtime(runtime_cfg)
    device = resolve_device(runtime_cfg)

    mp = str(cfg.get("train", {}).get("mixed_precision", "fp16"))
    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    use_amp = device.type == "cuda" and mp in {"bf16", "fp16"}

    ckpt = load_checkpoint(str(checkpoint_path), map_location="cpu")
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
    try:
        model.phi_minus.load_state_dict(ckpt["phi_minus"])
        model.phi_plus.load_state_dict(ckpt["phi_plus"])
    except Exception as exc:
        msg = (
            f"adapter checkpoint incompatible with model config; checkpoint={checkpoint_path} "
            f"model_config={model_config_path} error={exc}"
        )
        if require_checkpoint:
            raise RuntimeError(msg) from exc
        print(f"[warn] {msg}; disabling nifi_eq7")
        return None
    model.freeze_backbone()
    model.eval()

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_config": str(model_config_path),
        "model_name_or_path": str(model_cfg.model_name_or_path),
        "device": str(device),
        "use_amp": bool(use_amp),
        "phi_minus_params": _count_params(model.phi_minus),
        "phi_plus_params": _count_params(model.phi_plus),
        "phi_minus_abs_weight_sum": _abs_weight_sum(model.phi_minus),
        "phi_plus_abs_weight_sum": _abs_weight_sum(model.phi_plus),
    }
    print("[info] loaded restoration checkpoint")
    print(json.dumps(summary, indent=2))

    return DemoRestorer(model=model, device=device, use_amp=use_amp, amp_dtype=amp_dtype)


def _apply_classical_unsharp(lq_tensor: torch.Tensor, classical_cfg: Dict[str, object]) -> torch.Tensor:
    img01 = ((lq_tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")

    smooth_passes = int(classical_cfg.get("smooth_passes", 1))
    for _ in range(max(0, smooth_passes)):
        pil = pil.filter(ImageFilter.SMOOTH_MORE)

    pil = pil.filter(
        ImageFilter.UnsharpMask(
            radius=float(classical_cfg.get("unsharp_radius", 1.6)),
            percent=int(classical_cfg.get("unsharp_percent", 200)),
            threshold=int(classical_cfg.get("unsharp_threshold", 2)),
        )
    )

    out = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).to(device=lq_tensor.device) * 2.0 - 1.0


@dataclass
class MetricState:
    lpips: Optional[float]
    dists: Optional[float]
    ssim: float
    sharpness: float


class MetricEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.perceptual = None
        self.init_error = None
        try:
            self.perceptual = PerceptualMetrics(device=device)
        except Exception as exc:
            self.init_error = str(exc)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> MetricState:
        lpips_val: Optional[float] = None
        dists_val: Optional[float] = None
        if self.perceptual is not None:
            try:
                lpips_val = float(self.perceptual.lpips(pred, target).item())
                dists_val = float(self.perceptual.dists(pred, target).item())
            except Exception:
                lpips_val = None
                dists_val = None

        pred01 = ((pred.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
        target01 = ((target.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
        ssim_val = float(ssim(pred01, target01))
        sharp = _laplacian_variance(pred)
        return MetricState(lpips=lpips_val, dists=dists_val, ssim=ssim_val, sharpness=sharp)


def _quality_gate(
    *,
    before: MetricState,
    after: MetricState,
    gate_cfg: Dict[str, object],
) -> Tuple[bool, str, float]:
    min_lpips_margin = float(gate_cfg.get("min_lpips_margin", 0.01))
    min_dists_margin = float(gate_cfg.get("min_dists_margin", 0.005))
    min_ssim_gain = float(gate_cfg.get("min_ssim_gain", 0.0))
    require_sharpness_non_decrease = bool(gate_cfg.get("require_sharpness_non_decrease", True))

    metric_name = "ssim"
    metric_gain = after.ssim - before.ssim
    perceptual_ok = metric_gain >= min_ssim_gain

    if before.lpips is not None and after.lpips is not None:
        metric_name = "lpips"
        metric_gain = before.lpips - after.lpips
        perceptual_ok = metric_gain >= min_lpips_margin
    elif before.dists is not None and after.dists is not None:
        metric_name = "dists"
        metric_gain = before.dists - after.dists
        perceptual_ok = metric_gain >= min_dists_margin

    sharp_ok = (after.sharpness >= before.sharpness) if require_sharpness_non_decrease else True
    passed = bool(perceptual_ok and sharp_ok)
    reason = (
        f"metric={metric_name}, gain={metric_gain:.6f}, "
        f"sharpness={before.sharpness:.6f}->{after.sharpness:.6f}, pass={passed}"
    )
    return passed, reason, metric_gain


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")
    cfg = _load_yaml(config_path)

    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    demo_cfg = cfg.get("demo", {}) if isinstance(cfg.get("demo", {}), dict) else {}
    restoration_cfg = cfg.get("restoration", {}) if isinstance(cfg.get("restoration", {}), dict) else {}
    gate_cfg = cfg.get("quality_gate", {}) if isinstance(cfg.get("quality_gate", {}), dict) else {}
    compression_cfg_raw = cfg.get("compression", {}) if isinstance(cfg.get("compression", {}), dict) else {}

    dataset_key = str(args.dataset or dataset_cfg.get("name", "mipnerf360"))
    scene_arg = str(args.scene) if args.scene is not None else ("garden" if dataset_key.lower() == "garden" else "")
    if not scene_arg:
        raise ValueError("--scene is required unless --dataset garden is used")
    scene_name = Path(scene_arg).name

    seed = int(args.seed)
    set_seed(seed, deterministic=True)
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "outputs" / "demo" / dataset_key / scene_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    compression_cfg = ArtifactSynthesisCompressionConfig(
        jpeg_quality_min=int(compression_cfg_raw.get("jpeg_quality_min", 8)),
        jpeg_quality_max=int(compression_cfg_raw.get("jpeg_quality_max", 55)),
        downsample_min=int(compression_cfg_raw.get("downsample_min", 2)),
        downsample_max=int(compression_cfg_raw.get("downsample_max", 8)),
    )

    rate_value = _resolve_rate_value(args.rate, demo_cfg)
    artifact_scene = _resolve_artifact_scene(
        dataset_key=dataset_key,
        scene_arg=scene_arg,
        out_dir=out_dir,
        rate_value=rate_value,
        compression_cfg=compression_cfg,
    )
    rate_dir = _ensure_rate_dir(artifact_scene=artifact_scene, rate_value=rate_value, compression_cfg=compression_cfg)

    shared = sorted({p.name for p in (artifact_scene / "clean").glob("*.png")} & {p.name for p in (artifact_scene / rate_dir).glob("*.png")})
    if not shared:
        raise RuntimeError(f"No shared clean/LQ views found in {artifact_scene}")

    explicit_views = _parse_views(args.views)
    if explicit_views is None:
        desired = int(demo_cfg.get("auto_num_views", 4))
        prefer_test = bool(demo_cfg.get("prefer_test_split", True))
        test = [name for name in _load_test_split(artifact_scene) if name in shared]
        pool = test if prefer_test and len(test) >= desired else shared
        selected = _select_evenly(pool, desired, seed=seed)
    else:
        selected = []
        for v in explicit_views:
            name = f"{int(v):05d}.png"
            if name not in shared:
                raise ValueError(f"Requested view {v} missing from scene")
            selected.append(name)

    model_paths_rel = restoration_cfg.get("model_paths", "configs/model_paths.yaml")
    model_paths_path = Path(model_paths_rel) if Path(model_paths_rel).is_absolute() else ROOT / str(model_paths_rel)
    model_paths_payload: Dict[str, object] = {}
    if model_paths_path.exists():
        model_paths_payload = _load_yaml(model_paths_path)

    model_paths_restoration = (
        model_paths_payload.get("restoration", {}) if isinstance(model_paths_payload.get("restoration", {}), dict) else {}
    )

    checkpoint_rel = args.checkpoint or restoration_cfg.get(
        "checkpoint",
        model_paths_restoration.get("adapter_checkpoint", "demo_assets/checkpoints/nifi_tiny_best.pt"),
    )
    model_config_rel = args.model_config or restoration_cfg.get(
        "model_config",
        model_paths_restoration.get("model_config", "configs/default.yaml"),
    )
    t0 = int(restoration_cfg.get("t0", 199))
    adapter_scale = float(restoration_cfg.get("adapter_scale", 1.0))
    detail_boost = float(restoration_cfg.get("detail_boost", 0.0))
    detail_sigma = float(restoration_cfg.get("detail_sigma", 1.2))
    prompt = str(restoration_cfg.get("prompt", ""))
    require_nifi_restorer = bool(restoration_cfg.get("require_nifi_restorer", False))
    final_output = str(restoration_cfg.get("final_output", "quality_gate_select"))
    candidate_order = restoration_cfg.get("candidate_order", ["nifi_eq7", "classical_unsharp", "identity_lq"])
    if not isinstance(candidate_order, list):
        candidate_order = ["nifi_eq7", "classical_unsharp", "identity_lq"]
    classical_cfg = restoration_cfg.get("classical", {}) if isinstance(restoration_cfg.get("classical", {}), dict) else {}

    checkpoint_path = Path(checkpoint_rel) if Path(checkpoint_rel).is_absolute() else ROOT / str(checkpoint_rel)
    model_config_path = Path(model_config_rel) if Path(model_config_rel).is_absolute() else ROOT / str(model_config_rel)
    needs_nifi = final_output == "nifi_eq7" or ("nifi_eq7" in candidate_order)
    restorer = None
    if needs_nifi:
        restorer = _build_restorer(
            checkpoint_path=checkpoint_path,
            model_config_path=model_config_path,
            device_flag=args.device,
            require_checkpoint=require_nifi_restorer,
        )
    device = restorer.device if restorer is not None else torch.device(_pick_device(args.device))

    metrics_engine = MetricEngine(device=device)

    print("[info] run configuration")
    print(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "dataset": dataset_key,
                "scene": scene_name,
                "artifact_scene": str(artifact_scene),
                "rate": rate_dir,
                "selected_views": selected,
                "seed": seed,
                "device": str(device),
                "restorer_available": restorer is not None,
                "needs_nifi": needs_nifi,
                "require_nifi_restorer": require_nifi_restorer,
                "t0": t0,
                "adapter_scale": adapter_scale,
                "detail_boost": detail_boost,
                "detail_sigma": detail_sigma,
                "final_output": final_output,
                "candidate_order": candidate_order,
                "model_paths": str(model_paths_path),
            },
            indent=2,
        )
    )

    hq_dir = out_dir / "hq"
    lq_dir = out_dir / "lq"
    restored_dir = out_dir / "restored"
    triptych_dir = out_dir / "triptych"
    for d in (hq_dir, lq_dir, restored_dir, triptych_dir):
        d.mkdir(parents=True, exist_ok=True)

    gate_enabled = bool(gate_cfg.get("enabled", True))
    fail_on_violation = bool(gate_cfg.get("fail_on_violation", True))
    if args.allow_gate_fail:
        fail_on_violation = False

    rows: List[Dict[str, object]] = []
    all_passed = True

    for k, source_name in enumerate(selected):
        hq = _to_minus1_1(artifact_scene / "clean" / source_name, device=device)
        lq = _to_minus1_1(artifact_scene / rate_dir / source_name, device=device)

        candidates: Dict[str, torch.Tensor] = {
            "identity_lq": lq,
            "classical_unsharp": _apply_classical_unsharp(lq, classical_cfg),
        }
        if restorer is not None:
            nifi_out = restorer.restore_eq7(lq, prompt=prompt, t0=t0, adapter_scale=adapter_scale)
            nifi_out = _inject_lq_detail(nifi_out, lq, amount=detail_boost, sigma=detail_sigma)
            candidates["nifi_eq7"] = nifi_out

        baseline = metrics_engine.evaluate(lq, hq)
        candidate_metrics: Dict[str, MetricState] = {
            name: metrics_engine.evaluate(tensor, hq) for name, tensor in candidates.items()
        }

        ordered_candidates = [name for name in candidate_order if name in candidates]
        if not ordered_candidates:
            ordered_candidates = list(candidates.keys())

        chosen_name = ordered_candidates[0]
        chosen_gain = -1e9
        chosen_gate_pass = False
        chosen_gate_reason = ""

        if final_output == "quality_gate_select":
            valid = []
            for name in ordered_candidates:
                cand = candidate_metrics[name]
                passed, reason, gain = _quality_gate(before=baseline, after=cand, gate_cfg=gate_cfg)
                if passed:
                    valid.append((gain, name, reason))
            if valid:
                valid.sort(reverse=True)
                chosen_gain, chosen_name, chosen_gate_reason = valid[0]
                chosen_gate_pass = True
            else:
                # pick the highest metric gain even if failed, but keep failure state
                scored = []
                for name in ordered_candidates:
                    cand = candidate_metrics[name]
                    passed, reason, gain = _quality_gate(before=baseline, after=cand, gate_cfg=gate_cfg)
                    scored.append((gain, name, passed, reason))
                scored.sort(reverse=True)
                chosen_gain, chosen_name, chosen_gate_pass, chosen_gate_reason = scored[0]
        else:
            if final_output not in candidates:
                raise ValueError(f"final_output='{final_output}' is unavailable; candidates={list(candidates.keys())}")
            chosen_name = final_output
            passed, reason, gain = _quality_gate(before=baseline, after=candidate_metrics[chosen_name], gate_cfg=gate_cfg)
            chosen_gain = gain
            chosen_gate_pass = passed
            chosen_gate_reason = reason

        chosen = candidates[chosen_name]
        if gate_enabled and not chosen_gate_pass:
            all_passed = False

        view_name = f"view_{k:03d}.png"
        hq_out = hq_dir / view_name
        lq_out = lq_dir / view_name
        restored_out = restored_dir / view_name
        triptych_out = triptych_dir / view_name

        _save_minus1_1(hq_out, hq)
        _save_minus1_1(lq_out, lq)
        _save_minus1_1(restored_out, chosen)

        title = f"Scene: {scene_name} | Dataset: {dataset_key} | Rate: {rate_dir} | Source view: {source_name}"
        save_labeled_triptych(
            hq_path=hq_out,
            lq_path=lq_out,
            restored_path=restored_out,
            out_path=triptych_out,
            title=title,
        )

        row = {
            "k": k,
            "view_name": view_name,
            "source_name": source_name,
            "source_index": int(Path(source_name).stem),
            "chosen_candidate": chosen_name,
            "quality_gate_pass": bool(chosen_gate_pass),
            "quality_gate_reason": chosen_gate_reason,
            "quality_gate_gain": float(chosen_gain),
            "baseline": {
                "lpips": baseline.lpips,
                "dists": baseline.dists,
                "ssim": baseline.ssim,
                "sharpness": baseline.sharpness,
            },
            "candidate_metrics": {
                name: {
                    "lpips": state.lpips,
                    "dists": state.dists,
                    "ssim": state.ssim,
                    "sharpness": state.sharpness,
                }
                for name, state in candidate_metrics.items()
            },
            "paths": {
                "hq": str(hq_out.resolve()),
                "lq": str(lq_out.resolve()),
                "restored": str(restored_out.resolve()),
                "triptych": str(triptych_out.resolve()),
            },
        }
        rows.append(row)
        print(f"[view {k:03d}] src={source_name} chosen={chosen_name} gate={chosen_gate_pass} {chosen_gate_reason}")

    manifest = {
        "config": str(config_path.resolve()),
        "dataset": dataset_key,
        "scene": scene_name,
        "artifact_scene": str(artifact_scene.resolve()),
        "out_dir": str(out_dir.resolve()),
        "rate_dir": rate_dir,
        "rate_value": float(rate_value),
        "seed": seed,
        "device": str(device),
        "restoration": {
            "checkpoint": str(checkpoint_path),
            "model_config": str(model_config_path),
            "model_paths": str(model_paths_path),
            "t0": t0,
            "adapter_scale": adapter_scale,
            "detail_boost": detail_boost,
            "detail_sigma": detail_sigma,
            "final_output": final_output,
            "candidate_order": ordered_candidates if rows else [],
            "restorer_available": restorer is not None,
            "needs_nifi": needs_nifi,
            "require_nifi_restorer": require_nifi_restorer,
        },
        "quality_gate": {
            "enabled": gate_enabled,
            "fail_on_violation": fail_on_violation,
            "all_views_passed": bool(all_passed),
            "lpips_available": rows[0]["baseline"]["lpips"] is not None if rows else False,
            "dists_available": rows[0]["baseline"]["dists"] is not None if rows else False,
            "perceptual_init_error": metrics_engine.init_error,
        },
        "views": rows,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("[done] outputs")
    print(f" - hq: {hq_dir}")
    print(f" - lq: {lq_dir}")
    print(f" - restored: {restored_dir}")
    print(f" - triptych: {triptych_dir}")
    print(f" - manifest: {manifest_path}")

    if gate_enabled and fail_on_violation and not all_passed:
        raise RuntimeError(
            "Quality gate failed for one or more selected views. "
            "Inspect manifest.json and run auto_tune_restore_hparams.py."
        )


if __name__ == "__main__":
    main()
