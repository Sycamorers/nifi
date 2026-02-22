#!/usr/bin/env python3
"""Sweep demo hyperparameters and emit a quality-gated known-good config."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import autocast
import numpy as np
from PIL import Image, ImageFilter
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_restoration import ArtifactRestorationDiffusionConfig, FrozenBackboneArtifactRestorationModel
from nifi.artifact_synthesis import ArtifactSynthesisCompressionConfig, ProxyArtifactSynthesisCompressor
from nifi.metrics.perceptual_metrics import PerceptualMetrics
from nifi.metrics.simple_metrics import ssim
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config
from nifi.utils.runtime import configure_runtime, get_runtime_defaults, resolve_device
from nifi.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-select demo hyperparameters with quality gate")
    parser.add_argument("--dataset", type=str, default="mipnerf360")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--views", nargs="*", default=["auto"])  # explicit indices or auto
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--rates", type=str, default="0.1,0.5")
    parser.add_argument("--t0_values", type=str, default="99,149,199")
    parser.add_argument("--adapter_scales", type=str, default="0.0,0.5,1.0")

    parser.add_argument("--smooth_passes", type=str, default="0,1,2")
    parser.add_argument("--unsharp_radii", type=str, default="1.2,1.6,2.0")
    parser.add_argument("--unsharp_percents", type=str, default="150,200,250")
    parser.add_argument("--unsharp_threshold", type=int, default=2)

    parser.add_argument("--checkpoint", type=str, default="runs/nifi_tiny/best.pt")
    parser.add_argument("--model_config", type=str, default="configs/default.yaml")

    parser.add_argument("--min_lpips_margin", type=float, default=0.01)
    parser.add_argument("--min_dists_margin", type=float, default=0.005)
    parser.add_argument("--min_ssim_gain", type=float, default=0.0)

    parser.add_argument("--out_summary", type=str, default=None)
    parser.add_argument("--out_config", type=str, default="configs/demo_realdata_known_good.yaml")
    return parser.parse_args()


def _parse_numeric_list(raw: str, cast_fn):
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            vals.append(cast_fn(part))
    if not vals:
        raise ValueError(f"empty numeric list: {raw}")
    return vals


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


def _is_artifact_scene(path: Path) -> bool:
    return path.is_dir() and (path / "clean").exists()


def _to_rate_dir(rate: float) -> str:
    return f"rate_{float(rate):.3f}"


def _resolve_dataset_scene(dataset: str, scene: str) -> Path:
    candidate = Path(scene)
    if candidate.exists() and candidate.is_dir() and not _is_artifact_scene(candidate):
        return candidate.resolve()

    checks = [
        ROOT / "data" / dataset / scene,
        ROOT / "data" / "benchmarks" / dataset / scene,
        ROOT / "data" / "mipnerf360" / scene,
    ]
    if dataset in {"deepblending", "tanks_temples"}:
        checks.extend(
            [
                ROOT / "data" / "tanks_temples_deepblending" / scene,
                ROOT / "data" / "benchmarks" / "tanks_temples_deepblending" / scene,
            ]
        )

    for p in checks:
        if p.exists() and p.is_dir():
            return p.resolve()

    raise FileNotFoundError(f"Could not resolve dataset scene for dataset={dataset}, scene={scene}")


def _resolve_artifact_scene(
    dataset: str,
    scene: str,
    out_root: Path,
    rates: List[float],
    compression_cfg: ArtifactSynthesisCompressionConfig,
) -> Path:
    by_scene = ROOT / "artifacts" / scene
    if _is_artifact_scene(by_scene):
        return by_scene.resolve()

    scene_path = Path(scene)
    if _is_artifact_scene(scene_path):
        return scene_path.resolve()

    dataset_scene = _resolve_dataset_scene(dataset, scene)
    artifact_scene = out_root / "_artifacts" / dataset_scene.name

    have_all = _is_artifact_scene(artifact_scene) and all((artifact_scene / _to_rate_dir(r)).exists() for r in rates)
    if have_all:
        return artifact_scene.resolve()

    artifact_scene.mkdir(parents=True, exist_ok=True)
    compressor = ProxyArtifactSynthesisCompressor(compression_cfg)
    compressor.synthesize_scene_artifacts(
        scene_dir=dataset_scene,
        rates_lambda=rates,
        out_dir=artifact_scene,
        holdout_every=8,
        max_images=None,
    )
    return artifact_scene.resolve()


def _ensure_rate_dir(
    artifact_scene: Path,
    rate: float,
    compression_cfg: ArtifactSynthesisCompressionConfig,
) -> str:
    rate_dir = _to_rate_dir(rate)
    full = artifact_scene / rate_dir
    if full.exists():
        return rate_dir

    compressor = ProxyArtifactSynthesisCompressor(compression_cfg)
    full.mkdir(parents=True, exist_ok=True)
    for src in sorted((artifact_scene / "clean").glob("*.png")):
        with Image.open(src) as img:
            out = compressor.synthesize_view_artifact(img.convert("RGB"), rate_lambda=rate)
            out.save(full / src.name)
    return rate_dir


def _load_test_split(scene: Path) -> List[str]:
    meta = scene / "metadata.json"
    if not meta.exists():
        return []
    try:
        payload = json.loads(meta.read_text())
        splits = payload.get("splits", {}) if isinstance(payload, dict) else {}
        test = splits.get("test", []) if isinstance(splits, dict) else []
        if isinstance(test, list):
            return [str(x) for x in test]
    except Exception:
        pass
    return []


def _select_evenly(items: Sequence[str], n: int, seed: int) -> List[str]:
    vals = list(items)
    if n <= 0:
        raise ValueError("n must be > 0")
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
    return out


def _to_minus1_1(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device) * 2.0 - 1.0


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


def _lap_var(x: torch.Tensor) -> float:
    gray = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=1, keepdim=True).float()
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    y = F.conv2d(gray, kernel, padding=1)
    return float(y.var().item())


def _apply_classical(lq: torch.Tensor, smooth_passes: int, radius: float, percent: int, threshold: int) -> torch.Tensor:
    img01 = ((lq.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")

    for _ in range(max(0, int(smooth_passes))):
        pil = pil.filter(ImageFilter.SMOOTH_MORE)

    pil = pil.filter(ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent), threshold=int(threshold)))

    out = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).to(device=lq.device) * 2.0 - 1.0


@dataclass
class Restorer:
    model: FrozenBackboneArtifactRestorationModel
    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype

    def run(self, lq: torch.Tensor, t0: int, adapter_scale: float, prompt: str = "") -> torch.Tensor:
        lq_pad, pads = _pad_to_multiple(lq, m=8)
        b = lq_pad.shape[0]
        t = torch.full((b,), int(t0), device=self.device, dtype=torch.long)
        with torch.no_grad(), autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            z = self.model.encode_images(lq_pad)
            zt, _ = self.model.q_sample(z, t, noise=torch.zeros_like(z))
            eps_base = self.model.predict_eps(zt, t, [prompt] * b, adapter_type=None, train_mode=False)
            eps_minus = self.model.predict_eps(zt, t, [prompt] * b, adapter_type="minus", train_mode=False)
            eps = eps_base + float(adapter_scale) * (eps_minus - eps_base)
            sigma = self.model.sigma(t).view(-1, 1, 1, 1)
            zhat = zt - sigma * eps
            out = self.model.decode_latents(zhat)
        return _unpad(out, pads)


def _pick_device(flag: str) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_restorer(checkpoint: Path, model_config: Path, device_flag: str) -> Optional[Restorer]:
    if not checkpoint.exists() or not model_config.exists():
        return None

    cfg = load_config(str(model_config))
    runtime = get_runtime_defaults()
    runtime.update(cfg.get("runtime", {}))
    runtime["deterministic"] = True
    runtime["cudnn_benchmark"] = False
    runtime["device"] = _pick_device(device_flag)
    configure_runtime(runtime)
    device = resolve_device(runtime)

    mp = str(cfg.get("train", {}).get("mixed_precision", "fp16"))
    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    use_amp = device.type == "cuda" and mp in {"bf16", "fp16"}

    ckpt = load_checkpoint(str(checkpoint), map_location="cpu")
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
    return Restorer(model=model, device=device, use_amp=use_amp, amp_dtype=amp_dtype)


@dataclass
class EvalMetrics:
    lpips: Optional[float]
    dists: Optional[float]
    ssim: float
    sharpness: float


class MetricEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.perceptual = None
        self.error = None
        try:
            self.perceptual = PerceptualMetrics(device=device)
        except Exception as exc:
            self.error = str(exc)

    def eval(self, pred: torch.Tensor, target: torch.Tensor) -> EvalMetrics:
        lpips_val = None
        dists_val = None
        if self.perceptual is not None:
            try:
                lpips_val = float(self.perceptual.lpips(pred, target).item())
                dists_val = float(self.perceptual.dists(pred, target).item())
            except Exception:
                lpips_val = None
                dists_val = None

        pred01 = ((pred.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
        target01 = ((target.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
        return EvalMetrics(
            lpips=lpips_val,
            dists=dists_val,
            ssim=float(ssim(pred01, target01)),
            sharpness=_lap_var(pred),
        )


def _gate(
    before: EvalMetrics,
    after: EvalMetrics,
    min_lpips_margin: float,
    min_dists_margin: float,
    min_ssim_gain: float,
) -> Tuple[bool, str, float]:
    metric_name = "ssim"
    gain = after.ssim - before.ssim
    ok = gain >= min_ssim_gain

    if before.lpips is not None and after.lpips is not None:
        metric_name = "lpips"
        gain = before.lpips - after.lpips
        ok = gain >= min_lpips_margin
    elif before.dists is not None and after.dists is not None:
        metric_name = "dists"
        gain = before.dists - after.dists
        ok = gain >= min_dists_margin

    sharp_ok = after.sharpness >= before.sharpness
    passed = bool(ok and sharp_ok)
    reason = f"metric={metric_name}, gain={gain:.6f}, sharp={before.sharpness:.6f}->{after.sharpness:.6f}"
    return passed, reason, gain


def main() -> None:
    args = parse_args()

    seed = int(args.seed)
    set_seed(seed, deterministic=True)
    random.seed(seed)
    np.random.seed(seed)

    rates = _parse_numeric_list(args.rates, float)
    t0_values = _parse_numeric_list(args.t0_values, int)
    adapter_scales = _parse_numeric_list(args.adapter_scales, float)
    smooth_vals = _parse_numeric_list(args.smooth_passes, int)
    radius_vals = _parse_numeric_list(args.unsharp_radii, float)
    percent_vals = _parse_numeric_list(args.unsharp_percents, int)

    out_root = ROOT / "outputs" / "auto_hparams" / args.dataset / Path(args.scene).name
    out_root.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary) if args.out_summary else (out_root / "sweep_summary.json")
    out_config = Path(args.out_config)

    compression_cfg = ArtifactSynthesisCompressionConfig(
        jpeg_quality_min=8,
        jpeg_quality_max=55,
        downsample_min=2,
        downsample_max=8,
    )

    artifact_scene = _resolve_artifact_scene(
        dataset=args.dataset,
        scene=args.scene,
        out_root=out_root,
        rates=rates,
        compression_cfg=compression_cfg,
    )

    # view selection from first rate
    first_rate_dir = _ensure_rate_dir(artifact_scene, rates[0], compression_cfg)
    shared = sorted({p.name for p in (artifact_scene / "clean").glob("*.png")} & {p.name for p in (artifact_scene / first_rate_dir).glob("*.png")})
    if not shared:
        raise RuntimeError("No shared views found in artifact scene")

    explicit = _parse_views(args.views)
    if explicit is not None:
        selected = []
        for v in explicit:
            name = f"{int(v):05d}.png"
            if name not in shared:
                raise ValueError(f"view {v} ({name}) missing")
            selected.append(name)
    else:
        test = [x for x in _load_test_split(artifact_scene) if x in shared]
        pool = test if len(test) >= args.num_views else shared
        selected = _select_evenly(pool, args.num_views, seed=seed)

    checkpoint = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else ROOT / args.checkpoint
    model_config = Path(args.model_config) if Path(args.model_config).is_absolute() else ROOT / args.model_config
    restorer = _build_restorer(checkpoint=checkpoint, model_config=model_config, device_flag=args.device)
    device = restorer.device if restorer is not None else torch.device(_pick_device(args.device))
    metric_engine = MetricEngine(device=device)

    print("[info] sweep setup")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "scene": args.scene,
                "artifact_scene": str(artifact_scene),
                "selected_views": selected,
                "rates": rates,
                "t0_values": t0_values,
                "adapter_scales": adapter_scales,
                "smooth_passes": smooth_vals,
                "unsharp_radii": radius_vals,
                "unsharp_percents": percent_vals,
                "checkpoint": str(checkpoint),
                "restorer_available": restorer is not None,
                "metric_engine_error": metric_engine.error,
            },
            indent=2,
        )
    )

    # cache HQ/LQ tensors per rate
    tensors: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    baselines: Dict[str, Dict[str, EvalMetrics]] = {}
    for rate in rates:
        rate_dir = _ensure_rate_dir(artifact_scene, rate, compression_cfg)
        tensors[rate_dir] = {}
        baselines[rate_dir] = {}
        for name in selected:
            hq = _to_minus1_1(artifact_scene / "clean" / name, device=device)
            lq = _to_minus1_1(artifact_scene / rate_dir / name, device=device)
            tensors[rate_dir][name] = {"hq": hq, "lq": lq}
            baselines[rate_dir][name] = metric_engine.eval(lq, hq)

    results: List[Dict[str, object]] = []

    # NiFi sweeps
    if restorer is not None:
        for rate in rates:
            rate_dir = _to_rate_dir(rate)
            for t0 in t0_values:
                for scale in adapter_scales:
                    per_view = []
                    all_pass = True
                    gains = []
                    for name in selected:
                        hq = tensors[rate_dir][name]["hq"]
                        lq = tensors[rate_dir][name]["lq"]
                        out = restorer.run(lq, t0=t0, adapter_scale=scale, prompt="")
                        before = baselines[rate_dir][name]
                        after = metric_engine.eval(out, hq)
                        passed, reason, gain = _gate(
                            before,
                            after,
                            min_lpips_margin=args.min_lpips_margin,
                            min_dists_margin=args.min_dists_margin,
                            min_ssim_gain=args.min_ssim_gain,
                        )
                        all_pass = all_pass and passed
                        gains.append(gain)
                        per_view.append(
                            {
                                "view": name,
                                "passed": passed,
                                "reason": reason,
                                "before": before.__dict__,
                                "after": after.__dict__,
                            }
                        )

                    results.append(
                        {
                            "mode": "nifi_eq7",
                            "rate": rate,
                            "rate_dir": rate_dir,
                            "t0": t0,
                            "adapter_scale": scale,
                            "passed": bool(all_pass),
                            "mean_gain": float(np.mean(gains)),
                            "per_view": per_view,
                        }
                    )

    # Classical sweeps
    for rate in rates:
        rate_dir = _to_rate_dir(rate)
        for smooth in smooth_vals:
            for radius in radius_vals:
                for percent in percent_vals:
                    per_view = []
                    all_pass = True
                    gains = []
                    for name in selected:
                        hq = tensors[rate_dir][name]["hq"]
                        lq = tensors[rate_dir][name]["lq"]
                        out = _apply_classical(
                            lq,
                            smooth_passes=smooth,
                            radius=radius,
                            percent=percent,
                            threshold=args.unsharp_threshold,
                        )
                        before = baselines[rate_dir][name]
                        after = metric_engine.eval(out, hq)
                        passed, reason, gain = _gate(
                            before,
                            after,
                            min_lpips_margin=args.min_lpips_margin,
                            min_dists_margin=args.min_dists_margin,
                            min_ssim_gain=args.min_ssim_gain,
                        )
                        all_pass = all_pass and passed
                        gains.append(gain)
                        per_view.append(
                            {
                                "view": name,
                                "passed": passed,
                                "reason": reason,
                                "before": before.__dict__,
                                "after": after.__dict__,
                            }
                        )

                    results.append(
                        {
                            "mode": "classical_unsharp",
                            "rate": rate,
                            "rate_dir": rate_dir,
                            "smooth_passes": smooth,
                            "unsharp_radius": radius,
                            "unsharp_percent": percent,
                            "unsharp_threshold": int(args.unsharp_threshold),
                            "passed": bool(all_pass),
                            "mean_gain": float(np.mean(gains)),
                            "per_view": per_view,
                        }
                    )

    passing = [r for r in results if r["passed"]]
    if passing:
        passing.sort(key=lambda x: x["mean_gain"], reverse=True)
        best = passing[0]
    else:
        results.sort(key=lambda x: x["mean_gain"], reverse=True)
        best = results[0]

    # Build config for run_demo
    chosen_mode = str(best["mode"])
    chosen_rate = float(best["rate"])

    demo_config = {
        "dataset": {
            "name": args.dataset,
            "scene": Path(args.scene).name,
        },
        "demo": {
            "rate_presets": {
                "low": chosen_rate,
                "medium": 0.5,
                "high": 1.0,
            },
            "default_rate": "low",
            "auto_num_views": max(3, int(args.num_views)),
            "prefer_test_split": True,
            "view_selection_mode": "deterministic_even",
            "expected_resolution": "native_source",
            "image_format": "png",
        },
        "restoration": {
            "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
            "model_config": str(model_config.relative_to(ROOT)) if model_config.is_relative_to(ROOT) else str(model_config),
            "prompt": "",
            "t0": int(best.get("t0", 199)),
            "adapter_scale": float(best.get("adapter_scale", 1.0)),
            "final_output": chosen_mode,
            "candidate_order": ["nifi_eq7", "classical_unsharp", "identity_lq"],
            "classical": {
                "smooth_passes": int(best.get("smooth_passes", 1)),
                "unsharp_radius": float(best.get("unsharp_radius", 1.6)),
                "unsharp_percent": int(best.get("unsharp_percent", 200)),
                "unsharp_threshold": int(best.get("unsharp_threshold", args.unsharp_threshold)),
            },
        },
        "quality_gate": {
            "enabled": True,
            "fail_on_violation": True,
            "min_lpips_margin": float(args.min_lpips_margin),
            "min_dists_margin": float(args.min_dists_margin),
            "min_ssim_gain": float(args.min_ssim_gain),
            "require_sharpness_non_decrease": True,
        },
        "compression": {
            "jpeg_quality_min": 8,
            "jpeg_quality_max": 55,
            "downsample_min": 2,
            "downsample_max": 8,
        },
        "selection": {
            "chosen_mode": chosen_mode,
            "chosen_rate": chosen_rate,
            "passed": bool(best["passed"]),
            "mean_gain": float(best["mean_gain"]),
            "views": selected,
            "num_candidates": len(results),
            "num_passing_candidates": len(passing),
        },
    }

    out_config.parent.mkdir(parents=True, exist_ok=True)
    with out_config.open("w") as f:
        yaml.safe_dump(demo_config, f, sort_keys=False)

    summary = {
        "dataset": args.dataset,
        "scene": args.scene,
        "artifact_scene": str(artifact_scene),
        "selected_views": selected,
        "checkpoint": str(checkpoint),
        "model_config": str(model_config),
        "min_lpips_margin": args.min_lpips_margin,
        "min_dists_margin": args.min_dists_margin,
        "min_ssim_gain": args.min_ssim_gain,
        "best": best,
        "num_candidates": len(results),
        "num_passing_candidates": len(passing),
        "passing_top5": passing[:5],
        "results": sorted(results, key=lambda x: x["mean_gain"], reverse=True),
        "out_config": str(out_config),
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))

    print("[done] sweep complete")
    print(f" - summary: {out_summary}")
    print(f" - config:  {out_config}")
    print("[best]")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
