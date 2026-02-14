#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import autocast
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.data import build_paired_dataloader
from nifi.diffusion import DiffusionConfig, FrozenLDMWithNiFiAdapters
from nifi.metrics import PerceptualMetrics, aggregate_scene_metrics
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config
from nifi.utils.logging import get_logger
from nifi.utils.seed import set_seed



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NiFi model (LPIPS + DISTS)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--out", type=str, required=True, help="Output metrics json path")
    p.add_argument("--config", type=str, default=None, help="Optional config override")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_restored", action="store_true")
    p.add_argument("--restored_dir", type=str, default=None)
    return p.parse_args()



def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w") as f:
            f.write("\n")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def restore_one_step(
    model: FrozenLDMWithNiFiAdapters,
    z_degraded: torch.Tensor,
    prompts,
    t0: int,
):
    b = z_degraded.shape[0]
    t0_t = torch.full((b,), t0, device=z_degraded.device, dtype=torch.long)
    z_tilde_t0, _ = model.q_sample(z_degraded, t0_t, noise=torch.zeros_like(z_degraded))
    eps_minus = model.predict_eps(z_tilde_t0, t0_t, prompts, adapter_type="minus", train_mode=False)
    sigma0 = model.sigma(t0_t).view(-1, 1, 1, 1)
    z_hat = z_tilde_t0 - sigma0 * eps_minus
    return z_hat



def main() -> None:
    args = parse_args()
    logger = get_logger("nifi.eval")

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")

    cfg = None
    if args.config:
        cfg = load_config(args.config)
    elif ckpt.get("extra", {}).get("config"):
        cfg = ckpt["extra"]["config"]
    else:
        raise ValueError("Could not find config in checkpoint. Pass --config explicitly.")

    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp = cfg["train"].get("mixed_precision", "fp16")
    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    use_amp = device.type == "cuda" and mp in {"bf16", "fp16"}

    model_cfg = DiffusionConfig(
        model_name_or_path=cfg["model"]["pretrained_model_name_or_path"],
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        lora_rank=int(cfg["model"]["lora_rank"]),
        guidance_scale=float(cfg["model"]["guidance_scale"]),
        prompt_dropout=float(cfg["model"]["prompt_dropout"]),
        max_token_length=int(cfg["model"]["max_token_length"]),
        vae_scaling_factor=float(cfg["model"]["vae_scaling_factor"]),
    )

    model = FrozenLDMWithNiFiAdapters(model_cfg, device=device, dtype=amp_dtype if use_amp else torch.float32)
    model.phi_minus.load_state_dict(ckpt["phi_minus"])
    model.phi_plus.load_state_dict(ckpt["phi_plus"])
    model.freeze_backbone()
    model.eval()

    image_size = int(cfg.get("image_size", 256) or 256)
    dl = build_paired_dataloader(
        data_root=args.data_root,
        split=args.split,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        max_samples=args.max_samples,
        allowed_rates=cfg.get("rates", None),
    )

    metrics = PerceptualMetrics(device=device)
    t0 = int(cfg["diffusion"]["num_train_timesteps"] - 1) if cfg["diffusion"].get("t_ablation_full", False) else int(cfg["diffusion"]["t0"])

    records: List[Dict[str, object]] = []

    restored_dir = Path(args.restored_dir) if args.restored_dir else None
    if args.save_restored:
        if restored_dir is None:
            restored_dir = Path(args.out).parent / "restored"
        restored_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(dl, desc="eval"):
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        prompts = list(batch["prompt"])
        scenes = list(batch["scene"])
        rates = list(batch["rate"])
        names = list(batch["name"])

        with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z_deg = model.encode_images(degraded)
            z_hat = restore_one_step(model, z_deg, prompts, t0=t0)
            restored = model.decode_latents(z_hat)

            lp_before = metrics.lpips(degraded.float(), clean.float())
            lp_after = metrics.lpips(restored.float(), clean.float())
            ds_before = metrics.dists(degraded.float(), clean.float())
            ds_after = metrics.dists(restored.float(), clean.float())

        if args.save_restored:
            imgs = ((restored.clamp(-1, 1) + 1.0) * 0.5).detach().cpu()
            for i, img in enumerate(imgs):
                scene_dir = restored_dir / scenes[i] / rates[i]
                scene_dir.mkdir(parents=True, exist_ok=True)
                out_path = scene_dir / f"{names[i]}.png"
                from torchvision.utils import save_image

                save_image(img, out_path)

        for i in range(clean.shape[0]):
            records.append(
                {
                    "scene": scenes[i],
                    "rate": rates[i],
                    "name": names[i],
                    "lpips_before": float(lp_before[i].item()),
                    "lpips_after": float(lp_after[i].item()),
                    "dists_before": float(ds_before[i].item()),
                    "dists_after": float(ds_after[i].item()),
                }
            )

    summary = aggregate_scene_metrics(records)
    payload = {
        "checkpoint": str(args.ckpt),
        "split": args.split,
        "num_samples": len(records),
        "metrics": summary,
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_path.with_suffix(".csv")
    save_csv(csv_path, records)

    agg = summary["aggregate"]
    logger.info(
        "LPIPS %.4f -> %.4f | DISTS %.4f -> %.4f",
        agg["lpips_before"],
        agg["lpips_after"],
        agg["dists_before"],
        agg["dists_after"],
    )
    logger.info("Wrote metrics to %s and %s", out_path, csv_path)


if __name__ == "__main__":
    main()
