#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterator

import torch
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.data import build_paired_dataloader
from nifi.diffusion import DiffusionConfig, FrozenLDMWithNiFiAdapters
from nifi.losses import ReconstructionLossBundle, gt_guidance_loss, kl_score_surrogate_loss
from nifi.metrics import PerceptualMetrics
from nifi.utils.checkpoint import load_checkpoint, save_checkpoint
from nifi.utils.config import load_config
from nifi.utils.logging import CSVLogger, dump_json, get_logger
from nifi.utils.sanity import assert_no_nan, assert_shape, tiny_overfit_guard
from nifi.utils.seed import set_seed



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NiFi repro-lite")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--exp", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke_test", action="store_true", help="1 scene-ish tiny run (10 imgs / 100 steps)")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    return p.parse_args()



def infinite_loader(dl) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in dl:
            yield batch



def pick_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32



def restore_latents_one_step(
    model: FrozenLDMWithNiFiAdapters,
    z_degraded: torch.Tensor,
    prompts,
    t0: int,
    stochastic: bool,
) -> torch.Tensor:
    b = z_degraded.shape[0]
    device = z_degraded.device
    t0_t = torch.full((b,), t0, device=device, dtype=torch.long)

    noise = torch.randn_like(z_degraded) if stochastic else torch.zeros_like(z_degraded)
    z_tilde_t0, _ = model.q_sample(z_degraded, t0_t, noise=noise)

    eps_minus = model.predict_eps(z_tilde_t0, t0_t, prompts, adapter_type="minus", train_mode=stochastic)
    sigma0 = model.sigma(t0_t).view(-1, 1, 1, 1)
    z_hat = z_tilde_t0 - sigma0 * eps_minus
    return z_hat



def evaluate(
    model: FrozenLDMWithNiFiAdapters,
    val_loader,
    device: torch.device,
    t0: int,
    max_batches: int = None,
) -> Dict[str, float]:
    metrics = PerceptualMetrics(device=device)

    lpips_before = []
    lpips_after = []
    dists_before = []
    dists_after = []

    model.eval()
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        prompts = list(batch["prompt"])

        with torch.no_grad():
            z_deg = model.encode_images(degraded)
            z_hat = restore_latents_one_step(model, z_deg, prompts, t0=t0, stochastic=False)
            restored = model.decode_latents(z_hat)

            lp_b = metrics.lpips(degraded.float(), clean.float())
            lp_a = metrics.lpips(restored.float(), clean.float())
            ds_b = metrics.dists(degraded.float(), clean.float())
            ds_a = metrics.dists(restored.float(), clean.float())

        lpips_before.extend(lp_b.detach().cpu().tolist())
        lpips_after.extend(lp_a.detach().cpu().tolist())
        dists_before.extend(ds_b.detach().cpu().tolist())
        dists_after.extend(ds_a.detach().cpu().tolist())

    model.train()

    return {
        "lpips_before": float(sum(lpips_before) / max(1, len(lpips_before))),
        "lpips_after": float(sum(lpips_after) / max(1, len(lpips_after))),
        "dists_before": float(sum(dists_before) / max(1, len(dists_before))),
        "dists_after": float(sum(dists_after) / max(1, len(dists_after))),
    }



def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    exp_dir = Path(args.exp)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("nifi.train")
    csv_logger = CSVLogger(exp_dir / "train_log.csv")

    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg["train"]
    loss_w = cfg["loss_weights"]

    if args.smoke_test:
        smoke = cfg.get("smoke_test", {})
        train_cfg["max_steps"] = int(smoke.get("train_steps", 100))
        train_cfg["max_train_samples"] = int(smoke.get("num_images", 10))
        train_cfg["max_eval_samples"] = int(smoke.get("num_images", 10))
        logger.info("Smoke test mode enabled")

    if args.max_steps is not None:
        train_cfg["max_steps"] = int(args.max_steps)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)

    image_size = int(args.image_size or cfg.get("image_size", 256) or 256)
    mixed_precision = train_cfg.get("mixed_precision", "fp16")
    amp_dtype = pick_dtype(mixed_precision)
    use_amp = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    rates = cfg.get("rates", None)

    train_loader = build_paired_dataloader(
        data_root=args.data_root,
        split="train",
        image_size=image_size,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        shuffle=True,
        max_samples=train_cfg.get("max_train_samples"),
        allowed_rates=rates,
    )

    val_loader = build_paired_dataloader(
        data_root=args.data_root,
        split="test",
        image_size=image_size,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        shuffle=False,
        max_samples=train_cfg.get("max_eval_samples"),
        allowed_rates=rates,
    )

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
    model.freeze_backbone()

    rec_losses = ReconstructionLossBundle().to(device)
    rec_losses.eval()
    for p in rec_losses.parameters():
        p.requires_grad_(False)

    opt_minus = torch.optim.AdamW(
        model.adapter_parameters("minus"),
        lr=float(train_cfg["lr_phi_minus"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    opt_plus = torch.optim.AdamW(
        model.adapter_parameters("plus"),
        lr=float(train_cfg["lr_phi_plus"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    start_step = 1
    best_score = math.inf

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        model.phi_minus.load_state_dict(ckpt["phi_minus"])
        model.phi_plus.load_state_dict(ckpt["phi_plus"])
        if ckpt.get("opt_minus"):
            opt_minus.load_state_dict(ckpt["opt_minus"])
        if ckpt.get("opt_plus"):
            opt_plus.load_state_dict(ckpt["opt_plus"])
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("best_metric") is not None:
            best_score = float(ckpt["best_metric"])
        start_step = int(ckpt.get("step", 0)) + 1
        logger.info("Resumed from %s at step=%d", args.resume, start_step)

    t0 = int(cfg["diffusion"]["num_train_timesteps"] - 1) if cfg["diffusion"].get("t_ablation_full", False) else int(cfg["diffusion"]["t0"])
    n_steps = int(train_cfg["max_steps"])
    log_every = int(train_cfg.get("log_every", 10))
    eval_every = int(train_cfg.get("eval_every", 100))
    save_every = int(train_cfg.get("save_every", 100))

    hist = {"start_loss": None, "end_loss": None}

    train_iter = infinite_loader(train_loader)
    opt_minus.zero_grad(set_to_none=True)
    opt_plus.zero_grad(set_to_none=True)

    pbar = tqdm(range(start_step, n_steps + 1), desc="train")
    for step in pbar:
        batch = next(train_iter)
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        prompts = list(batch["prompt"])

        assert_shape("clean", clean, 4)
        assert_shape("degraded", degraded, 4)

        bsz = clean.shape[0]
        t0_t = torch.full((bsz,), t0, device=device, dtype=torch.long)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z_clean = model.encode_images(clean)
            z_deg = model.encode_images(degraded)

            z_tilde_t0, _ = model.q_sample(z_deg, t0_t)
            eps_minus = model.predict_eps(z_tilde_t0, t0_t, prompts, adapter_type="minus", train_mode=True)

            sigma0 = model.sigma(t0_t).view(-1, 1, 1, 1)
            z_hat = z_tilde_t0 - sigma0 * eps_minus

            t_dist = torch.randint(0, model_cfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            z_hat_t, _ = model.q_sample(z_hat, t_dist)

            eps_real = model.predict_eps(z_hat_t, t_dist, prompts, adapter_type=None, train_mode=False)
            eps_restore = model.predict_eps(z_hat_t, t_dist, prompts, adapter_type="plus", train_mode=False)

            sigma_dist = model.sigma(t_dist).view(-1, 1, 1, 1)
            s_real = model.score_from_eps(eps_real, sigma_dist)
            s_restore = model.score_from_eps(eps_restore, sigma_dist)

            loss_kl = kl_score_surrogate_loss(z_hat, s_real, s_restore)
            loss_gt = gt_guidance_loss(eps_minus, z_tilde_t0, z_clean, sigma0)

            restored = model.decode_latents(z_hat)

        rec = rec_losses(clean.float(), restored.float())

        total_minus = (
            float(loss_w["alpha"]) * float(loss_w["kl"]) * loss_kl
            + (1.0 - float(loss_w["alpha"])) * float(loss_w["gt"]) * loss_gt
            + float(loss_w["l2"]) * rec["l2"]
            + float(loss_w["lpips"]) * rec["lpips"]
            + float(loss_w["dists"]) * rec["dists"]
        )

        assert_no_nan("total_minus", total_minus)
        scaled_minus = total_minus / grad_accum

        if scaler.is_enabled():
            scaler.scale(scaled_minus).backward()
        else:
            scaled_minus.backward()

        step_ready = (step % grad_accum) == 0
        if step_ready:
            if scaler.is_enabled():
                scaler.unscale_(opt_minus)
            torch.nn.utils.clip_grad_norm_(model.phi_minus.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(opt_minus)
            else:
                opt_minus.step()
            opt_minus.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z_hat_det = z_hat.detach()
            t_plus = torch.randint(0, model_cfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            z_plus_t, noise_plus = model.q_sample(z_hat_det, t_plus)
            eps_plus = model.predict_eps(z_plus_t, t_plus, prompts, adapter_type="plus", train_mode=True)
            loss_plus = F.mse_loss(eps_plus, noise_plus)

        assert_no_nan("loss_plus", loss_plus)
        scaled_plus = loss_plus / grad_accum

        if scaler.is_enabled():
            scaler.scale(scaled_plus).backward()
        else:
            scaled_plus.backward()

        if step_ready:
            if scaler.is_enabled():
                scaler.unscale_(opt_plus)
            torch.nn.utils.clip_grad_norm_(model.phi_plus.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(opt_plus)
                scaler.update()
            else:
                opt_plus.step()
            opt_plus.zero_grad(set_to_none=True)

        if hist["start_loss"] is None:
            hist["start_loss"] = float(total_minus.item())
        hist["end_loss"] = float(total_minus.item())

        row = {
            "step": step,
            "loss_minus": float(total_minus.item()),
            "loss_plus": float(loss_plus.item()),
            "loss_kl": float(loss_kl.item()),
            "loss_gt": float(loss_gt.item()),
            "loss_l2": float(rec["l2"].item()),
            "loss_lpips": float(rec["lpips"].item()),
            "loss_dists": float(rec["dists"].item()),
        }
        csv_logger.log(row)

        if step % log_every == 0:
            pbar.set_postfix({"loss_m": f"{row['loss_minus']:.4f}", "loss_p": f"{row['loss_plus']:.4f}"})
            logger.info("step=%d loss_minus=%.5f loss_plus=%.5f", step, row["loss_minus"], row["loss_plus"])

        if step % eval_every == 0 or step == n_steps:
            t_start = time.time()
            val = evaluate(model, val_loader, device=device, t0=t0, max_batches=10 if args.smoke_test else None)
            score = float(val["lpips_after"] + val["dists_after"])
            val_row = {"step": step, **val, "score": score}
            csv_logger.log(val_row)
            logger.info(
                "eval step=%d lpips %.4f->%.4f dists %.4f->%.4f (%.2fs)",
                step,
                val["lpips_before"],
                val["lpips_after"],
                val["dists_before"],
                val["dists_after"],
                time.time() - t_start,
            )

            if score < best_score:
                best_score = score
                save_checkpoint(
                    exp_dir / "best.pt",
                    step=step,
                    phi_minus_state=model.phi_minus.state_dict(),
                    phi_plus_state=model.phi_plus.state_dict(),
                    opt_minus_state=opt_minus.state_dict(),
                    opt_plus_state=opt_plus.state_dict(),
                    scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                    best_metric=best_score,
                    extra={"config": cfg},
                )

        if step % save_every == 0 or step == n_steps:
            save_checkpoint(
                exp_dir / "latest.pt",
                step=step,
                phi_minus_state=model.phi_minus.state_dict(),
                phi_plus_state=model.phi_plus.state_dict(),
                opt_minus_state=opt_minus.state_dict(),
                opt_plus_state=opt_plus.state_dict(),
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                best_metric=best_score,
                extra={"config": cfg},
            )

    if not tiny_overfit_guard(hist):
        logger.warning("Overfit sanity check did not improve loss (start %.5f -> end %.5f)", hist["start_loss"], hist["end_loss"])

    summary = {
        "best_score": best_score,
        "t0": t0,
        "max_steps": n_steps,
        "smoke_test": bool(args.smoke_test),
    }
    dump_json(exp_dir / "train_summary.json", summary)
    logger.info("Training complete. best_score=%.6f", best_score)


if __name__ == "__main__":
    main()
