#!/usr/bin/env python3
"""Garden debug+adjust loop when 10k-step quality gate is insufficient."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug+adjust loop for Garden training")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--metrics_csv", type=str, default="outputs/garden/metrics.csv")
    p.add_argument("--train_summary", type=str, default="runs/garden_adapters/train_summary.json")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/garden")
    p.add_argument("--target_steps", type=int, default=30000)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--min_lpips_gain_trigger", type=float, default=0.01)
    return p.parse_args()


def _as_abs(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def _read_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _float(v: object, d: float = float("nan")) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return d


def _needs_adjust(rows: List[Dict[str, str]], min_lpips_gain_trigger: float) -> bool:
    after_10k = [r for r in rows if int(float(r.get("step", 0))) >= 10000]
    if not after_10k:
        return True
    if any(str(r.get("gate_pass", "")).lower() == "true" for r in after_10k):
        return False
    best_lpips = max((_float(r.get("fixed_mean_lpips_gain", "nan"), d=float("-inf")) for r in after_10k), default=float("-inf"))
    best_sharp = max((_float(r.get("fixed_mean_sharpness_gain", "nan"), d=float("-inf")) for r in after_10k), default=float("-inf"))
    return bool(best_lpips < float(min_lpips_gain_trigger) or best_sharp < 0.0)


def _run(cmd: List[str]) -> None:
    print(json.dumps({"run": cmd}, indent=2))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise RuntimeError("garden_debug_adjust.py requires --device cuda")

    metrics_csv = _as_abs(args.metrics_csv)
    rows = _read_metrics(metrics_csv)
    needs = _needs_adjust(rows, min_lpips_gain_trigger=float(args.min_lpips_gain_trigger))

    report = {
        "metrics_csv": str(metrics_csv),
        "num_rows": len(rows),
        "needs_adjust": bool(needs),
        "min_lpips_gain_trigger": float(args.min_lpips_gain_trigger),
    }
    print(json.dumps(report, indent=2))
    if not needs:
        print(json.dumps({"status": "no_action", "reason": "quality gate already improved"}, indent=2))
        return

    # 1) Re-verify restoration path is not no-op.
    _run(
        [
            sys.executable,
            "scripts/garden_prove_restore_not_noop.py",
            "--device",
            "cuda",
            "--out_dir",
            "outputs/garden/noop_check_adjust",
        ]
    )

    # 2) Hyperparameter sweep for t0/scale/rate/detail.
    _run(
        [
            sys.executable,
            "scripts/auto_tune_garden.py",
            "--device",
            "cuda",
            "--rates",
            "0.1,0.2",
            "--t0_values",
            "99,149,199,249",
            "--adapter_scales",
            "0.4,0.6,0.8,1.0",
            "--detail_boosts",
            "0.0,0.5,1.0",
            "--views",
            "0,64,184",
        ]
    )

    cfg_path = ROOT / "configs" / "garden_known_good.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected tuned config at {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    restoration = cfg.get("restoration", {}) if isinstance(cfg, dict) else {}
    selection = cfg.get("selection", {}) if isinstance(cfg, dict) else {}
    tuned_t0 = int(restoration.get("t0", 149))
    tuned_rate = float(selection.get("chosen_rate", 0.1))

    ckpt_dir = _as_abs(args.checkpoint_dir)
    resume_ckpt = ckpt_dir / "adapter_latest.pt"
    if not resume_ckpt.exists():
        fallback = ckpt_dir / "adapter_best.pt"
        if fallback.exists():
            resume_ckpt = fallback
        else:
            raise FileNotFoundError(f"No checkpoint found to resume in {ckpt_dir}")

    # 3) Resume/continue training to target steps.
    _run(
        [
            sys.executable,
            "scripts/train_garden_adapters.py",
            "--device",
            "cuda",
            "--max_steps",
            str(int(args.target_steps)),
            "--val_every",
            str(int(args.val_every)),
            "--save_every",
            str(int(args.save_every)),
            "--batch_size",
            str(int(args.batch_size)),
            "--image_size",
            str(int(args.image_size)),
            "--t0",
            str(tuned_t0),
            "--rate",
            str(tuned_rate),
            "--resume",
            str(resume_ckpt),
            "--update_model_paths",
        ]
    )

    # 4) Re-check post-adjust quality progress.
    rows_after = _read_metrics(metrics_csv)
    still_bad = _needs_adjust(rows_after, min_lpips_gain_trigger=float(args.min_lpips_gain_trigger))
    result = {
        "status": "fail" if still_bad else "ok",
        "rows_before": len(rows),
        "rows_after": len(rows_after),
        "resume_checkpoint": str(resume_ckpt),
        "tuned_t0": tuned_t0,
        "tuned_rate": tuned_rate,
    }
    print(json.dumps(result, indent=2))
    if still_bad:
        raise SystemExit(
            "Garden still under quality target after debug-adjust. "
            "Inspect normalization/scheduler/resolution pipeline and retrain."
        )


if __name__ == "__main__":
    main()
