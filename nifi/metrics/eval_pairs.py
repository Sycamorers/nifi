import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from nifi.metrics.simple_metrics import psnr, ssim
from nifi.utils.diagnostics import enable_faulthandler_signal
from nifi.utils.logging import get_logger
from nifi.utils.seed import set_seed



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate paired images with LPIPS/DISTS (optional) + PSNR/SSIM (always)")
    p.add_argument("--data_root", type=str, required=True, help="Root containing scene/rate/split/{clean,pred_key}")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--pred_key", type=str, default="degraded", help="Folder name to compare against clean")
    p.add_argument("--out", type=str, required=True, help="Output JSON path")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")



def discover_pairs(data_root: Path, split: str, pred_key: str) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    for clean_path in data_root.rglob("*.png"):
        if clean_path.parent.name != "clean":
            continue
        if clean_path.parent.parent.name != split:
            continue

        pred_path = clean_path.parent.parent / pred_key / clean_path.name
        if not pred_path.exists():
            continue

        parts = clean_path.parts
        scene = "unknown"
        rate = "unknown"
        # .../<scene>/<rate>/<split>/clean/<name>.png
        if len(parts) >= 5:
            scene = clean_path.parents[3].name
            rate = clean_path.parents[2].name

        pairs.append(
            {
                "scene": scene,
                "rate": rate,
                "name": clean_path.stem,
                "clean": clean_path,
                "pred": pred_path,
            }
        )

    pairs.sort(key=lambda x: (x["scene"], x["rate"], x["name"]))
    return pairs



def load_img(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = torch.from_numpy(np.array(img)).float() / 255.0
    # HWC -> BCHW
    return arr.permute(2, 0, 1).unsqueeze(0).to(device)



class OptionalPerceptual:
    def __init__(self, device: torch.device, logger):
        self.device = device
        self.logger = logger
        self.lpips_model = None
        self.dists_model = None
        self.dists_backend = None

        try:
            import lpips  # type: ignore

            self.lpips_model = lpips.LPIPS(net="vgg").to(device)
            self.lpips_model.eval()
            logger.info("LPIPS enabled")
        except Exception as e:
            logger.warning("LPIPS unavailable: %s", e)

        try:
            from DISTS_pytorch import DISTS  # type: ignore

            self.dists_model = DISTS().to(device)
            self.dists_model.eval()
            self.dists_backend = "dists_pytorch"
            logger.info("DISTS enabled via DISTS_pytorch")
        except Exception:
            try:
                import piq  # type: ignore

                self.dists_model = piq.DISTS(reduction="none").to(device)
                self.dists_model.eval()
                self.dists_backend = "piq"
                logger.info("DISTS enabled via piq")
            except Exception as e:
                logger.warning("DISTS unavailable: %s", e)

    @property
    def has_lpips(self) -> bool:
        return self.lpips_model is not None

    @property
    def has_dists(self) -> bool:
        return self.dists_model is not None

    @torch.no_grad()
    def lpips(self, pred: torch.Tensor, clean: torch.Tensor) -> Optional[float]:
        if self.lpips_model is None:
            return None
        x = pred * 2.0 - 1.0
        y = clean * 2.0 - 1.0
        return float(self.lpips_model(x, y).mean().item())

    @torch.no_grad()
    def dists(self, pred: torch.Tensor, clean: torch.Tensor) -> Optional[float]:
        if self.dists_model is None:
            return None

        if self.dists_backend == "dists_pytorch":
            out = self.dists_model(pred, clean, batch_average=False)
        else:
            out = self.dists_model(pred, clean)
        return float(out.mean().item())



def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["psnr", "ssim", "lpips", "dists"]
    out: Dict[str, Any] = {"num_images": len(records)}

    for k in keys:
        vals = [float(r[k]) for r in records if r.get(k) is not None and math.isfinite(float(r[k]))]
        out[k] = float(sum(vals) / len(vals)) if vals else None

    return out



def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w") as f:
            f.write("\n")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    enable_faulthandler_signal()
    args = parse_args()
    logger = get_logger("nifi.eval_pairs")

    set_seed(int(args.seed), deterministic=False)
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    device = resolve_device(args.device)
    logger.info("Evaluating pairs from %s on %s", data_root, device)

    pairs = discover_pairs(data_root, split=args.split, pred_key=args.pred_key)
    if args.max_images is not None:
        pairs = pairs[: int(args.max_images)]

    if not pairs:
        raise RuntimeError(
            f"No pairs found in {data_root} for split={args.split} pred_key={args.pred_key}. "
            "Expected .../<scene>/<rate>/<split>/clean and .../<pred_key>."
        )

    optional = OptionalPerceptual(device=device, logger=logger)

    records: List[Dict[str, Any]] = []
    for item in tqdm(pairs, desc="eval_pairs"):
        clean = load_img(item["clean"], device=device)
        pred = load_img(item["pred"], device=device)

        if clean.shape != pred.shape:
            # Resize pred to clean for robust evaluation.
            pred = torch.nn.functional.interpolate(pred, size=clean.shape[-2:], mode="bilinear", align_corners=False)

        rec: Dict[str, Any] = {
            "scene": item["scene"],
            "rate": item["rate"],
            "name": item["name"],
            "clean_path": str(item["clean"]),
            "pred_path": str(item["pred"]),
            "psnr": psnr(pred, clean),
            "ssim": ssim(pred, clean),
            "lpips": optional.lpips(pred, clean),
            "dists": optional.dists(pred, clean),
        }
        records.append(rec)

    agg = aggregate(records)

    payload = {
        "data_root": str(data_root),
        "split": args.split,
        "pred_key": args.pred_key,
        "device": str(device),
        "available_metrics": {
            "lpips": optional.has_lpips,
            "dists": optional.has_dists,
            "psnr": True,
            "ssim": True,
        },
        "aggregate": agg,
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_path.with_suffix(".csv")
    write_csv(csv_path, records)

    logger.info("Wrote metrics JSON: %s", out_path)
    logger.info("Wrote metrics CSV : %s", csv_path)
    logger.info("Aggregate metrics : %s", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
