from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np


class PerceptualMetrics(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        try:
            import lpips  # type: ignore

            self.lpips_model = lpips.LPIPS(net="vgg").to(device)
            self.lpips_backend = "lpips"
        except Exception:
            import piq  # type: ignore

            self.lpips_model = piq.LPIPS(reduction="none").to(device)
            self.lpips_backend = "piq"

        try:
            from DISTS_pytorch import DISTS  # type: ignore

            self.dists_model = DISTS().to(device)
            self.dists_backend = "dists_pytorch"
        except Exception:
            import piq  # type: ignore

            self.dists_model = piq.DISTS(reduction="none").to(device)
            self.dists_backend = "piq"

        self.eval()

    @torch.no_grad()
    def lpips(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        if self.lpips_backend == "lpips":
            out = self.lpips_model(pred, target)
            return out.flatten()

        pred01 = (pred + 1.0) * 0.5
        target01 = (target + 1.0) * 0.5
        out = self.lpips_model(pred01, target01)
        return out.flatten()

    @torch.no_grad()
    def dists(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        pred01 = (pred + 1.0) * 0.5
        target01 = (target + 1.0) * 0.5

        if self.dists_backend == "dists_pytorch":
            out = self.dists_model(pred01, target01, batch_average=False)
        else:
            out = self.dists_model(pred01, target01)

        return out.flatten()



def aggregate_scene_metrics(records: List[Dict[str, object]]) -> Dict[str, object]:
    grouped = defaultdict(list)
    for r in records:
        grouped[r["scene"]].append(r)

    per_scene = {}
    for scene, items in grouped.items():
        per_scene[scene] = {
            "lpips_before": float(np.mean([x["lpips_before"] for x in items])),
            "lpips_after": float(np.mean([x["lpips_after"] for x in items])),
            "dists_before": float(np.mean([x["dists_before"] for x in items])),
            "dists_after": float(np.mean([x["dists_after"] for x in items])),
            "num_images": len(items),
        }

    aggregate = {
        "lpips_before": float(np.mean([x["lpips_before"] for x in records])) if records else float("nan"),
        "lpips_after": float(np.mean([x["lpips_after"] for x in records])) if records else float("nan"),
        "dists_before": float(np.mean([x["dists_before"] for x in records])) if records else float("nan"),
        "dists_after": float(np.mean([x["dists_after"] for x in records])) if records else float("nan"),
        "num_images": len(records),
    }
    return {
        "per_scene": per_scene,
        "aggregate": aggregate,
    }
