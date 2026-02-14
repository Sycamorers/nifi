from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = None
        self.model = None

        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="vgg")
            self.backend = "lpips"
        except Exception:
            import piq  # type: ignore

            self.model = piq.LPIPS(reduction="none")
            self.backend = "piq"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.backend == "lpips":
            out = self.model(x, y)
            return out.mean()

        # piq LPIPS expects [0, 1]
        x01 = (x + 1.0) * 0.5
        y01 = (y + 1.0) * 0.5
        out = self.model(x01, y01)
        return out.mean()


class DISTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = None
        self.model = None

        try:
            from DISTS_pytorch import DISTS  # type: ignore

            self.model = DISTS()
            self.backend = "dists_pytorch"
        except Exception:
            import piq  # type: ignore

            self.model = piq.DISTS(reduction="none")
            self.backend = "piq"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x01 = (x + 1.0) * 0.5
        y01 = (y + 1.0) * 0.5

        if self.backend == "dists_pytorch":
            out = self.model(x01, y01, batch_average=False)
        else:
            out = self.model(x01, y01)

        return out.mean()


class ReconstructionLossBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LPIPSLoss()
        self.dists = DISTSLoss()

    def forward(self, clean: torch.Tensor, restored: torch.Tensor) -> Dict[str, torch.Tensor]:
        l2 = F.mse_loss(restored, clean)
        lp = self.lpips(restored, clean)
        ds = self.dists(restored, clean)
        return {
            "l2": l2,
            "lpips": lp,
            "dists": ds,
        }
