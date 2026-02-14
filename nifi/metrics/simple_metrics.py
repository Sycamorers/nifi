from typing import Tuple

import torch
import torch.nn.functional as F



def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Returns PSNR in dB for tensors in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor((max_val ** 2) / mse)).item())



def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d



def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """Computes SSIM over RGB tensors in [0, 1] with shape [1, C, H, W]."""
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError("SSIM expects tensors with shape [1, C, H, W]")

    _, c, h, w = pred.shape
    win = min(window_size, h, w)
    if win % 2 == 0:
        win = max(1, win - 1)
    if win < 3:
        # Degenerate tiny image fallback.
        return max(0.0, min(1.0, 1.0 - F.mse_loss(pred, target).item()))

    kernel_2d = _gaussian_kernel(win, sigma=sigma, device=pred.device, dtype=pred.dtype)
    kernel = kernel_2d.expand(c, 1, win, win).contiguous()

    padding = win // 2
    mu_x = F.conv2d(pred, kernel, padding=padding, groups=c)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_map = num / (den + 1e-12)
    out = ssim_map.mean().item()
    return float(max(0.0, min(1.0, out)))
