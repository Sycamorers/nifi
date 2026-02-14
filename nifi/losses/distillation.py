import torch
import torch.nn.functional as F


def kl_score_surrogate_loss(
    restored_latents: torch.Tensor,
    score_real: torch.Tensor,
    score_restore: torch.Tensor,
) -> torch.Tensor:
    """
    Distribution matching surrogate:
      grad wrt restored_latents is (s_restore - s_real).
    """
    score_diff = (score_restore - score_real).detach()
    return (score_diff * restored_latents).mean()


def gt_guidance_loss(pred_eps: torch.Tensor, noisy_latents: torch.Tensor, clean_latents: torch.Tensor, sigma_t: torch.Tensor):
    eps_gt = (noisy_latents - (1.0 - sigma_t) * clean_latents) / (sigma_t + 1e-6)
    return F.mse_loss(pred_eps, eps_gt)
