from typing import Optional, Tuple

import torch


class SigmaSchedule:
    """
    Simple sigma schedule for one-step restoration formulation:
      x_t = (1 - sigma_t) * x + sigma_t * eps
    """

    def __init__(self, num_train_timesteps: int = 1000, sigma_min: float = 1e-4, sigma_max: float = 1.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.registered_sigmas = torch.linspace(sigma_min, sigma_max, self.num_train_timesteps)

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> "SigmaSchedule":
        self.registered_sigmas = self.registered_sigmas.to(device=device, dtype=dtype)
        return self

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.clamp(min=0, max=self.num_train_timesteps - 1)
        return self.registered_sigmas[timesteps]

    def q_sample(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        sigma = self.sigma(timesteps).view(-1, 1, 1, 1)
        x_t = (1.0 - sigma) * x + sigma * noise
        return x_t, noise

    @staticmethod
    def score_from_eps(eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Using score approximation under additive noise convention
        return -eps / (sigma + 1e-6)
