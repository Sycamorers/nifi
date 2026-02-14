from typing import Optional

import torch
import torch.nn as nn


class LowRankSpatialAdapter(nn.Module):
    """
    Lightweight LoRA-style adapter that predicts residual noise from latent, timestep,
    and text embeddings using low-rank channels.
    """

    def __init__(self, in_channels: int = 4, text_dim: int = 768, rank: int = 64, max_t: int = 1000):
        super().__init__()
        self.rank = rank
        self.down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.mid = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(rank, rank, kernel_size=3, padding=1, bias=True),
            nn.SiLU(),
        )
        self.up = nn.Conv2d(rank, in_channels, kernel_size=1, bias=False)

        self.t_embed = nn.Embedding(max_t, rank)
        self.text_proj = nn.Linear(text_dim, rank)

        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.down(latents)

        t = timesteps.clamp(min=0, max=self.t_embed.num_embeddings - 1)
        t_bias = self.t_embed(t).unsqueeze(-1).unsqueeze(-1)

        if encoder_hidden_states is None:
            text_bias = 0.0
        else:
            text = encoder_hidden_states.mean(dim=1)
            text_bias = self.text_proj(text).unsqueeze(-1).unsqueeze(-1)

        h = h + t_bias + text_bias
        h = self.mid(h)
        out = self.up(h)
        return out
