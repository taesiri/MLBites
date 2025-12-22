from __future__ import annotations

import torch
import torch.nn as nn


class CachedMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # TODO: Initialize linear projections and attention parameters
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Compute Q, K, V for new tokens
        # TODO: Concatenate with cached K, V if available
        # TODO: Compute attention with causal masking
        # TODO: Return output and updated cache
        raise NotImplementedError




