from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization module.

    Args:
        normalized_shape: Size of the last dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input.

        Args:
            x: Input tensor of shape (..., normalized_shape).

        Returns:
            Normalized tensor of the same shape as input.
        """
        raise NotImplementedError



