from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization module.

    Normalizes the input over the last dimension, then applies
    a learnable scale (weight) and shift (bias).

    Args:
        normalized_shape: Size of the last dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters: weight (scale) and bias (shift)
        # Initialized to identity transform: weight=1, bias=0
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input.

        Args:
            x: Input tensor of shape (..., normalized_shape).

        Returns:
            Normalized tensor of the same shape as input.
        """
        # Compute mean over the last dimension
        mean = x.mean(dim=-1, keepdim=True)

        # Compute variance over the last dimension (population variance)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize: (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift
        return self.weight * x_norm + self.bias

