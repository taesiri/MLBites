from __future__ import annotations

import torch
import torch.nn as nn


class LeakyReLU(nn.Module):
    """Leaky ReLU activation function that allows a small gradient for negative inputs.

    Args:
        negative_slope: Coefficient for negative inputs.
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Leaky ReLU activation element-wise.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape with Leaky ReLU applied.
        """
        # For positive values, return x unchanged
        # For non-positive values, multiply by negative_slope
        return torch.where(x > 0, x, self.negative_slope * x)



