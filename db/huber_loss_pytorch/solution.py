from __future__ import annotations

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber Loss module that combines MSE (for small errors) and MAE (for large errors).

    Huber loss formula:
      - If |residual| <= delta: loss = 0.5 * residual^2
      - If |residual| > delta:  loss = delta * (|residual| - 0.5 * delta)

    Args:
        delta: Threshold for switching between quadratic and linear loss.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Huber Loss between predictions and targets.

        Args:
            predictions: Tensor of predicted values.
            targets: Tensor of target values (same shape as predictions).

        Returns:
            Scalar tensor containing the mean Huber loss.
        """
        residual = predictions - targets
        abs_residual = torch.abs(residual)

        quadratic = 0.5 * residual**2
        linear = self.delta * (abs_residual - 0.5 * self.delta)

        loss = torch.where(abs_residual <= self.delta, quadratic, linear)
        return loss.mean()
