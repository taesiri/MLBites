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
        # TODO: compute residual = predictions - targets
        # TODO: compute absolute residual
        # TODO: compute quadratic loss term: 0.5 * residual^2
        # TODO: compute linear loss term: delta * (|residual| - 0.5 * delta)
        # TODO: use torch.where to select quadratic when |residual| <= delta, else linear
        # TODO: return mean of element-wise losses
        raise NotImplementedError
