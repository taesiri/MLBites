from __future__ import annotations

import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    """KL Divergence Loss module.

    Computes the Kullback-Leibler divergence between target distribution P
    and input distribution Q (provided as log-probabilities).

    KL(P || Q) = sum(P * (log(P) - log(Q)))

    Note: Following PyTorch convention, `input` should be log-probabilities
    and `target` should be probabilities.

    Args:
        reduction: Specifies the reduction to apply to the output:
            - "none": no reduction
            - "mean": mean of all elements
            - "batchmean": sum divided by batch size (recommended for KL divergence)
            - "sum": sum of all elements
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the KL Divergence loss.

        Args:
            input: Tensor of log-probabilities (log Q).
            target: Tensor of probabilities (P), same shape as input.

        Returns:
            KL divergence loss (scalar if reduced, same shape as input otherwise).
        """
        # Compute element-wise KL: target * (log(target) - input)
        # Use torch.xlogy to handle 0 * log(0) = 0 correctly
        kl_div = torch.xlogy(target, target) - target * input

        # Apply reduction
        if self.reduction == "none":
            return kl_div
        elif self.reduction == "sum":
            return kl_div.sum()
        elif self.reduction == "mean":
            return kl_div.mean()
        elif self.reduction == "batchmean":
            return kl_div.sum() / input.size(0)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


