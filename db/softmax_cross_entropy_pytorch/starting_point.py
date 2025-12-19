from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute numerically stable softmax along the specified dimension.

    Args:
        x: Input tensor of any shape containing raw logits.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of same shape as x with softmax probabilities (sums to 1 along dim).
    """
    raise NotImplementedError


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute numerically stable log-softmax along the specified dimension.

    Args:
        x: Input tensor of any shape containing raw logits.
        dim: Dimension along which to compute log-softmax.

    Returns:
        Tensor of same shape as x with log-probabilities.
    """
    raise NotImplementedError


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for classification.

    Args:
        logits: Raw logits of shape (N, C) where N is batch size, C is num classes.
        targets: Integer class indices of shape (N,) with values in [0, C-1].

    Returns:
        Scalar tensor containing mean cross-entropy loss.
    """
    raise NotImplementedError


