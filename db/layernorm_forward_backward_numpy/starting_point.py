from __future__ import annotations

import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """LayerNorm forward pass.

    Args:
        x: (N, D)
        gamma: (D,) or (1, D)
        beta: (D,) or (1, D)
        eps: numerical stability constant

    Returns:
        y: (N, D)
        cache: values needed for backward pass
    """
    # y = (x - mu) / sqrt(var + eps) * gamma + beta
    raise NotImplementedError


def layernorm_backward(dy, cache):
    """LayerNorm backward pass.

    Args:
        dy: (N, D) upstream gradient
        cache: from layernorm_forward

    Returns:
        dx: (N, D)
        dgamma: (D,) or (1, D)
        dbeta: (D,) or (1, D)
    """
    raise NotImplementedError
