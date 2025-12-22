from __future__ import annotations

import torch


class AdamW:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """A minimal AdamW optimizer (bias-corrected, decoupled weight decay) for a list of tensors.

        Args:
            params: List of parameter tensors to update in-place. Gradients are read
                from each tensor's `.grad` field (set by autograd).
            lr: Learning rate.
            betas: (beta1, beta2) coefficients for the first/second moment estimates.
            eps: Small constant added to the denominator for numerical stability.
            weight_decay: Weight decay coefficient (applied directly to parameters, not gradients).

        Returns:
            None.
        """
        raise NotImplementedError

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using AdamW.

        Args:
            None.

        Returns:
            None. Updates parameters in-place.
        """
        raise NotImplementedError
