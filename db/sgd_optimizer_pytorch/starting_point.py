from __future__ import annotations

import torch


class SGD:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> None:
        """A minimal SGD optimizer with optional weight decay.

        Args:
            params: List of parameter tensors to optimize.
            lr: Learning rate.
            weight_decay: L2 penalty coefficient.
        """
        raise NotImplementedError

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using SGD.

        Must match torch.optim.SGD behavior for the supported features.
        """
        raise NotImplementedError
