from __future__ import annotations

import torch


class SGDMomentum:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.9,
    ) -> None:
        """A minimal SGD optimizer with momentum.

        Args:
            params: List of parameter tensors to update in-place. Gradients are
                read from each tensor's `.grad` attribute.
            lr: Learning rate (step size).
            momentum: Momentum coefficient (typically 0.9).
        """
        # TODO: store parameters, hyperparameters, and initialize velocity buffers
        raise NotImplementedError

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using SGD with momentum.

        The update rule for each parameter p with gradient g:
            v = momentum * v + g
            p = p - lr * v

        This matches PyTorch's torch.optim.SGD behavior.
        """
        # TODO: implement momentum update
        # 1. For each parameter, get its gradient
        # 2. Update the velocity buffer: v = momentum * v + g
        # 3. Update the parameter: p = p - lr * v
        raise NotImplementedError


