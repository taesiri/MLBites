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
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum

        # Initialize velocity buffer for each parameter (zeros with same shape)
        self.velocity = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using SGD with momentum.

        The update rule for each parameter p with gradient g:
            v = momentum * v + g
            p = p - lr * v

        This matches PyTorch's torch.optim.SGD behavior.
        """
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            v = self.velocity[i]

            # Update velocity: v = momentum * v + g
            v.mul_(self.momentum).add_(g)

            # Update parameter: p = p - lr * v
            p.add_(v, alpha=-self.lr)




