from __future__ import annotations

import torch


class SGD:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> None:
        """A minimal SGD optimizer with optional weight decay."""
        # Store parameters as a list (allows multiple iteration)
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using SGD (matches torch.optim.SGD subset)."""
        for p in self.params:
            # Read gradient from the parameter's .grad attribute (set by autograd)
            g = p.grad
            if g is None:
                # Skip parameters that don't have gradients (e.g., frozen layers)
                continue

            # Start with the raw gradient
            d_p = g

            # Apply weight decay: d_p = g + weight_decay * p
            # This is L2 regularization added to the gradient
            if self.weight_decay != 0.0:
                d_p = d_p.add(p, alpha=self.weight_decay)

            # Update parameter: p = p - lr * d_p
            p.add_(d_p, alpha=-self.lr)
