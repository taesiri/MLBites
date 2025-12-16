from __future__ import annotations

import torch


class SGD:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        """A minimal SGD optimizer with optional momentum and weight decay."""
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.buf = [torch.zeros_like(p) for p in self.params] if momentum != 0.0 else []

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using SGD (matches torch.optim.SGD subset)."""
        use_momentum = self.momentum != 0.0

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            d_p = g
            if self.weight_decay != 0.0:
                d_p = d_p.add(p, alpha=self.weight_decay)

            if use_momentum:
                b = self.buf[i]
                b.mul_(self.momentum).add_(d_p)
                d_p = b

            p.add_(d_p, alpha=-self.lr)
