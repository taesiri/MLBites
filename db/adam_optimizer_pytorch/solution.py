from __future__ import annotations

import math

import torch


class Adam:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        """A minimal Adam optimizer (bias-corrected) for a list of tensors.

        Args:
            params: List of parameter tensors to update in-place. Gradients are read
                from each tensor's `.grad` field (set by autograd).
            lr: Learning rate.
            betas: (beta1, beta2) coefficients for the first/second moment estimates.
            eps: Small constant added to the denominator for numerical stability.
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using Adam (PyTorch-style bias correction)."""
        self.t += 1

        bias_correction1 = 1.0 - self.beta1**self.t
        bias_correction2 = 1.0 - self.beta2**self.t
        step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            m = self.m[i]
            v = self.v[i]

            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            denom = v.sqrt().add_(self.eps)
            p.addcdiv_(m, denom, value=-step_size)
