from __future__ import annotations

import math

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
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize step counter and moment buffers
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using AdamW (PyTorch-style ordering)."""
        self.t += 1

        # Compute bias correction factors
        bias_correction1 = 1.0 - self.beta1**self.t
        bias_correction2 = 1.0 - self.beta2**self.t
        # PyTorch-style step size with fused bias correction
        step_size = self.lr / bias_correction1
        bc2_sqrt = math.sqrt(bias_correction2)

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                # Even if gradient is None, apply weight decay
                if self.weight_decay != 0:
                    p.mul_(1.0 - self.lr * self.weight_decay)
                continue

            m = self.m[i]
            v = self.v[i]

            # Update first moment (momentum)
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            # Update second moment (squared gradient)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            # Compute denominator with bias correction
            # Match torch.optim.AdamW ordering: denom = (sqrt(v) / sqrt(bias_correction2)).add_(eps)
            denom = v.sqrt().div_(bc2_sqrt).add_(self.eps)

            # Apply Adam update: p -= step_size * m / denom
            p.addcdiv_(m, denom, value=-step_size)

            # Apply decoupled weight decay: p -= lr * weight_decay * p
            if self.weight_decay != 0:
                p.mul_(1.0 - self.lr * self.weight_decay)
