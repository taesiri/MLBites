from __future__ import annotations

import torch


class AdamW:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """A minimal AdamW optimizer (decoupled weight decay) for a list of tensors."""
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using AdamW (matches torch.optim.AdamW ordering)."""
        self.t += 1

        bias_correction1 = 1.0 - self.beta1**self.t
        bias_correction2 = 1.0 - self.beta2**self.t
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)

            m = self.m[i]
            v = self.v[i]

            # Match torch.optim.adam._single_tensor_adam ordering:
            # exp_avg.lerp_(grad, 1 - beta1)
            m.lerp_(g, 1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            denom = v.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            p.addcdiv_(m, denom, value=-step_size)



