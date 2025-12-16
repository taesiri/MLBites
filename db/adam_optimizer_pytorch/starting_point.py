from __future__ import annotations

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

        Notes:
            - This is intentionally minimal and interview-friendly.
            - You do NOT need to subclass torch.optim.Optimizer.
            - Assume grads are dense tensors (not sparse) and the inputs satisfy the contract.
        """
        # TODO: store params + hyperparameters
        # TODO: create per-parameter state buffers m and v (initialized to zeros_like each param)
        # TODO: initialize step counter t (int), where the first update uses t=1
        raise NotImplementedError

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using Adam.

        For each parameter p with gradient g = p.grad:
            m = beta1*m + (1-beta1)*g
            v = beta2*v + (1-beta2)*(g*g)

        Use the same ordering as torch.optim.Adam:
            bias_correction1 = 1 - beta1**t
            bias_correction2 = 1 - beta2**t
            step_size = lr / bias_correction1
            denom = sqrt(v) / sqrt(bias_correction2) + eps
            p -= step_size * m / denom

        Returns:
            None. Updates parameters in-place.
        """
        # TODO: increment step counter
        # TODO: for each param with non-None grad, update m, v, and p in-place
        raise NotImplementedError
