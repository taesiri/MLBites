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
        """A minimal SGD optimizer with optional momentum and weight decay.

        Args:
            params: List of parameter tensors to update in-place. Gradients are read
                from each tensor's `.grad` field (set by autograd).
            lr: Learning rate.
            momentum: Momentum factor. If 0.0, do plain SGD.
            weight_decay: L2 penalty coefficient (adds `weight_decay * p` to the grad).

        Notes:
            - Interview-friendly: you do NOT need to subclass torch.optim.Optimizer.
            - Assume grads are dense tensors and inputs satisfy the contract.
        """
        # TODO: store params + hyperparameters
        # TODO: if momentum > 0, create a per-parameter momentum buffer initialized to zeros_like(p)
        raise NotImplementedError

    @torch.no_grad()
    def step(self) -> None:
        """Update all parameters in-place using SGD.

        Match torch.optim.SGD behavior for the supported features:
            d_p = grad
            if weight_decay != 0: d_p = d_p + weight_decay * p
            if momentum != 0:
                buf = momentum * buf + d_p
                d_p = buf
            p = p - lr * d_p

        Returns:
            None. Updates parameters in-place.
        """
        # TODO: for each param with non-None grad, compute d_p and update p in-place
        raise NotImplementedError
