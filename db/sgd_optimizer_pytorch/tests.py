from __future__ import annotations

from types import ModuleType

import torch


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module."""
    if not hasattr(candidate, "SGD"):
        raise AssertionError("Candidate must define class `SGD`.")

    torch.manual_seed(0)
    dtype = torch.float64

    # --- test 1: plain SGD matches torch.optim.SGD ---
    w0 = torch.tensor([1.0, -2.0, 0.5], dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    lr = 0.1
    opt_candidate = candidate.SGD([w0], lr=lr, momentum=0.0, weight_decay=0.0)
    opt_torch = torch.optim.SGD([w1], lr=lr, momentum=0.0, weight_decay=0.0)

    target = torch.tensor([3.0, -1.0, 2.0], dtype=dtype)

    for t in range(1, 6):
        loss0 = ((w0 - target) ** 2).sum()
        loss1 = ((w1 - target) ** 2).sum()
        loss0.backward()
        loss1.backward()

        opt_candidate.step()
        opt_torch.step()

        w0.grad.zero_()
        w1.grad.zero_()

        _assert_allclose(
            w0,
            w1,
            atol=0.0,
            rtol=0.0,
            msg=f"Plain SGD diverged from torch.optim.SGD at step {t}.",
        )

    # --- test 2: momentum + weight_decay matches torch.optim.SGD ---
    w0 = torch.tensor([0.25, -0.75, 1.5], dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    lr = 0.05
    momentum = 0.9
    weight_decay = 0.1

    opt_candidate = candidate.SGD(
        [w0], lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    opt_torch = torch.optim.SGD([w1], lr=lr, momentum=momentum, weight_decay=weight_decay)

    for t in range(1, 11):
        loss0 = ((w0 - target) ** 2).sum()
        loss1 = ((w1 - target) ** 2).sum()
        loss0.backward()
        loss1.backward()

        opt_candidate.step()
        opt_torch.step()

        w0.grad.zero_()
        w1.grad.zero_()

        _assert_allclose(
            w0,
            w1,
            atol=0.0,
            rtol=0.0,
            msg=f"Momentum/weight_decay SGD diverged from torch.optim.SGD at step {t}.",
        )

    # --- test 3: grad=None params are skipped ---
    a0 = torch.tensor([1.0], dtype=dtype, requires_grad=True)
    a1 = a0.detach().clone().requires_grad_(True)
    b0 = torch.tensor([5.0], dtype=dtype, requires_grad=False)  # grad stays None
    b1 = b0.detach().clone().requires_grad_(False)

    opt_candidate = candidate.SGD([a0, b0], lr=0.01, momentum=0.9, weight_decay=0.1)
    opt_torch = torch.optim.SGD([a1], lr=0.01, momentum=0.9, weight_decay=0.1)

    loss_a0 = (a0 - 2.0).pow(2).sum()
    loss_a1 = (a1 - 2.0).pow(2).sum()
    loss_a0.backward()
    loss_a1.backward()

    opt_candidate.step()
    opt_torch.step()

    _assert_allclose(a0, a1, atol=0.0, rtol=0.0, msg="SGD update mismatch for grad param.")
    _assert_allclose(b0, b1, atol=0.0, rtol=0.0, msg="Param with grad=None should be unchanged.")


