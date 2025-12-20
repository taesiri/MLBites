from __future__ import annotations

from types import ModuleType

import torch


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module."""
    if not hasattr(candidate, "SGDMomentum"):
        raise AssertionError("Candidate must define class `SGDMomentum`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- Test 1: Single step matches torch.optim.SGD ---
    w0 = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    lr = 0.1
    momentum = 0.9

    opt_candidate = candidate.SGDMomentum([w0], lr=lr, momentum=momentum)
    opt_torch = torch.optim.SGD([w1], lr=lr, momentum=momentum)

    # Compute gradients
    target = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    loss0 = ((w0 - target) ** 2).sum()
    loss1 = ((w1 - target) ** 2).sum()
    loss0.backward()
    loss1.backward()

    opt_candidate.step()
    opt_torch.step()

    _assert_allclose(
        w0,
        w1,
        atol=1e-12,
        rtol=1e-12,
        msg="Single step SGDMomentum diverged from torch.optim.SGD.",
    )

    # --- Test 2: Multiple steps with momentum accumulation ---
    w0 = torch.tensor([0.5, -0.5, 1.0], dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    lr = 0.05
    momentum = 0.95

    opt_candidate = candidate.SGDMomentum([w0], lr=lr, momentum=momentum)
    opt_torch = torch.optim.SGD([w1], lr=lr, momentum=momentum)

    target = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)

    for t in range(1, 11):
        # Zero grads manually (not using zero_grad method)
        if w0.grad is not None:
            w0.grad.zero_()
        if w1.grad is not None:
            w1.grad.zero_()

        loss0 = ((w0 - target) ** 2).sum()
        loss1 = ((w1 - target) ** 2).sum()
        loss0.backward()
        loss1.backward()

        opt_candidate.step()
        opt_torch.step()

        _assert_allclose(
            w0,
            w1,
            atol=1e-12,
            rtol=1e-12,
            msg=f"SGDMomentum diverged from torch.optim.SGD at step {t}.",
        )

    # --- Test 3: Multiple parameters ---
    a0 = torch.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    b0 = torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=dtype, requires_grad=True)
    a1 = a0.detach().clone().requires_grad_(True)
    b1 = b0.detach().clone().requires_grad_(True)

    lr = 0.01
    momentum = 0.8

    opt_candidate = candidate.SGDMomentum([a0, b0], lr=lr, momentum=momentum)
    opt_torch = torch.optim.SGD([a1, b1], lr=lr, momentum=momentum)

    for _ in range(5):
        if a0.grad is not None:
            a0.grad.zero_()
        if b0.grad is not None:
            b0.grad.zero_()
        if a1.grad is not None:
            a1.grad.zero_()
        if b1.grad is not None:
            b1.grad.zero_()

        loss0 = a0.sum() + (b0**2).sum()
        loss1 = a1.sum() + (b1**2).sum()
        loss0.backward()
        loss1.backward()

        opt_candidate.step()
        opt_torch.step()

    _assert_allclose(
        a0, a1, atol=1e-12, rtol=1e-12, msg="Multi-param test: param 'a' mismatch."
    )
    _assert_allclose(
        b0, b1, atol=1e-12, rtol=1e-12, msg="Multi-param test: param 'b' mismatch."
    )

    # --- Test 4: Grad=None parameters are skipped ---
    a0 = torch.tensor([1.0], dtype=dtype, requires_grad=True)
    a1 = a0.detach().clone().requires_grad_(True)
    b0 = torch.tensor([5.0], dtype=dtype, requires_grad=False)  # grad stays None
    b1 = b0.detach().clone()

    opt_candidate = candidate.SGDMomentum([a0, b0], lr=0.1, momentum=0.9)
    opt_torch = torch.optim.SGD([a1], lr=0.1, momentum=0.9)

    loss0 = (a0 - 2.0).pow(2).sum()
    loss1 = (a1 - 2.0).pow(2).sum()
    loss0.backward()
    loss1.backward()

    opt_candidate.step()
    opt_torch.step()

    _assert_allclose(
        a0, a1, atol=1e-12, rtol=1e-12, msg="Param with grad should be updated."
    )
    _assert_allclose(
        b0, b1, atol=1e-12, rtol=1e-12, msg="Param with grad=None should be unchanged."
    )

    # --- Test 5: Zero momentum degenerates to vanilla SGD ---
    w0 = torch.tensor([2.0, -3.0], dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    lr = 0.2
    momentum = 0.0

    opt_candidate = candidate.SGDMomentum([w0], lr=lr, momentum=momentum)
    opt_torch = torch.optim.SGD([w1], lr=lr, momentum=momentum)

    for _ in range(3):
        if w0.grad is not None:
            w0.grad.zero_()
        if w1.grad is not None:
            w1.grad.zero_()

        loss0 = (w0**2).sum()
        loss1 = (w1**2).sum()
        loss0.backward()
        loss1.backward()

        opt_candidate.step()
        opt_torch.step()

    _assert_allclose(
        w0,
        w1,
        atol=1e-12,
        rtol=1e-12,
        msg="Zero momentum should match vanilla SGD.",
    )




