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
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "AdamW"):
        raise AssertionError("Candidate must define class `AdamW`.")

    torch.manual_seed(0)
    dtype = torch.float64

    # --- test 1: match torch.optim.AdamW step-by-step on a simple quadratic ---
    w0 = torch.tensor([1.0, -2.0, 0.5], dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    lr = 1e-2
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2

    opt_candidate = candidate.AdamW([w0], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    opt_torch = torch.optim.AdamW([w1], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    target = torch.tensor([3.0, -1.0, 2.0], dtype=dtype)

    for t in range(1, 11):
        loss0 = ((w0 - target) ** 2).sum()
        loss1 = ((w1 - target) ** 2).sum()

        loss0.backward()
        loss1.backward()

        opt_candidate.step()
        opt_torch.step()

        # Keep deterministic, avoid set_to_none differences.
        w0.grad.zero_()
        w1.grad.zero_()

        _assert_allclose(
            w0,
            w1,
            atol=1e-12,
            rtol=1e-12,
            msg=f"Parameters diverged from torch.optim.AdamW at step {t}.",
        )

    # --- test 2: params with grad=None still get weight decay ---
    a0 = torch.tensor([1.0], dtype=dtype, requires_grad=True)
    a1 = a0.detach().clone().requires_grad_(True)
    b0 = torch.tensor([5.0], dtype=dtype, requires_grad=False)  # grad will stay None
    b1 = b0.detach().clone().requires_grad_(False)

    opt_candidate2 = candidate.AdamW([a0, b0], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    opt_torch2 = torch.optim.AdamW([a1, b1], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    loss_a0 = (a0 - 2.0).pow(2).sum()
    loss_a1 = (a1 - 2.0).pow(2).sum()
    loss_a0.backward()
    loss_a1.backward()

    opt_candidate2.step()
    opt_torch2.step()

    _assert_allclose(a0, a1, atol=1e-12, rtol=1e-12, msg="Grad param update mismatch.")
    _assert_allclose(
        b0, b1, atol=1e-12, rtol=1e-12, msg="Param with grad=None should still get weight decay."
    )

    # --- test 3: zero weight decay should match Adam behavior (approximately) ---
    w2 = torch.tensor([1.0, -2.0], dtype=dtype, requires_grad=True)
    w3 = w2.detach().clone().requires_grad_(True)

    opt_adamw = candidate.AdamW([w2], lr=lr, betas=betas, eps=eps, weight_decay=0.0)
    opt_adam = torch.optim.Adam([w3], lr=lr, betas=betas, eps=eps)

    for t in range(1, 6):
        loss2 = ((w2 - target[:2]) ** 2).sum()
        loss3 = ((w3 - target[:2]) ** 2).sum()

        loss2.backward()
        loss3.backward()

        opt_adamw.step()
        opt_adam.step()

        w2.grad.zero_()
        w3.grad.zero_()

        _assert_allclose(
            w2,
            w3,
            atol=1e-12,
            rtol=1e-12,
            msg=f"AdamW with weight_decay=0 should match Adam at step {t}.",
        )
