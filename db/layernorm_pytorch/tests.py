from __future__ import annotations

from types import ModuleType

import torch
import torch.nn as nn


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "LayerNorm"):
        raise AssertionError("Candidate must define class `LayerNorm`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- Test 1: Basic normalization matches torch.nn.LayerNorm ---
    normalized_shape = 8
    candidate_ln = candidate.LayerNorm(normalized_shape).to(dtype=dtype)
    ref_ln = nn.LayerNorm(normalized_shape).to(dtype=dtype)

    with torch.no_grad():
        ref_ln.weight.copy_(candidate_ln.weight)
        ref_ln.bias.copy_(candidate_ln.bias)

    x1 = torch.randn(4, 6, normalized_shape, dtype=dtype)
    y1 = candidate_ln(x1)
    y1_ref = ref_ln(x1)

    _assert_allclose(
        y1, y1_ref, atol=1e-12, rtol=1e-12, msg="Output mismatch vs torch.nn.LayerNorm."
    )

    # --- Test 2: 2D input ---
    x2 = torch.randn(5, normalized_shape, dtype=dtype)
    y2 = candidate_ln(x2)
    y2_ref = ref_ln(x2)

    _assert_allclose(
        y2, y2_ref, atol=1e-12, rtol=1e-12, msg="2D input output mismatch."
    )

    # --- Test 3: Verify mean ≈ 0 and std ≈ 1 after normalization (with default params) ---
    candidate_ln2 = candidate.LayerNorm(16).to(dtype=dtype)
    x3 = torch.randn(10, 16, dtype=dtype)
    y3 = candidate_ln2(x3)

    mean_check = y3.mean(dim=-1)
    std_check = y3.std(dim=-1, unbiased=False)

    _assert_allclose(
        mean_check,
        torch.zeros_like(mean_check),
        atol=1e-6,
        rtol=1e-6,
        msg="Normalized output should have mean ≈ 0.",
    )
    _assert_allclose(
        std_check,
        torch.ones_like(std_check),
        atol=1e-4,
        rtol=1e-4,
        msg="Normalized output should have std ≈ 1.",
    )

    # --- Test 4: Custom weight and bias ---
    candidate_ln3 = candidate.LayerNorm(4).to(dtype=dtype)
    with torch.no_grad():
        candidate_ln3.weight.copy_(torch.tensor([2.0, 1.0, 0.5, 3.0], dtype=dtype))
        candidate_ln3.bias.copy_(torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=dtype))

    ref_ln3 = nn.LayerNorm(4).to(dtype=dtype)
    with torch.no_grad():
        ref_ln3.weight.copy_(candidate_ln3.weight)
        ref_ln3.bias.copy_(candidate_ln3.bias)

    x4 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=dtype)
    y4 = candidate_ln3(x4)
    y4_ref = ref_ln3(x4)

    _assert_allclose(
        y4, y4_ref, atol=1e-12, rtol=1e-12, msg="Custom weight/bias output mismatch."
    )

    # --- Test 5: Gradients flow correctly ---
    candidate_ln4 = candidate.LayerNorm(8).to(dtype=dtype)
    x5 = torch.randn(3, 5, 8, dtype=dtype, requires_grad=True)
    y5 = candidate_ln4(x5).sum()
    y5.backward()

    if x5.grad is None:
        raise AssertionError("Expected non-None gradient for input.")
    if not torch.isfinite(x5.grad).all():
        raise AssertionError("Found non-finite values in input gradient.")

    # --- Test 6: Weight and bias are trainable parameters ---
    candidate_ln5 = candidate.LayerNorm(4)
    param_names = {name for name, _ in candidate_ln5.named_parameters()}
    if "weight" not in param_names:
        raise AssertionError("LayerNorm must have a trainable 'weight' parameter.")
    if "bias" not in param_names:
        raise AssertionError("LayerNorm must have a trainable 'bias' parameter.")

    # --- Test 7: Default initialization (weight=1, bias=0) ---
    candidate_ln6 = candidate.LayerNorm(5).to(dtype=dtype)
    _assert_allclose(
        candidate_ln6.weight,
        torch.ones(5, dtype=dtype),
        atol=1e-12,
        rtol=1e-12,
        msg="Weight should be initialized to ones.",
    )
    _assert_allclose(
        candidate_ln6.bias,
        torch.zeros(5, dtype=dtype),
        atol=1e-12,
        rtol=1e-12,
        msg="Bias should be initialized to zeros.",
    )

    # --- Test 8: Single element edge case ---
    candidate_ln7 = candidate.LayerNorm(1).to(dtype=dtype)
    x6 = torch.tensor([[5.0], [10.0]], dtype=dtype)
    y6 = candidate_ln7(x6)

    if not torch.isfinite(y6).all():
        raise AssertionError("Output should be finite even for single-element normalization.")

