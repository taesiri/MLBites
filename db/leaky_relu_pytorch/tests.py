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
    if not hasattr(candidate, "LeakyReLU"):
        raise AssertionError("Candidate must define class `LeakyReLU`.")

    LeakyReLU = candidate.LeakyReLU
    dtype = torch.float64

    # --- test 1: positive inputs (unchanged) ---
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    leaky_relu = LeakyReLU(negative_slope=0.01)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 1 failed: positive inputs")

    # --- test 2: negative inputs ---
    x = torch.tensor([-1.0, -2.0, -3.0], dtype=dtype)
    expected = torch.tensor([-0.01, -0.02, -0.03], dtype=dtype)
    leaky_relu = LeakyReLU(negative_slope=0.01)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 2 failed: negative inputs")

    # --- test 3: mixed inputs with custom slope ---
    x = torch.tensor([-2.0, 0.0, 2.0], dtype=dtype)
    expected = torch.tensor([-0.2, 0.0, 2.0], dtype=dtype)
    leaky_relu = LeakyReLU(negative_slope=0.1)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 3 failed: mixed inputs")

    # --- test 4: zero input ---
    x = torch.tensor([0.0], dtype=dtype)
    expected = torch.tensor([0.0], dtype=dtype)
    leaky_relu = LeakyReLU(negative_slope=0.01)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 4 failed: zero input")

    # --- test 5: custom negative slope ---
    x = torch.tensor([-1.0, -2.0], dtype=dtype)
    expected = torch.tensor([-0.2, -0.4], dtype=dtype)
    leaky_relu = LeakyReLU(negative_slope=0.2)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 5 failed: custom negative slope")

    # --- test 6: match PyTorch's LeakyReLU ---
    torch.manual_seed(42)
    x = torch.randn(10, 5, dtype=dtype)
    negative_slope = 0.01

    torch_leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)
    expected = torch_leaky_relu(x)
    leaky_relu = LeakyReLU(negative_slope=negative_slope)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 6 failed: mismatch with PyTorch LeakyReLU")

    # --- test 7: different negative slope with PyTorch comparison ---
    negative_slope = 0.2
    torch_leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)
    expected = torch_leaky_relu(x)
    leaky_relu = LeakyReLU(negative_slope=negative_slope)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 7 failed: mismatch with negative_slope=0.2")

    # --- test 8: 2D tensor ---
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=dtype)
    expected = torch.tensor([[-0.01, 2.0], [3.0, -0.04]], dtype=dtype)
    leaky_relu = LeakyReLU(negative_slope=0.01)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 8 failed: 2D tensor")

    # --- test 9: verify it's an nn.Module ---
    leaky_relu = LeakyReLU(negative_slope=0.01)
    if not isinstance(leaky_relu, torch.nn.Module):
        raise AssertionError("Test 9 failed: LeakyReLU must inherit from nn.Module")

    # --- test 10: verify forward method exists ---
    if not hasattr(leaky_relu, "forward"):
        raise AssertionError("Test 10 failed: LeakyReLU must have a forward method")

    # --- test 11: large negative slope (slope=1.0 is just identity) ---
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=dtype)
    expected = x.clone()  # slope=1.0 means identity for all values
    leaky_relu = LeakyReLU(negative_slope=1.0)
    result = leaky_relu(x)
    _assert_allclose(result, expected, atol=1e-10, rtol=1e-10, msg="Test 11 failed: slope=1.0")

    # --- test 12: gradient flow check ---
    x = torch.tensor([-2.0, 0.0, 2.0], dtype=dtype, requires_grad=True)
    leaky_relu = LeakyReLU(negative_slope=0.1)
    output = leaky_relu(x)
    loss = output.sum()
    loss.backward()
    # Expected gradients: [0.1, 0.1, 1.0] (negative_slope for non-positive, 1 for positive)
    expected_grad = torch.tensor([0.1, 0.1, 1.0], dtype=dtype)
    _assert_allclose(x.grad, expected_grad, atol=1e-10, rtol=1e-10, msg="Test 12 failed: gradient check")



