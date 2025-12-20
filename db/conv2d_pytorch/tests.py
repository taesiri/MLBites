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
    if not hasattr(candidate, "Conv2d"):
        raise AssertionError("Candidate must define class `Conv2d`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- Test 1: Basic convolution matches torch.nn.Conv2d ---
    in_channels, out_channels, kernel_size = 3, 8, 3
    candidate_conv = candidate.Conv2d(in_channels, out_channels, kernel_size).to(dtype=dtype)
    ref_conv = nn.Conv2d(in_channels, out_channels, kernel_size).to(dtype=dtype)

    with torch.no_grad():
        ref_conv.weight.copy_(candidate_conv.weight)
        ref_conv.bias.copy_(candidate_conv.bias)

    x1 = torch.randn(2, in_channels, 8, 8, dtype=dtype)
    y1 = candidate_conv(x1)
    y1_ref = ref_conv(x1)

    _assert_allclose(
        y1, y1_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch vs torch.nn.Conv2d (basic)."
    )

    # --- Test 2: With padding ---
    candidate_conv2 = candidate.Conv2d(1, 4, kernel_size=3, padding=1).to(dtype=dtype)
    ref_conv2 = nn.Conv2d(1, 4, kernel_size=3, padding=1).to(dtype=dtype)

    with torch.no_grad():
        ref_conv2.weight.copy_(candidate_conv2.weight)
        ref_conv2.bias.copy_(candidate_conv2.bias)

    x2 = torch.randn(1, 1, 5, 5, dtype=dtype)
    y2 = candidate_conv2(x2)
    y2_ref = ref_conv2(x2)

    _assert_allclose(
        y2, y2_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with padding=1."
    )

    # Verify same spatial size with padding=1, kernel=3
    if y2.shape != (1, 4, 5, 5):
        raise AssertionError(f"Expected output shape (1, 4, 5, 5), got {y2.shape}")

    # --- Test 3: With stride ---
    candidate_conv3 = candidate.Conv2d(2, 6, kernel_size=3, stride=2).to(dtype=dtype)
    ref_conv3 = nn.Conv2d(2, 6, kernel_size=3, stride=2).to(dtype=dtype)

    with torch.no_grad():
        ref_conv3.weight.copy_(candidate_conv3.weight)
        ref_conv3.bias.copy_(candidate_conv3.bias)

    x3 = torch.randn(3, 2, 10, 10, dtype=dtype)
    y3 = candidate_conv3(x3)
    y3_ref = ref_conv3(x3)

    _assert_allclose(
        y3, y3_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with stride=2."
    )

    expected_size = (10 - 3) // 2 + 1  # = 4
    if y3.shape != (3, 6, expected_size, expected_size):
        raise AssertionError(f"Expected output shape (3, 6, 4, 4), got {y3.shape}")

    # --- Test 4: With both padding and stride ---
    candidate_conv4 = candidate.Conv2d(3, 16, kernel_size=5, stride=2, padding=2).to(dtype=dtype)
    ref_conv4 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2).to(dtype=dtype)

    with torch.no_grad():
        ref_conv4.weight.copy_(candidate_conv4.weight)
        ref_conv4.bias.copy_(candidate_conv4.bias)

    x4 = torch.randn(2, 3, 32, 32, dtype=dtype)
    y4 = candidate_conv4(x4)
    y4_ref = ref_conv4(x4)

    _assert_allclose(
        y4, y4_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with stride=2, padding=2."
    )

    expected_size4 = (32 + 2 * 2 - 5) // 2 + 1  # = 16
    if y4.shape != (2, 16, expected_size4, expected_size4):
        raise AssertionError(f"Expected output shape (2, 16, 16, 16), got {y4.shape}")

    # --- Test 5: Single channel, kernel=1 (1x1 convolution) ---
    candidate_conv5 = candidate.Conv2d(4, 8, kernel_size=1).to(dtype=dtype)
    ref_conv5 = nn.Conv2d(4, 8, kernel_size=1).to(dtype=dtype)

    with torch.no_grad():
        ref_conv5.weight.copy_(candidate_conv5.weight)
        ref_conv5.bias.copy_(candidate_conv5.bias)

    x5 = torch.randn(1, 4, 7, 7, dtype=dtype)
    y5 = candidate_conv5(x5)
    y5_ref = ref_conv5(x5)

    _assert_allclose(
        y5, y5_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch for 1x1 convolution."
    )

    # --- Test 6: Gradients flow correctly ---
    candidate_conv6 = candidate.Conv2d(2, 4, kernel_size=3, padding=1).to(dtype=dtype)
    x6 = torch.randn(2, 2, 6, 6, dtype=dtype, requires_grad=True)
    y6 = candidate_conv6(x6).sum()
    y6.backward()

    if x6.grad is None:
        raise AssertionError("Expected non-None gradient for input.")
    if not torch.isfinite(x6.grad).all():
        raise AssertionError("Found non-finite values in input gradient.")

    # --- Test 7: Weight and bias are trainable parameters ---
    candidate_conv7 = candidate.Conv2d(3, 8, kernel_size=3)
    param_names = {name for name, _ in candidate_conv7.named_parameters()}
    if "weight" not in param_names:
        raise AssertionError("Conv2d must have a trainable 'weight' parameter.")
    if "bias" not in param_names:
        raise AssertionError("Conv2d must have a trainable 'bias' parameter.")

    # --- Test 8: Correct weight shape ---
    candidate_conv8 = candidate.Conv2d(5, 10, kernel_size=4)
    expected_weight_shape = (10, 5, 4, 4)
    expected_bias_shape = (10,)

    if candidate_conv8.weight.shape != expected_weight_shape:
        raise AssertionError(
            f"Weight shape should be {expected_weight_shape}, got {candidate_conv8.weight.shape}"
        )
    if candidate_conv8.bias.shape != expected_bias_shape:
        raise AssertionError(
            f"Bias shape should be {expected_bias_shape}, got {candidate_conv8.bias.shape}"
        )

    # --- Test 9: Non-square input ---
    candidate_conv9 = candidate.Conv2d(1, 2, kernel_size=3, padding=1).to(dtype=dtype)
    ref_conv9 = nn.Conv2d(1, 2, kernel_size=3, padding=1).to(dtype=dtype)

    with torch.no_grad():
        ref_conv9.weight.copy_(candidate_conv9.weight)
        ref_conv9.bias.copy_(candidate_conv9.bias)

    x9 = torch.randn(1, 1, 8, 12, dtype=dtype)  # Non-square: 8x12
    y9 = candidate_conv9(x9)
    y9_ref = ref_conv9(x9)

    _assert_allclose(
        y9, y9_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch for non-square input."
    )

    if y9.shape != (1, 2, 8, 12):
        raise AssertionError(f"Expected output shape (1, 2, 8, 12), got {y9.shape}")

