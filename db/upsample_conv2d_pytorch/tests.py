from __future__ import annotations

from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}")


def _build_reference(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    scale_factor: int,
    mode: str,
    padding: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> nn.Module:
    """Build a reference module with the same weights for comparison."""

    class RefUpsampleConv2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale_factor = scale_factor
            self.mode = mode
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            with torch.no_grad():
                self.conv.weight.copy_(conv_weight)
                self.conv.bias.copy_(conv_bias)

        def forward(self, x):
            if self.mode == "bilinear":
                x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
            else:
                x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
            return self.conv(x)

    return RefUpsampleConv2d()


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "UpsampleConv2d"):
        raise AssertionError("Candidate must define class `UpsampleConv2d`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- Test 1: Basic 2x upsampling with nearest neighbor ---
    in_ch, out_ch, ksize = 3, 8, 3
    candidate_layer = candidate.UpsampleConv2d(in_ch, out_ch, ksize, scale_factor=2, mode="nearest").to(dtype=dtype)

    ref_layer = _build_reference(
        in_ch, out_ch, ksize, 2, "nearest", 0,
        candidate_layer.conv.weight, candidate_layer.conv.bias
    ).to(dtype=dtype)

    x1 = torch.randn(2, in_ch, 8, 8, dtype=dtype)
    y1 = candidate_layer(x1)
    y1_ref = ref_layer(x1)

    _assert_allclose(y1, y1_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch (basic nearest 2x).")

    expected_h1 = 8 * 2 - 3 + 1  # 14
    if y1.shape != (2, out_ch, expected_h1, expected_h1):
        raise AssertionError(f"Expected shape (2, {out_ch}, {expected_h1}, {expected_h1}), got {y1.shape}")

    # --- Test 2: With padding to preserve upsampled size ---
    candidate_layer2 = candidate.UpsampleConv2d(8, 4, kernel_size=3, scale_factor=2, padding=1).to(dtype=dtype)
    ref_layer2 = _build_reference(
        8, 4, 3, 2, "nearest", 1,
        candidate_layer2.conv.weight, candidate_layer2.conv.bias
    ).to(dtype=dtype)

    x2 = torch.randn(2, 8, 16, 16, dtype=dtype)
    y2 = candidate_layer2(x2)
    y2_ref = ref_layer2(x2)

    _assert_allclose(y2, y2_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with padding=1.")

    expected_h2 = 16 * 2 + 2 * 1 - 3 + 1  # 32
    if y2.shape != (2, 4, expected_h2, expected_h2):
        raise AssertionError(f"Expected shape (2, 4, {expected_h2}, {expected_h2}), got {y2.shape}")

    # --- Test 3: Bilinear upsampling ---
    candidate_layer3 = candidate.UpsampleConv2d(
        in_channels=1, out_channels=8, kernel_size=5,
        scale_factor=4, mode="bilinear", padding=2
    ).to(dtype=dtype)
    ref_layer3 = _build_reference(
        1, 8, 5, 4, "bilinear", 2,
        candidate_layer3.conv.weight, candidate_layer3.conv.bias
    ).to(dtype=dtype)

    x3 = torch.randn(1, 1, 7, 7, dtype=dtype)
    y3 = candidate_layer3(x3)
    y3_ref = ref_layer3(x3)

    _assert_allclose(y3, y3_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with bilinear mode.")

    expected_h3 = 7 * 4 + 2 * 2 - 5 + 1  # 28
    if y3.shape != (1, 8, expected_h3, expected_h3):
        raise AssertionError(f"Expected shape (1, 8, {expected_h3}, {expected_h3}), got {y3.shape}")

    # --- Test 4: Scale factor = 1 (no upsampling, just conv) ---
    candidate_layer4 = candidate.UpsampleConv2d(2, 4, kernel_size=3, scale_factor=1, padding=1).to(dtype=dtype)
    ref_layer4 = _build_reference(
        2, 4, 3, 1, "nearest", 1,
        candidate_layer4.conv.weight, candidate_layer4.conv.bias
    ).to(dtype=dtype)

    x4 = torch.randn(1, 2, 10, 10, dtype=dtype)
    y4 = candidate_layer4(x4)
    y4_ref = ref_layer4(x4)

    _assert_allclose(y4, y4_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with scale_factor=1.")

    if y4.shape != (1, 4, 10, 10):
        raise AssertionError(f"Expected shape (1, 4, 10, 10), got {y4.shape}")

    # --- Test 5: Non-square input ---
    candidate_layer5 = candidate.UpsampleConv2d(1, 2, kernel_size=3, scale_factor=2, padding=1).to(dtype=dtype)
    ref_layer5 = _build_reference(
        1, 2, 3, 2, "nearest", 1,
        candidate_layer5.conv.weight, candidate_layer5.conv.bias
    ).to(dtype=dtype)

    x5 = torch.randn(1, 1, 8, 12, dtype=dtype)
    y5 = candidate_layer5(x5)
    y5_ref = ref_layer5(x5)

    _assert_allclose(y5, y5_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch for non-square input.")

    expected_h5 = 8 * 2 + 2 * 1 - 3 + 1  # 16
    expected_w5 = 12 * 2 + 2 * 1 - 3 + 1  # 24
    if y5.shape != (1, 2, expected_h5, expected_w5):
        raise AssertionError(f"Expected shape (1, 2, {expected_h5}, {expected_w5}), got {y5.shape}")

    # --- Test 6: Gradients flow correctly ---
    candidate_layer6 = candidate.UpsampleConv2d(2, 4, kernel_size=3, scale_factor=2, padding=1).to(dtype=dtype)
    x6 = torch.randn(2, 2, 6, 6, dtype=dtype, requires_grad=True)
    y6 = candidate_layer6(x6).sum()
    y6.backward()

    if x6.grad is None:
        raise AssertionError("Expected non-None gradient for input.")
    if not torch.isfinite(x6.grad).all():
        raise AssertionError("Found non-finite values in input gradient.")

    # --- Test 7: Weight and bias are trainable parameters ---
    candidate_layer7 = candidate.UpsampleConv2d(3, 8, kernel_size=3)
    param_names = {name for name, _ in candidate_layer7.named_parameters()}
    if "conv.weight" not in param_names:
        raise AssertionError("UpsampleConv2d must have a trainable 'conv.weight' parameter.")
    if "conv.bias" not in param_names:
        raise AssertionError("UpsampleConv2d must have a trainable 'conv.bias' parameter.")

    # --- Test 8: Correct weight shape ---
    candidate_layer8 = candidate.UpsampleConv2d(5, 10, kernel_size=4)
    expected_weight_shape = (10, 5, 4, 4)
    expected_bias_shape = (10,)

    if candidate_layer8.conv.weight.shape != expected_weight_shape:
        raise AssertionError(
            f"Weight shape should be {expected_weight_shape}, got {candidate_layer8.conv.weight.shape}"
        )
    if candidate_layer8.conv.bias.shape != expected_bias_shape:
        raise AssertionError(
            f"Bias shape should be {expected_bias_shape}, got {candidate_layer8.conv.bias.shape}"
        )

    # --- Test 9: Large scale factor (4x) with nearest ---
    candidate_layer9 = candidate.UpsampleConv2d(1, 4, kernel_size=3, scale_factor=4, padding=1).to(dtype=dtype)
    ref_layer9 = _build_reference(
        1, 4, 3, 4, "nearest", 1,
        candidate_layer9.conv.weight, candidate_layer9.conv.bias
    ).to(dtype=dtype)

    x9 = torch.randn(1, 1, 4, 4, dtype=dtype)
    y9 = candidate_layer9(x9)
    y9_ref = ref_layer9(x9)

    _assert_allclose(y9, y9_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with 4x upsampling.")

    expected_h9 = 4 * 4 + 2 * 1 - 3 + 1  # 16
    if y9.shape != (1, 4, expected_h9, expected_h9):
        raise AssertionError(f"Expected shape (1, 4, {expected_h9}, {expected_h9}), got {y9.shape}")

    # --- Test 10: Kernel size = 1 ---
    candidate_layer10 = candidate.UpsampleConv2d(4, 8, kernel_size=1, scale_factor=2).to(dtype=dtype)
    ref_layer10 = _build_reference(
        4, 8, 1, 2, "nearest", 0,
        candidate_layer10.conv.weight, candidate_layer10.conv.bias
    ).to(dtype=dtype)

    x10 = torch.randn(2, 4, 5, 5, dtype=dtype)
    y10 = candidate_layer10(x10)
    y10_ref = ref_layer10(x10)

    _assert_allclose(y10, y10_ref, atol=1e-10, rtol=1e-10, msg="Output mismatch with kernel_size=1.")

    expected_h10 = 5 * 2  # 10
    if y10.shape != (2, 8, expected_h10, expected_h10):
        raise AssertionError(f"Expected shape (2, 8, {expected_h10}, {expected_h10}), got {y10.shape}")



