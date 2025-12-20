from __future__ import annotations

from types import ModuleType

import torch


def _assert_shape(tensor: torch.Tensor, expected: tuple, msg: str) -> None:
    if tensor.shape != expected:
        raise AssertionError(f"{msg}\nExpected shape {expected}, got {tensor.shape}")


def _assert_finite(tensor: torch.Tensor, msg: str) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{msg}\nFound non-finite values in tensor.")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "PatchProjection"):
        raise AssertionError("Candidate must define class `PatchProjection`.")

    torch.manual_seed(42)

    # --- Test 1: Basic output shape (small model) ---
    proj = candidate.PatchProjection(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
    )
    proj.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        patches = proj(x)

    # 32/8 = 4, so 4*4 = 16 patches
    _assert_shape(patches, (2, 16, 64), "Test 1: Output shape mismatch for small model.")
    _assert_finite(patches, "Test 1: Output contains non-finite values.")

    # --- Test 2: Standard ViT configuration ---
    proj2 = candidate.PatchProjection(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
    )
    proj2.eval()

    x2 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        patches2 = proj2(x2)

    # 224/16 = 14, so 14*14 = 196 patches
    _assert_shape(patches2, (1, 196, 768), "Test 2: Output shape mismatch for standard ViT config.")
    _assert_finite(patches2, "Test 2: Output contains non-finite values.")

    # --- Test 3: Single channel input (e.g., grayscale) ---
    proj3 = candidate.PatchProjection(
        image_size=28,
        patch_size=7,
        in_channels=1,
        embed_dim=128,
    )
    proj3.eval()

    x3 = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        patches3 = proj3(x3)

    # 28/7 = 4, so 4*4 = 16 patches
    _assert_shape(patches3, (4, 16, 128), "Test 3: Output shape mismatch for single channel.")
    _assert_finite(patches3, "Test 3: Output contains non-finite values.")

    # --- Test 4: Different patch sizes ---
    proj4 = candidate.PatchProjection(
        image_size=64,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
    )
    proj4.eval()

    x4 = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        patches4 = proj4(x4)

    # 64/4 = 16, so 16*16 = 256 patches
    _assert_shape(patches4, (3, 256, 256), "Test 4: Output shape mismatch for small patch size.")
    _assert_finite(patches4, "Test 4: Output contains non-finite values.")

    # --- Test 5: Gradients flow correctly ---
    proj5 = candidate.PatchProjection(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
    )
    proj5.train()

    x5 = torch.randn(2, 3, 32, 32, requires_grad=True)
    patches5 = proj5(x5)
    loss = patches5.sum()
    loss.backward()

    if x5.grad is None:
        raise AssertionError("Test 5: Expected non-None gradient for input.")
    _assert_finite(x5.grad, "Test 5: Input gradient contains non-finite values.")

    # Check that model parameters have gradients
    has_grad = False
    for param in proj5.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    if not has_grad:
        raise AssertionError("Test 5: No parameter received non-zero gradient.")

    # --- Test 6: Verify num_patches attribute ---
    proj6 = candidate.PatchProjection(
        image_size=48,
        patch_size=12,
        in_channels=3,
        embed_dim=96,
    )

    expected_num_patches = (48 // 12) ** 2  # 4*4 = 16
    if hasattr(proj6, "num_patches"):
        if proj6.num_patches != expected_num_patches:
            raise AssertionError(
                f"Test 6: num_patches attribute incorrect. "
                f"Expected {expected_num_patches}, got {proj6.num_patches}"
            )

    # --- Test 7: Determinism (same input -> same output in eval mode) ---
    proj7 = candidate.PatchProjection(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
    )
    proj7.eval()

    x7 = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out1 = proj7(x7)
        out2 = proj7(x7)

    if not torch.allclose(out1, out2, atol=1e-6):
        raise AssertionError("Test 7: Model is not deterministic in eval mode.")

    # --- Test 8: Batch size 1 ---
    proj8 = candidate.PatchProjection(
        image_size=16,
        patch_size=4,
        in_channels=3,
        embed_dim=32,
    )
    proj8.eval()

    x8 = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        patches8 = proj8(x8)

    # 16/4 = 4, so 4*4 = 16 patches
    _assert_shape(patches8, (1, 16, 32), "Test 8: Output shape mismatch for batch size 1.")
    _assert_finite(patches8, "Test 8: Output contains non-finite values.")



