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
    if not hasattr(candidate, "ViT"):
        raise AssertionError("Candidate must define class `ViT`.")

    torch.manual_seed(42)

    # --- Test 1: Basic output shape (small model) ---
    model = candidate.ViT(
        image_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    _assert_shape(logits, (2, 10), "Test 1: Output shape mismatch for small model.")
    _assert_finite(logits, "Test 1: Output contains non-finite values.")

    # --- Test 2: Larger model with standard ViT-like config ---
    model2 = candidate.ViT(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=192,
        num_heads=3,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    model2.eval()

    x2 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits2 = model2(x2)

    _assert_shape(logits2, (1, 1000), "Test 2: Output shape mismatch for larger model.")
    _assert_finite(logits2, "Test 2: Output contains non-finite values.")

    # --- Test 3: Verify number of patches is correct ---
    # 224/16 = 14, so 14*14 = 196 patches + 1 CLS = 197 tokens
    # This is implicitly tested by the model working correctly

    # --- Test 4: Different patch sizes ---
    model3 = candidate.ViT(
        image_size=64,
        patch_size=4,
        in_channels=3,
        num_classes=5,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    model3.eval()

    x3 = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        logits3 = model3(x3)

    _assert_shape(logits3, (4, 5), "Test 3: Output shape mismatch for custom patch size.")
    _assert_finite(logits3, "Test 3: Output contains non-finite values.")

    # --- Test 5: Gradients flow correctly ---
    model4 = candidate.ViT(
        image_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    model4.train()

    x4 = torch.randn(2, 3, 32, 32, requires_grad=True)
    logits4 = model4(x4)
    loss = logits4.sum()
    loss.backward()

    if x4.grad is None:
        raise AssertionError("Test 5: Expected non-None gradient for input.")
    _assert_finite(x4.grad, "Test 5: Input gradient contains non-finite values.")

    # Check that model parameters have gradients
    has_grad = False
    for param in model4.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    if not has_grad:
        raise AssertionError("Test 5: No parameter received non-zero gradient.")

    # --- Test 6: Single channel input ---
    model5 = candidate.ViT(
        image_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        num_layers=1,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    model5.eval()

    x5 = torch.randn(3, 1, 28, 28)
    with torch.no_grad():
        logits5 = model5(x5)

    _assert_shape(logits5, (3, 10), "Test 6: Output shape mismatch for single channel.")
    _assert_finite(logits5, "Test 6: Output contains non-finite values.")

    # --- Test 7: Determinism (same input -> same output in eval mode) ---
    model6 = candidate.ViT(
        image_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    model6.eval()

    x6 = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out1 = model6(x6)
        out2 = model6(x6)

    if not torch.allclose(out1, out2, atol=1e-6):
        raise AssertionError("Test 7: Model is not deterministic in eval mode.")




