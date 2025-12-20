from __future__ import annotations

from types import ModuleType

import torch
import torch.nn as nn


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "LeNet5"):
        raise AssertionError("Candidate must define class `LeNet5`.")

    torch.manual_seed(42)

    # --- test 1: output shape for single image ---
    model = candidate.LeNet5()
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    assert out.shape == torch.Size([1, 10]), (
        f"Expected output shape (1, 10), got {out.shape}"
    )

    # --- test 2: output shape for batch ---
    x_batch = torch.randn(16, 1, 32, 32)
    out_batch = model(x_batch)
    assert out_batch.shape == torch.Size([16, 10]), (
        f"Expected output shape (16, 10), got {out_batch.shape}"
    )

    # --- test 3: model is an nn.Module ---
    assert isinstance(model, nn.Module), "LeNet5 must be a subclass of nn.Module"

    # --- test 4: verify architecture has expected layers ---
    # Check conv layers exist with correct channel dimensions
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(conv_layers) >= 2, (
        f"Expected at least 2 Conv2d layers, found {len(conv_layers)}"
    )

    # Check first conv: 1 -> 6
    assert conv_layers[0].in_channels == 1, (
        f"First conv should have 1 input channel, got {conv_layers[0].in_channels}"
    )
    assert conv_layers[0].out_channels == 6, (
        f"First conv should have 6 output channels, got {conv_layers[0].out_channels}"
    )

    # Check second conv: 6 -> 16
    assert conv_layers[1].in_channels == 6, (
        f"Second conv should have 6 input channels, got {conv_layers[1].in_channels}"
    )
    assert conv_layers[1].out_channels == 16, (
        f"Second conv should have 16 output channels, got {conv_layers[1].out_channels}"
    )

    # Check linear layers
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) >= 3, (
        f"Expected at least 3 Linear layers, found {len(linear_layers)}"
    )

    # Check FC dimensions
    assert linear_layers[0].out_features == 120, (
        f"FC1 should have 120 output features, got {linear_layers[0].out_features}"
    )
    assert linear_layers[1].out_features == 84, (
        f"FC2 should have 84 output features, got {linear_layers[1].out_features}"
    )
    assert linear_layers[2].out_features == 10, (
        f"FC3 should have 10 output features, got {linear_layers[2].out_features}"
    )

    # --- test 5: gradient flow ---
    model.zero_grad()
    x = torch.randn(4, 1, 32, 32, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.shape == x.shape, "Input gradient shape should match input shape"

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"

    # --- test 6: determinism (same input -> same output) ---
    model.eval()
    x_test = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out1 = model(x_test)
        out2 = model(x_test)
    assert torch.allclose(out1, out2), "Model should be deterministic"

    # --- test 7: different inputs produce different outputs ---
    x1 = torch.randn(1, 1, 32, 32)
    x2 = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2), (
        "Different inputs should produce different outputs"
    )




