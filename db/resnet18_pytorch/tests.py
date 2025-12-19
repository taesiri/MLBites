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
    if not hasattr(candidate, "BasicBlock"):
        raise AssertionError("Candidate must define class `BasicBlock`.")
    if not hasattr(candidate, "ResNet18"):
        raise AssertionError("Candidate must define class `ResNet18`.")

    torch.manual_seed(42)

    # --- test 1: BasicBlock output shape (no downsampling) ---
    block = candidate.BasicBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 56, 56)
    out = block(x)
    assert out.shape == torch.Size([2, 64, 56, 56]), (
        f"BasicBlock(64->64, stride=1): expected shape (2, 64, 56, 56), got {out.shape}"
    )

    # --- test 2: BasicBlock output shape (with downsampling) ---
    block_down = candidate.BasicBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 56, 56)
    out = block_down(x)
    assert out.shape == torch.Size([2, 128, 28, 28]), (
        f"BasicBlock(64->128, stride=2): expected shape (2, 128, 28, 28), got {out.shape}"
    )

    # --- test 3: BasicBlock is nn.Module ---
    assert isinstance(block, nn.Module), "BasicBlock must be a subclass of nn.Module"

    # --- test 4: ResNet18 output shape (default 1000 classes) ---
    model = candidate.ResNet18()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == torch.Size([1, 1000]), (
        f"ResNet18(1000): expected output shape (1, 1000), got {out.shape}"
    )

    # --- test 5: ResNet18 output shape (custom classes) ---
    model_10 = candidate.ResNet18(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    out = model_10(x)
    assert out.shape == torch.Size([4, 10]), (
        f"ResNet18(10): expected output shape (4, 10), got {out.shape}"
    )

    # --- test 6: ResNet18 is nn.Module ---
    assert isinstance(model, nn.Module), "ResNet18 must be a subclass of nn.Module"

    # --- test 7: verify architecture has expected layer structure ---
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    # ResNet-18 has: 1 initial conv + 8 BasicBlocks * 2 convs = 17 convs + downsample convs
    assert len(conv_layers) >= 17, (
        f"Expected at least 17 Conv2d layers, found {len(conv_layers)}"
    )

    # Check initial conv
    assert conv_layers[0].in_channels == 3, (
        f"Initial conv should have 3 input channels, got {conv_layers[0].in_channels}"
    )
    assert conv_layers[0].out_channels == 64, (
        f"Initial conv should have 64 output channels, got {conv_layers[0].out_channels}"
    )
    assert conv_layers[0].kernel_size == (7, 7), (
        f"Initial conv should have kernel_size 7, got {conv_layers[0].kernel_size}"
    )

    # Check final fc layer
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) >= 1, "Expected at least 1 Linear layer"
    assert linear_layers[-1].in_features == 512, (
        f"FC should have 512 input features, got {linear_layers[-1].in_features}"
    )
    assert linear_layers[-1].out_features == 1000, (
        f"FC should have 1000 output features, got {linear_layers[-1].out_features}"
    )

    # --- test 8: verify BatchNorm layers exist ---
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_layers) >= 17, (
        f"Expected at least 17 BatchNorm2d layers, found {len(bn_layers)}"
    )

    # --- test 9: gradient flow ---
    model_10.zero_grad()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    out = model_10(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.shape == x.shape, "Input gradient shape should match input shape"

    # Check all parameters have gradients
    for name, param in model_10.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"

    # --- test 10: determinism (same input -> same output) ---
    model_10.eval()
    x_test = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out1 = model_10(x_test)
        out2 = model_10(x_test)
    assert torch.allclose(out1, out2), "Model should be deterministic"

    # --- test 11: BasicBlock skip connection works (residual property) ---
    # Test that setting all weights to zero still produces non-zero output
    # due to skip connection
    block_test = candidate.BasicBlock(64, 64, stride=1)
    block_test.eval()
    with torch.no_grad():
        # Zero out conv weights
        block_test.conv1.weight.zero_()
        block_test.conv2.weight.zero_()
        # BN in eval mode with default params acts as identity
        x_skip = torch.randn(1, 64, 8, 8)
        out_skip = block_test(x_skip)
        # Output should be relu(x) since conv outputs are zero
        expected = torch.relu(x_skip)
        assert torch.allclose(out_skip, expected, atol=1e-5), (
            "Skip connection should pass input through when conv weights are zero"
        )

    # --- test 12: different inputs produce different outputs ---
    model.eval()
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2), (
        "Different inputs should produce different outputs"
    )


