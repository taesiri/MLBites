from __future__ import annotations

from types import ModuleType

import torch
import torch.nn as nn


def _assert_shape(tensor: torch.Tensor, expected: tuple, msg: str) -> None:
    if tensor.shape != expected:
        raise AssertionError(f"{msg}: expected shape {expected}, got {tensor.shape}")


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required classes for this question.

    Raises:
        AssertionError: if any test fails.
    """
    # Check required classes exist
    required_classes = ["CausalSelfAttention", "MLP", "TransformerBlock", "GPT2"]
    for cls_name in required_classes:
        if not hasattr(candidate, cls_name):
            raise AssertionError(f"Candidate must define class `{cls_name}`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- Test 1: MLP shape and forward pass ---
    mlp = candidate.MLP(embed_dim=16, dropout_p=0.0).to(dtype=dtype)
    mlp.eval()
    x_mlp = torch.randn(2, 5, 16, dtype=dtype)
    y_mlp = mlp(x_mlp)
    _assert_shape(y_mlp, (2, 5, 16), "MLP output shape mismatch")

    # Check MLP has correct hidden dimension (4x expansion)
    fc1_out_features = mlp.fc1.out_features if hasattr(mlp, "fc1") else None
    if fc1_out_features != 64:
        raise AssertionError(
            f"MLP fc1 should have out_features=4*embed_dim=64, got {fc1_out_features}"
        )

    # --- Test 2: CausalSelfAttention shape ---
    attn = candidate.CausalSelfAttention(
        embed_dim=8, num_heads=2, block_size=16, dropout_p=0.0
    ).to(dtype=dtype)
    attn.eval()
    x_attn = torch.randn(3, 10, 8, dtype=dtype)
    y_attn = attn(x_attn)
    _assert_shape(y_attn, (3, 10, 8), "CausalSelfAttention output shape mismatch")

    # --- Test 3: Causal masking (future tokens should not affect past outputs) ---
    torch.manual_seed(123)
    attn2 = candidate.CausalSelfAttention(
        embed_dim=8, num_heads=2, block_size=8, dropout_p=0.0
    ).to(dtype=dtype)
    attn2.eval()

    x_causal = torch.randn(1, 6, 8, dtype=dtype)
    y_causal = attn2(x_causal)

    # Modify last two tokens - earlier outputs should remain unchanged
    x_modified = x_causal.clone()
    x_modified[0, 4:, :] = torch.randn(2, 8, dtype=dtype) * 100
    y_modified = attn2(x_modified)

    # First 4 positions should be identical
    if not torch.allclose(y_causal[0, :4], y_modified[0, :4], atol=1e-12, rtol=1e-12):
        raise AssertionError(
            "Causal masking failed: modifying future tokens affected past outputs"
        )

    # --- Test 4: TransformerBlock shape ---
    block = candidate.TransformerBlock(
        embed_dim=16, num_heads=4, block_size=32, dropout_p=0.0
    ).to(dtype=dtype)
    block.eval()
    x_block = torch.randn(2, 8, 16, dtype=dtype)
    y_block = block(x_block)
    _assert_shape(y_block, (2, 8, 16), "TransformerBlock output shape mismatch")

    # --- Test 5: GPT2 full model shape ---
    model = candidate.GPT2(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        block_size=64,
        dropout_p=0.0,
    ).to(dtype=dtype)
    model.eval()

    idx = torch.randint(0, 100, (4, 20))
    logits = model(idx)
    _assert_shape(logits, (4, 20, 100), "GPT2 logits shape mismatch")

    # --- Test 6: GPT2 causal property (changing future tokens) ---
    torch.manual_seed(456)
    model2 = candidate.GPT2(
        vocab_size=50,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        block_size=32,
        dropout_p=0.0,
    ).to(dtype=dtype)
    model2.eval()

    idx1 = torch.randint(0, 50, (1, 10))
    logits1 = model2(idx1)

    idx2 = idx1.clone()
    idx2[0, 7:] = torch.randint(0, 50, (3,))  # change last 3 tokens
    logits2 = model2(idx2)

    # First 7 positions should have same logits
    if not torch.allclose(logits1[0, :7], logits2[0, :7], atol=1e-10, rtol=1e-10):
        raise AssertionError(
            "GPT2 causal property failed: future token changes affected past logits"
        )

    # --- Test 7: Gradient flow ---
    model.train()
    idx_grad = torch.randint(0, 100, (2, 10))
    logits_grad = model(idx_grad)
    loss = logits_grad.sum()
    loss.backward()

    # Check gradients exist for key parameters
    for name, param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"No gradient for parameter: {name}")
        if not torch.isfinite(param.grad).all():
            raise AssertionError(f"Non-finite gradient for parameter: {name}")

    # --- Test 8: Check pre-normalization structure in TransformerBlock ---
    # The block should have ln1, attn, ln2, mlp structure
    block_test = candidate.TransformerBlock(
        embed_dim=8, num_heads=2, block_size=16, dropout_p=0.0
    )
    required_attrs = ["ln1", "ln2", "attn", "mlp"]
    for attr in required_attrs:
        if not hasattr(block_test, attr):
            raise AssertionError(
                f"TransformerBlock missing attribute `{attr}` for pre-norm structure"
            )

    # --- Test 9: Different sequence lengths work correctly ---
    model.eval()
    for seq_len in [1, 5, 15, 30]:
        idx_test = torch.randint(0, 100, (2, seq_len))
        logits_test = model(idx_test)
        _assert_shape(
            logits_test,
            (2, seq_len, 100),
            f"Shape mismatch for sequence length {seq_len}",
        )

    # --- Test 10: Single token input works ---
    idx_single = torch.randint(0, 100, (1, 1))
    logits_single = model(idx_single)
    _assert_shape(logits_single, (1, 1, 100), "Single token input shape mismatch")




