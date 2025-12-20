from __future__ import annotations

from types import ModuleType

import torch


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "train_gpt2"):
        raise AssertionError("Candidate must define function `train_gpt2`.")

    # --- Test 1: Basic return types and shapes ---
    model, losses = candidate.train_gpt2(
        vocab_size=50,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        block_size=16,
        batch_size=2,
        num_steps=5,
        lr=1e-3,
        seed=42,
    )

    if not isinstance(model, torch.nn.Module):
        raise AssertionError("train_gpt2 should return an nn.Module as first element.")

    if not isinstance(losses, list):
        raise AssertionError("train_gpt2 should return a list of losses as second element.")

    if len(losses) != 5:
        raise AssertionError(f"Expected 5 losses for num_steps=5, got {len(losses)}.")

    # --- Test 2: Losses should be positive floats ---
    for i, loss in enumerate(losses):
        if not isinstance(loss, float):
            raise AssertionError(f"Loss at step {i} should be a float, got {type(loss)}.")
        if loss <= 0:
            raise AssertionError(f"Loss at step {i} should be positive, got {loss}.")
        if not torch.isfinite(torch.tensor(loss)):
            raise AssertionError(f"Loss at step {i} is not finite: {loss}.")

    # --- Test 3: Model output shape ---
    model.eval()
    idx = torch.randint(0, 50, (2, 10))
    with torch.no_grad():
        logits = model(idx)

    if logits.shape != (2, 10, 50):
        raise AssertionError(
            f"Model output shape mismatch: expected (2, 10, 50), got {logits.shape}."
        )

    # --- Test 4: Determinism with same seed ---
    _, losses1 = candidate.train_gpt2(
        vocab_size=30,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        block_size=8,
        batch_size=2,
        num_steps=3,
        lr=1e-3,
        seed=999,
    )

    _, losses2 = candidate.train_gpt2(
        vocab_size=30,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        block_size=8,
        batch_size=2,
        num_steps=3,
        lr=1e-3,
        seed=999,
    )

    if losses1 != losses2:
        raise AssertionError(
            f"Training should be deterministic with same seed.\n"
            f"Run 1: {losses1}\nRun 2: {losses2}"
        )

    # --- Test 5: Different seeds produce different results ---
    _, losses_a = candidate.train_gpt2(
        vocab_size=30,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        block_size=8,
        batch_size=2,
        num_steps=3,
        lr=1e-3,
        seed=111,
    )

    _, losses_b = candidate.train_gpt2(
        vocab_size=30,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        block_size=8,
        batch_size=2,
        num_steps=3,
        lr=1e-3,
        seed=222,
    )

    if losses_a == losses_b:
        raise AssertionError("Different seeds should produce different losses.")

    # --- Test 6: Loss decreases with training (enough steps) ---
    _, losses_train = candidate.train_gpt2(
        vocab_size=100,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        block_size=32,
        batch_size=8,
        num_steps=100,
        lr=3e-3,
        seed=42,
    )

    early_avg = sum(losses_train[:10]) / 10
    late_avg = sum(losses_train[-10:]) / 10

    if late_avg >= early_avg:
        raise AssertionError(
            f"Loss should generally decrease with training.\n"
            f"Early avg: {early_avg:.4f}, Late avg: {late_avg:.4f}"
        )

    # --- Test 7: Model has trainable parameters ---
    test_model, _ = candidate.train_gpt2(
        vocab_size=20,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        block_size=8,
        batch_size=2,
        num_steps=1,
        seed=42,
    )

    num_params = sum(p.numel() for p in test_model.parameters())
    if num_params == 0:
        raise AssertionError("Model should have trainable parameters.")

    # --- Test 8: Different input lengths work ---
    test_model.eval()
    for seq_len in [1, 4, 8]:
        idx_test = torch.randint(0, 20, (1, seq_len))
        with torch.no_grad():
            out = test_model(idx_test)
        if out.shape != (1, seq_len, 20):
            raise AssertionError(
                f"Model output shape wrong for seq_len={seq_len}: "
                f"expected (1, {seq_len}, 20), got {out.shape}"
            )

    # --- Test 9: Causal property (future tokens don't affect past outputs) ---
    torch.manual_seed(123)
    causal_model, _ = candidate.train_gpt2(
        vocab_size=50,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        block_size=16,
        batch_size=2,
        num_steps=1,
        seed=456,
    )
    causal_model.eval()

    idx1 = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        logits1 = causal_model(idx1)

    idx2 = idx1.clone()
    idx2[0, 5:] = torch.randint(0, 50, (3,))  # change last 3 tokens
    with torch.no_grad():
        logits2 = causal_model(idx2)

    # First 5 positions should have same logits
    if not torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-5, rtol=1e-5):
        raise AssertionError(
            "Model should be causal: future token changes affected past logits."
        )




