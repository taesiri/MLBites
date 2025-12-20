from __future__ import annotations

from types import ModuleType

import torch


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "LinearProbe"):
        raise AssertionError("Candidate must define class `LinearProbe`.")

    LinearProbe = candidate.LinearProbe

    # --- Test 1: Basic 2-class classification with separable data ---
    torch.manual_seed(42)

    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    probe = LinearProbe(feature_dim=2, num_classes=2, lr=0.5)
    losses = probe.fit(features, labels, epochs=100)

    # Check that losses is a list of correct length
    assert isinstance(losses, list), "fit() must return a list"
    assert len(losses) == 100, f"Expected 100 losses, got {len(losses)}"

    # Check that loss decreased
    assert losses[-1] < losses[0], (
        f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )

    # Check predictions on training data
    preds = probe.predict(features)
    assert preds.shape == (6,), f"Expected shape (6,), got {preds.shape}"
    assert (preds == labels).all(), f"Expected {labels.tolist()}, got {preds.tolist()}"

    # --- Test 2: Multi-class classification (10 classes) ---
    torch.manual_seed(123)

    # Create somewhat separable features for 10 classes
    num_samples_per_class = 20
    feature_dim = 64
    num_classes = 10

    all_features = []
    all_labels = []
    for c in range(num_classes):
        # Class c has features centered around a random point
        center = torch.randn(feature_dim) * 2
        class_features = center + torch.randn(num_samples_per_class, feature_dim) * 0.5
        all_features.append(class_features)
        all_labels.append(torch.full((num_samples_per_class,), c, dtype=torch.long))

    features_10 = torch.cat(all_features, dim=0)
    labels_10 = torch.cat(all_labels, dim=0)

    probe_10 = LinearProbe(feature_dim=feature_dim, num_classes=num_classes, lr=0.1)
    losses_10 = probe_10.fit(features_10, labels_10, epochs=200)

    assert len(losses_10) == 200, f"Expected 200 losses, got {len(losses_10)}"
    assert losses_10[-1] < losses_10[0], "Loss should decrease for multi-class"

    preds_10 = probe_10.predict(features_10)
    accuracy = (preds_10 == labels_10).float().mean().item()
    assert accuracy > 0.8, f"Expected accuracy > 0.8, got {accuracy:.2f}"

    # --- Test 3: Verify predict() output shape and dtype ---
    torch.manual_seed(0)
    test_features = torch.randn(5, feature_dim)
    test_preds = probe_10.predict(test_features)

    assert test_preds.shape == (5,), f"Expected shape (5,), got {test_preds.shape}"
    assert test_preds.dtype == torch.long or test_preds.dtype == torch.int64, (
        f"Expected integer dtype, got {test_preds.dtype}"
    )

    # --- Test 4: Single sample prediction ---
    single_feature = torch.randn(1, 2)
    single_probe = LinearProbe(feature_dim=2, num_classes=3, lr=0.1)
    # Just initialize, don't train - should still predict something
    single_pred = single_probe.predict(single_feature)
    assert single_pred.shape == (1,), f"Expected shape (1,), got {single_pred.shape}"
    assert 0 <= single_pred.item() < 3, "Prediction should be in [0, num_classes)"

    # --- Test 5: Verify gradients are properly zeroed (no accumulation) ---
    torch.manual_seed(999)
    features_small = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels_small = torch.tensor([0, 1])

    probe_grad = LinearProbe(feature_dim=2, num_classes=2, lr=0.01)

    # Get initial weight
    initial_weight = probe_grad.linear.weight.clone()

    # Train for 1 epoch
    probe_grad.fit(features_small, labels_small, epochs=1)
    weight_after_1 = probe_grad.linear.weight.clone()

    # Train for 1 more epoch (separate call)
    probe_grad.fit(features_small, labels_small, epochs=1)
    weight_after_2 = probe_grad.linear.weight.clone()

    # The changes should be reasonable (not exploding due to gradient accumulation)
    delta1 = (weight_after_1 - initial_weight).abs().max().item()
    delta2 = (weight_after_2 - weight_after_1).abs().max().item()

    # Both deltas should be small and similar in magnitude
    assert delta1 < 1.0, f"First update too large: {delta1}"
    assert delta2 < 1.0, f"Second update too large: {delta2}"




