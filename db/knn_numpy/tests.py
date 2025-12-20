from __future__ import annotations

from types import ModuleType

import numpy as np


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "knn_predict"):
        raise AssertionError("Candidate must define function `knn_predict`.")

    knn_predict = candidate.knn_predict

    rng = np.random.default_rng(42)

    # --- Test 1: Simple 2D classification ---
    X_train_1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0]
    ])
    y_train_1 = np.array([0, 0, 0, 1, 1, 1])
    X_test_1 = np.array([
        [0.5, 0.5],
        [5.5, 5.5]
    ])

    predictions_1 = knn_predict(X_train_1, y_train_1, X_test_1, k=3)
    expected_1 = np.array([0, 1])
    if not np.array_equal(predictions_1, expected_1):
        raise AssertionError(
            f"Test 1 failed (simple 2D classification).\n"
            f"expected={expected_1}\nactual={predictions_1}"
        )

    # --- Test 2: k=1 (nearest neighbor) ---
    X_train_2 = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    y_train_2 = np.array([0, 0, 0, 1, 1])
    X_test_2 = np.array([[0.5], [10.5]])

    predictions_2 = knn_predict(X_train_2, y_train_2, X_test_2, k=1)
    expected_2 = np.array([0, 1])
    if not np.array_equal(predictions_2, expected_2):
        raise AssertionError(
            f"Test 2 failed (k=1).\n"
            f"expected={expected_2}\nactual={predictions_2}"
        )

    # --- Test 3: Tie-breaking (return smallest label) ---
    X_train_3 = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train_3 = np.array([0, 1, 0, 1])
    X_test_3 = np.array([[1.5]])

    predictions_3 = knn_predict(X_train_3, y_train_3, X_test_3, k=2)
    expected_3 = np.array([0])
    if not np.array_equal(predictions_3, expected_3):
        raise AssertionError(
            f"Test 3 failed (tie-breaking).\n"
            f"expected={expected_3}\nactual={predictions_3}"
        )

    # --- Test 4: Multi-class classification ---
    X_train_4 = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 0.0],
        [6.0, 1.0],
        [0.0, 5.0],
        [1.0, 6.0]
    ])
    y_train_4 = np.array([0, 0, 1, 1, 2, 2])
    X_test_4 = np.array([
        [0.5, 0.5],   # class 0
        [5.5, 0.5],   # class 1
        [0.5, 5.5]    # class 2
    ])

    predictions_4 = knn_predict(X_train_4, y_train_4, X_test_4, k=2)
    expected_4 = np.array([0, 1, 2])
    if not np.array_equal(predictions_4, expected_4):
        raise AssertionError(
            f"Test 4 failed (multi-class).\n"
            f"expected={expected_4}\nactual={predictions_4}"
        )

    # --- Test 5: Larger k with clear majority ---
    X_train_5 = np.array([
        [0.0], [0.1], [0.2], [0.3], [0.4],  # class 0
        [10.0], [10.1]                       # class 1
    ])
    y_train_5 = np.array([0, 0, 0, 0, 0, 1, 1])
    X_test_5 = np.array([[0.15]])

    predictions_5 = knn_predict(X_train_5, y_train_5, X_test_5, k=5)
    expected_5 = np.array([0])
    if not np.array_equal(predictions_5, expected_5):
        raise AssertionError(
            f"Test 5 failed (larger k).\n"
            f"expected={expected_5}\nactual={predictions_5}"
        )

    # --- Test 6: Random data with known structure ---
    n_per_class = 20
    n_features = 3
    n_classes = 3

    class_centers = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [-10.0, 5.0, 0.0]
    ])

    X_train_parts = []
    y_train_parts = []
    for c in range(n_classes):
        points = class_centers[c] + rng.normal(0, 0.5, (n_per_class, n_features))
        X_train_parts.append(points)
        y_train_parts.append(np.full(n_per_class, c))

    X_train_6 = np.vstack(X_train_parts)
    y_train_6 = np.concatenate(y_train_parts)

    # Test points close to each class center
    X_test_6 = class_centers + rng.normal(0, 0.1, (n_classes, n_features))

    predictions_6 = knn_predict(X_train_6, y_train_6, X_test_6, k=5)
    expected_6 = np.array([0, 1, 2])
    if not np.array_equal(predictions_6, expected_6):
        raise AssertionError(
            f"Test 6 failed (random structured data).\n"
            f"expected={expected_6}\nactual={predictions_6}"
        )

    # --- Test 7: Output shape verification ---
    X_train_7 = rng.normal(size=(50, 4))
    y_train_7 = rng.integers(0, 3, size=50)
    X_test_7 = rng.normal(size=(20, 4))

    predictions_7 = knn_predict(X_train_7, y_train_7, X_test_7, k=5)
    if predictions_7.shape != (20,):
        raise AssertionError(
            f"Test 7 failed (output shape).\n"
            f"expected shape=(20,), got {predictions_7.shape}"
        )

    # --- Test 8: k equals n_train (use all neighbors) ---
    X_train_8 = np.array([[0.0], [1.0], [2.0]])
    y_train_8 = np.array([0, 0, 1])
    X_test_8 = np.array([[1.0]])

    predictions_8 = knn_predict(X_train_8, y_train_8, X_test_8, k=3)
    expected_8 = np.array([0])  # 2 votes for 0, 1 vote for 1
    if not np.array_equal(predictions_8, expected_8):
        raise AssertionError(
            f"Test 8 failed (k=n_train).\n"
            f"expected={expected_8}\nactual={predictions_8}"
        )

    # --- Test 9: Single test point ---
    X_train_9 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y_train_9 = np.array([0, 1, 1])
    X_test_9 = np.array([[1.5, 1.5]])

    predictions_9 = knn_predict(X_train_9, y_train_9, X_test_9, k=2)
    expected_9 = np.array([1])  # nearest are [1,1] and [2,2], both class 1
    if not np.array_equal(predictions_9, expected_9):
        raise AssertionError(
            f"Test 9 failed (single test point).\n"
            f"expected={expected_9}\nactual={predictions_9}"
        )

    # --- Test 10: Higher dimensional data ---
    X_train_10 = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0, 11.0, 11.0]
    ])
    y_train_10 = np.array([0, 0, 1, 1])
    X_test_10 = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [10.5, 10.5, 10.5, 10.5, 10.5]
    ])

    predictions_10 = knn_predict(X_train_10, y_train_10, X_test_10, k=2)
    expected_10 = np.array([0, 1])
    if not np.array_equal(predictions_10, expected_10):
        raise AssertionError(
            f"Test 10 failed (higher dimensional).\n"
            f"expected={expected_10}\nactual={predictions_10}"
        )




