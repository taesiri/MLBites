from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_allclose(
    a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, msg: str
) -> None:
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "assign_clusters"):
        raise AssertionError("Candidate must define function `assign_clusters`.")
    if not hasattr(candidate, "update_centroids"):
        raise AssertionError("Candidate must define function `update_centroids`.")

    assign_clusters = candidate.assign_clusters
    update_centroids = candidate.update_centroids

    rng = np.random.default_rng(42)
    atol = 1e-10
    rtol = 1e-10

    # --- Test 1: Simple 2D clustering ---
    X1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [5.0, 5.0],
        [6.0, 5.0]
    ])
    centroids1 = np.array([
        [0.5, 0.0],
        [5.5, 5.0]
    ])

    assignments1 = assign_clusters(X1, centroids1)
    expected_assignments1 = np.array([0, 0, 1, 1])
    if not np.array_equal(assignments1, expected_assignments1):
        raise AssertionError(
            f"assign_clusters failed on simple 2D test.\n"
            f"expected={expected_assignments1}\nactual={assignments1}"
        )

    new_centroids1 = update_centroids(X1, assignments1, k=2)
    expected_centroids1 = np.array([[0.5, 0.0], [5.5, 5.0]])
    _assert_allclose(
        new_centroids1, expected_centroids1,
        atol=atol, rtol=rtol,
        msg="update_centroids failed on simple 2D test."
    )

    # --- Test 2: 1D data ---
    X2 = np.array([[1.0], [2.0], [10.0], [11.0], [12.0]])
    centroids2 = np.array([[0.0], [15.0]])

    assignments2 = assign_clusters(X2, centroids2)
    expected_assignments2 = np.array([0, 0, 1, 1, 1])
    if not np.array_equal(assignments2, expected_assignments2):
        raise AssertionError(
            f"assign_clusters failed on 1D test.\n"
            f"expected={expected_assignments2}\nactual={assignments2}"
        )

    new_centroids2 = update_centroids(X2, assignments2, k=2)
    expected_centroids2 = np.array([[1.5], [11.0]])
    _assert_allclose(
        new_centroids2, expected_centroids2,
        atol=atol, rtol=rtol,
        msg="update_centroids failed on 1D test."
    )

    # --- Test 3: Empty cluster handling ---
    X3 = np.array([[1.0, 1.0], [2.0, 2.0]])
    assignments3 = np.array([0, 0])

    new_centroids3 = update_centroids(X3, assignments3, k=2)
    expected_centroids3 = np.array([[1.5, 1.5], [0.0, 0.0]])
    _assert_allclose(
        new_centroids3, expected_centroids3,
        atol=atol, rtol=rtol,
        msg="update_centroids failed on empty cluster test."
    )

    # --- Test 4: Random data with known structure ---
    n_per_cluster = 20
    n_features = 3
    k = 3

    true_centers = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [-10.0, 5.0, 0.0]
    ])

    X4_parts = []
    for i in range(k):
        cluster_points = true_centers[i] + rng.normal(0, 0.1, (n_per_cluster, n_features))
        X4_parts.append(cluster_points)
    X4 = np.vstack(X4_parts)

    assignments4 = assign_clusters(X4, true_centers)

    for i in range(k):
        start_idx = i * n_per_cluster
        end_idx = (i + 1) * n_per_cluster
        cluster_assignments = assignments4[start_idx:end_idx]
        if not np.all(cluster_assignments == i):
            raise AssertionError(
                f"assign_clusters failed: points from cluster {i} were not all assigned to cluster {i}.\n"
                f"assignments={cluster_assignments}"
            )

    new_centroids4 = update_centroids(X4, assignments4, k=k)
    _assert_allclose(
        new_centroids4, true_centers,
        atol=0.1, rtol=0.1,
        msg="update_centroids did not recover approximate true centers."
    )

    # --- Test 5: Verify one full iteration of k-means ---
    X5 = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0]
    ])
    initial_centroids5 = np.array([
        [0.0, 0.0],
        [10.0, 10.0]
    ])

    assignments5 = assign_clusters(X5, initial_centroids5)
    expected_assignments5 = np.array([0, 0, 0, 1, 1, 1])
    if not np.array_equal(assignments5, expected_assignments5):
        raise AssertionError(
            f"assign_clusters failed on full iteration test.\n"
            f"expected={expected_assignments5}\nactual={assignments5}"
        )

    new_centroids5 = update_centroids(X5, assignments5, k=2)
    expected_centroids5 = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [31.0 / 3.0, 31.0 / 3.0]
    ])
    _assert_allclose(
        new_centroids5, expected_centroids5,
        atol=atol, rtol=rtol,
        msg="update_centroids failed on full iteration test."
    )

    # --- Test 6: Edge case - single point per cluster ---
    X6 = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    assignments6 = np.array([0, 1, 2])

    new_centroids6 = update_centroids(X6, assignments6, k=3)
    expected_centroids6 = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    _assert_allclose(
        new_centroids6, expected_centroids6,
        atol=atol, rtol=rtol,
        msg="update_centroids failed on single point per cluster test."
    )

    # --- Test 7: Verify output shapes ---
    X7 = rng.normal(size=(100, 5))
    centroids7 = rng.normal(size=(4, 5))

    assignments7 = assign_clusters(X7, centroids7)
    if assignments7.shape != (100,):
        raise AssertionError(
            f"assign_clusters returned wrong shape: expected (100,), got {assignments7.shape}"
        )

    new_centroids7 = update_centroids(X7, assignments7, k=4)
    if new_centroids7.shape != (4, 5):
        raise AssertionError(
            f"update_centroids returned wrong shape: expected (4, 5), got {new_centroids7.shape}"
        )


