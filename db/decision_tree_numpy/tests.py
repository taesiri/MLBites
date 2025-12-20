from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_close(a: float, b: float, *, atol: float, msg: str) -> None:
    if abs(a - b) > atol:
        raise AssertionError(f"{msg}\nexpected={b}, actual={a}, diff={abs(a - b)}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "compute_gini"):
        raise AssertionError("Candidate must define function `compute_gini`.")
    if not hasattr(candidate, "find_best_split"):
        raise AssertionError("Candidate must define function `find_best_split`.")

    compute_gini = candidate.compute_gini
    find_best_split = candidate.find_best_split

    atol = 1e-10

    # --- Test 1: Pure node (all same class) ---
    y1 = np.array([0, 0, 0, 0])
    gini1 = compute_gini(y1)
    _assert_close(gini1, 0.0, atol=atol, msg="compute_gini failed on pure node (all 0s).")

    y1b = np.array([1, 1, 1])
    gini1b = compute_gini(y1b)
    _assert_close(gini1b, 0.0, atol=atol, msg="compute_gini failed on pure node (all 1s).")

    # --- Test 2: Maximum impurity for binary (50-50 split) ---
    y2 = np.array([0, 0, 1, 1])
    gini2 = compute_gini(y2)
    _assert_close(gini2, 0.5, atol=atol, msg="compute_gini failed on 50-50 binary split.")

    # --- Test 3: Three classes uniform ---
    y3 = np.array([0, 1, 2])
    gini3 = compute_gini(y3)
    expected_gini3 = 1.0 - (1 / 9 + 1 / 9 + 1 / 9)  # 2/3
    _assert_close(gini3, expected_gini3, atol=atol, msg="compute_gini failed on 3-class uniform.")

    # --- Test 4: Imbalanced binary ---
    y4 = np.array([0, 0, 0, 1])
    gini4 = compute_gini(y4)
    expected_gini4 = 1.0 - (0.75**2 + 0.25**2)  # 1 - 0.5625 - 0.0625 = 0.375
    _assert_close(gini4, expected_gini4, atol=atol, msg="compute_gini failed on imbalanced binary.")

    # --- Test 5: Empty array ---
    y5 = np.array([], dtype=int)
    gini5 = compute_gini(y5)
    _assert_close(gini5, 0.0, atol=atol, msg="compute_gini failed on empty array.")

    # --- Test 6: Simple perfect split ---
    X6 = np.array([[1.0], [2.0], [3.0], [4.0]])
    y6 = np.array([0, 0, 1, 1])
    feat6, thresh6, gini6 = find_best_split(X6, y6)

    if feat6 != 0:
        raise AssertionError(f"find_best_split failed: expected feature 0, got {feat6}")
    if thresh6 != 2.0:
        raise AssertionError(f"find_best_split failed: expected threshold 2.0, got {thresh6}")
    _assert_close(gini6, 0.0, atol=atol, msg="find_best_split failed: expected gini 0.0 for perfect split.")

    # --- Test 7: Two features, both can achieve perfect split ---
    X7 = np.array([
        [1.0, 5.0],
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 2.0]
    ])
    y7 = np.array([0, 0, 1, 1])
    feat7, thresh7, gini7 = find_best_split(X7, y7)

    # Either feature 0 with threshold 2.0 or feature 1 with threshold 3.0 is valid
    _assert_close(gini7, 0.0, atol=atol, msg="find_best_split failed: expected gini 0.0 for two-feature test.")
    if feat7 not in (0, 1):
        raise AssertionError(f"find_best_split returned invalid feature index: {feat7}")

    # --- Test 8: Imperfect split ---
    X8 = np.array([[1.0], [2.0], [3.0], [4.0]])
    y8 = np.array([0, 1, 0, 1])  # Interleaved, no perfect split possible

    feat8, thresh8, gini8 = find_best_split(X8, y8)

    # Any split will have impurity > 0 but < 0.5
    if gini8 >= 0.5 or gini8 < 0:
        raise AssertionError(f"find_best_split failed on imperfect split: gini={gini8}")

    # --- Test 9: All same feature values (no valid split) ---
    X9 = np.array([[1.0], [1.0], [1.0], [1.0]])
    y9 = np.array([0, 0, 1, 1])
    feat9, thresh9, gini9 = find_best_split(X9, y9)

    if feat9 != -1:
        raise AssertionError(f"find_best_split should return -1 when no valid split exists, got {feat9}")
    if gini9 != float("inf"):
        raise AssertionError(f"find_best_split should return inf gini when no valid split exists, got {gini9}")

    # --- Test 10: Multi-class split ---
    X10 = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y10 = np.array([0, 0, 1, 1, 2, 2])
    feat10, thresh10, gini10 = find_best_split(X10, y10)

    # Best split at threshold 4.0: left=[0,0,1,1], right=[2,2]
    # Left gini = 1 - (0.5^2 + 0.5^2) = 0.5
    # Right gini = 0 (pure)
    # Weighted = (4/6)*0.5 + (2/6)*0 = 1/3
    expected_gini10 = 1.0 / 3.0
    _assert_close(gini10, expected_gini10, atol=atol, msg="find_best_split failed on multi-class split.")

    # --- Test 11: Random data verification ---
    rng = np.random.default_rng(42)
    n_samples = 50
    n_features = 5
    X11 = rng.standard_normal((n_samples, n_features))
    y11 = rng.integers(0, 3, size=n_samples)

    feat11, thresh11, gini11 = find_best_split(X11, y11)

    # Should return valid feature index and finite gini
    if feat11 < 0 or feat11 >= n_features:
        raise AssertionError(f"find_best_split returned invalid feature index: {feat11}")
    if not np.isfinite(gini11):
        raise AssertionError(f"find_best_split returned non-finite gini: {gini11}")
    if gini11 < 0 or gini11 > 1:
        raise AssertionError(f"find_best_split returned gini outside [0,1]: {gini11}")

    # --- Test 12: Verify the split is actually the best ---
    X12 = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y12 = np.array([0, 0, 1, 1, 1])

    feat12, thresh12, gini12 = find_best_split(X12, y12)

    # Verify by computing all possible splits manually
    best_manual_gini = float("inf")
    for thresh in [1.0, 2.0, 3.0, 4.0, 5.0]:
        left_mask = X12[:, 0] <= thresh
        right_mask = ~left_mask
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        if n_left == 0 or n_right == 0:
            continue
        gini_left = compute_gini(y12[left_mask])
        gini_right = compute_gini(y12[right_mask])
        weighted = (n_left * gini_left + n_right * gini_right) / len(y12)
        if weighted < best_manual_gini:
            best_manual_gini = weighted

    _assert_close(gini12, best_manual_gini, atol=atol, msg="find_best_split did not find optimal split.")

    # --- Test 13: Single sample (edge case) ---
    X13 = np.array([[1.0]])
    y13 = np.array([0])
    feat13, thresh13, gini13 = find_best_split(X13, y13)

    if feat13 != -1:
        raise AssertionError(f"find_best_split should return -1 for single sample, got {feat13}")




