Approach

- Treat linear regression as a least-squares problem: predict \(y \approx Xw + b\).
- If `fit_intercept=True`, append a column of ones to `X` so the intercept becomes an extra parameter.
- Solve for parameters using:
  - ridge (L2) normal equations when `l2_reg > 0`: \((X^T X + \lambda I)\theta = X^T y\)
  - `np.linalg.lstsq` when `l2_reg == 0` to handle rank-deficient inputs robustly.
- Do not regularize the intercept: the last diagonal entry of the ridge matrix is 0.

Correctness

- The returned `(w, b)` minimize the objective \(||Xw + b - y||^2 + \lambda ||w||^2\).
- Augmenting with ones makes the intercept just another parameter, while setting its ridge penalty to 0 ensures only `w` is regularized.
- Using `lstsq` for the unregularized case returns a valid least-squares solution even when \(X\) is not full rank.

Complexity

- Let \(N\) be the number of samples and \(D\) the number of features.
- Time:
  - forming \(X^T X\): \(O(ND^2)\)
  - solving the linear system: \(O(D^3)\)
- Space: \(O(D^2)\) for the normal-equation matrix.

Common Pitfalls

- Regularizing the intercept (it should not be penalized).
- Forgetting to handle `y` shaped `(N, 1)` vs `(N,)`.
- Using `np.linalg.solve` on a singular matrix when `l2_reg=0` (use `lstsq` instead).
- Returning `w` with shape `(D,1)` instead of `(D,)`.


