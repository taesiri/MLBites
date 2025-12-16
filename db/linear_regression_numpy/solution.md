Approach
- If `fit_intercept=True`, build a design matrix by appending a column of ones to `X`.
- Solve the least-squares problem using `np.linalg.lstsq` to get parameters.
- Split parameters into `coef_` (feature weights) and `intercept_` (last element) when using an intercept.
- Predict with the learned linear function: \(\hat{y} = Xw + b\).

Correctness
- `np.linalg.lstsq` returns parameters that minimize \(\|X\beta - y\|_2^2\), which is exactly ordinary least squares.
- When using an intercept, augmenting `X` with a ones column is equivalent to learning \(b\) as an additional parameter.
- `predict` uses the same learned parameters, so outputs match the fitted linear model.

Complexity
- Time: \(O(n d^2)\) for the least-squares solve in typical dense settings (depends on the algorithm used internally).
- Space: \(O(nd)\) to hold the design matrix (or \(O(nd)\) for `X` if `fit_intercept=False`).

Common Pitfalls
- Explicitly computing \((X^T X)^{-1}\) (numerically worse than using `lstsq`/`solve`).
- Forgetting to add the ones column when `fit_intercept=True`.
- Mixing up shapes (e.g., `y` should be shape `(n_samples,)`).
- Returning predictions with shape `(n_samples, 1)` instead of `(n_samples,)`.


