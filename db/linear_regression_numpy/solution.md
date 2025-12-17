# Linear Regression Solution

## Approach
- **Fitting**: Use the closed-form normal equation to compute optimal weights.
- The normal equation directly minimizes the sum of squared errors: \( \|Xw - y\|^2 \).
- Taking the derivative and setting to zero gives: \( X^T X w = X^T y \).
- Solve this linear system using `np.linalg.solve` for numerical stability.
- **Prediction**: Simple matrix-vector multiplication \( \hat{y} = Xw \).
- Using `solve` instead of explicit matrix inverse avoids numerical issues.

## Math

The objective is to minimize the squared error:

\[
\mathcal{L}(w) = \|Xw - y\|^2 = (Xw - y)^T (Xw - y)
\]

Taking the gradient and setting to zero:

\[
\nabla_w \mathcal{L} = 2 X^T (Xw - y) = 0
\]

Solving for \( w \):

\[
X^T X w = X^T y
\]

\[
w = (X^T X)^{-1} X^T y
\]

Prediction is then:

\[
\hat{y} = X w
\]

## Correctness
- The normal equation gives the exact solution minimizing squared error.
- Using `np.linalg.solve` is equivalent to computing the inverse but more stable.
- Works correctly when \( X^T X \) is invertible (full column rank).
- Prediction is a direct matrix-vector product.

## Complexity
- **fit**:
  - Time: \( O(n \cdot d^2 + d^3) \) — matrix multiplication \( O(n \cdot d^2) \) plus solving \( O(d^3) \)
  - Space: \( O(d^2) \) for storing \( X^T X \)
- **predict**:
  - Time: \( O(n \cdot d) \) for matrix-vector multiplication
  - Space: \( O(n) \) for output predictions

Where \( n \) is the number of samples and \( d \) is the number of features.

## Common Pitfalls
- Using `np.linalg.inv` instead of `np.linalg.solve` (less stable numerically).
- Forgetting to transpose \( X \) when computing \( X^T X \).
- Shape mismatches when \( y \) is 2D instead of 1D.
- Not handling the case where \( X^T X \) is singular (though we assume full rank).
- Returning weights with wrong shape (e.g., 2D instead of 1D).
