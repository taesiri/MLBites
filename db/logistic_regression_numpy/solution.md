Approach
- Initialize parameters `w = zeros(d)` and `b = 0.0`.
- Repeat `num_steps` times (full-batch gradient descent):
  - Compute logits: `z = X @ w + b`
  - Compute probabilities: `p = sigmoid(z)`
  - Use the standard logistic regression gradient for average loss:
    - `grad_logits = (p - y) / n`
    - `grad_w = X.T @ grad_logits + l2 * w` (L2 only on `w`)
    - `grad_b = sum(grad_logits)`
  - Update: `w -= lr * grad_w`, `b -= lr * grad_b`

Correctness
- `p = sigmoid(X @ w + b)` matches the logistic model definition.
- For average binary cross-entropy, the derivative w.r.t. logits is `(p - y) / n`, so the chain rule yields:
  - `grad_w = X.T @ ((p - y) / n)`
  - `grad_b = sum((p - y) / n)`
- Adding L2 penalty \(\tfrac{1}{2}l2\|w\|^2\) contributes `l2*w` to `grad_w`, leaving `grad_b` unchanged.
- Each iteration applies a standard gradient descent step, so the returned `(w, b)` are exactly the parameters after `num_steps` updates.

Complexity
- Time: \(O(\text{num_steps} \cdot n \cdot d)\) due to the matrix-vector multiply and gradient computation each step.
- Space: \(O(d)\) for parameters (plus input storage).

Common Pitfalls
- Forgetting to average by `n` (using a sum changes the learning dynamics and expected outputs).
- Mixing up the sign (using `(y - p)` instead of `(p - y)`).
- Regularizing the bias `b` (this prompt regularizes `w` only).
- Returning `w` with shape `(d, 1)` instead of `(d,)`.


