## Approach

- **Sigmoid function**: The core of logistic regression — maps any real number to (0, 1) range, interpreted as probability.
- **Model**: Predict probability as \( p = \sigma(X w + b) \) where \(\sigma\) is the sigmoid function.
- **Loss**: Binary cross-entropy measures how well predicted probabilities match true labels.
- **Gradient derivation**: The gradient of BCE w.r.t. logits \(z\) simplifies to \(p - y\), making the update rule elegant.
- **Training**: Standard gradient descent — iteratively update \(w\) and \(b\) in the negative gradient direction.
- **Numerical stability**: Clip probabilities before taking log to avoid `log(0) = -inf`.

## Math

The sigmoid function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Model prediction:

\[
p = \sigma(X w + b)
\]

Binary cross-entropy loss:

\[
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
\]

Gradients (using the fact that \(\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i\)):

\[
\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T (p - y)
\]

\[
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)
\]

Gradient descent update:

\[
w \leftarrow w - \alpha \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}
\]

## Correctness

- The sigmoid maps logits to valid probabilities in (0, 1).
- BCE loss is the standard loss for binary classification, derived from maximum likelihood.
- The gradient derivation \((p - y)\) is well-known and exact for sigmoid + BCE combination.
- Clipping predictions prevents numerical issues without affecting optimization significantly.
- Zero initialization is a valid starting point for logistic regression (unlike deep networks).

## Complexity

- **Time**: \(O(n \cdot d)\) per iteration where \(n\) is samples and \(d\) is features.
  Total: \(O(T \cdot n \cdot d)\) for \(T\) iterations.
- **Space**: \(O(n + d)\) for intermediate arrays (predictions, gradients).

## Common Pitfalls

- Forgetting to clip probabilities before `log()` — causes `nan` or `-inf`.
- Using `X @ w` when `w` is uninitialized or wrong shape — leads to broadcasting errors.
- Dividing by `n` inconsistently between loss and gradient — causes learning rate mismatch.
- Returning `db` as an array instead of a scalar.
- Using `axis=0` incorrectly when computing `X.T @ error` — transpose handles the reduction.
- Forgetting to convert final bias to float when returning.
