## Approach

- **Softmax stability**: Subtract `max(logits)` per row before computing `exp()`. This prevents overflow since the largest exponent becomes `exp(0) = 1`.
- The subtraction doesn't change the result: \(\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i}}{\sum_j e^{x_j}}\) for any constant \(c\).
- **Cross-entropy via log-sum-exp**: Instead of computing `softmax` then `log`, compute log-softmax directly.
- Log-softmax: \(\log(\text{softmax}(x))_i = x_i - \log\sum_j e^{x_j}\)
- **Log-sum-exp trick**: \(\log\sum_j e^{x_j} = m + \log\sum_j e^{x_j - m}\) where \(m = \max(x)\).
- Cross-entropy for class \(k\): \(-\log(\text{softmax}(x))_k = \log\sum_j e^{x_j} - x_k\)
- Use NumPy advanced indexing to select the correct class logit per sample.
- Return the mean loss over all samples.

## Math

The softmax function:

\[
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}}
\]

Numerically stable version (subtracting max):

\[
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j=1}^{C} e^{x_j - m}}, \quad m = \max_j(x_j)
\]

Cross-entropy loss for true class \(k\):

\[
\mathcal{L} = -\log(\text{softmax}(x)_k) = \log\sum_{j=1}^{C} e^{x_j} - x_k
\]

Log-sum-exp trick:

\[
\log\sum_{j=1}^{C} e^{x_j} = m + \log\sum_{j=1}^{C} e^{x_j - m}
\]

## Correctness

- Subtracting the max ensures all exponents are ≤ 0, preventing overflow.
- The log-sum-exp trick avoids taking log of very small numbers (which would cause underflow in log).
- Advanced indexing `logits[np.arange(n), targets]` correctly selects each sample's true class logit.

## Complexity

- **Time**: \(O(n \cdot C)\) where \(n\) is samples and \(C\) is classes — single pass over the data.
- **Space**: \(O(n \cdot C)\) for intermediate arrays (shifted logits, exp values).

## Common Pitfalls

- Forgetting to subtract max before `exp()` — causes overflow for large logits.
- Computing `log(softmax(x))` instead of log-softmax directly — causes underflow when probabilities are tiny.
- Using `axis=0` instead of `axis=1` — logits shape is `(n_samples, n_classes)`, reduction is over classes.
- Forgetting `keepdims=True` for proper broadcasting.
- Off-by-one errors in advanced indexing for selecting correct class logits.
