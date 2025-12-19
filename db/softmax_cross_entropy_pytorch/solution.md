# Solution: Softmax, LogSoftmax, and CrossEntropy

## Mathematical Formulation

### Softmax
The softmax function converts logits \(\mathbf{x}\) into a probability distribution:

\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
\]

For numerical stability, we use the shift invariance property. For any constant \(c\):

\[
\text{softmax}(x_i) = \frac{e^{x_i - c}}{\sum_{j} e^{x_j - c}}
\]

By choosing \(c = \max(\mathbf{x})\), we prevent overflow since \(e^{x_i - \max(\mathbf{x})} \leq 1\).

### Log-Softmax
Log-softmax is the logarithm of softmax:

\[
\log\text{softmax}(x_i) = x_i - \log\left(\sum_{j} e^{x_j}\right)
\]

Using the log-sum-exp trick with \(c = \max(\mathbf{x})\):

\[
\log\left(\sum_{j} e^{x_j}\right) = c + \log\left(\sum_{j} e^{x_j - c}\right)
\]

Thus:

\[
\log\text{softmax}(x_i) = (x_i - c) - \log\left(\sum_{j} e^{x_j - c}\right)
\]

### Cross-Entropy Loss
For a batch of samples with logits \(\mathbf{X} \in \mathbb{R}^{N \times C}\) and targets \(\mathbf{y} \in \{0, \ldots, C-1\}^N\):

\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log\text{softmax}(X_i)_{y_i}
\]

## Approach

### Softmax
- Subtract the maximum value along the specified dimension (numerical stability).
- Compute `exp` of the shifted values.
- Normalize by dividing by the sum along the same dimension.

### Log-Softmax
- Subtract the maximum (log-sum-exp trick).
- Compute `log(sum(exp(shifted)))`.
- Return `shifted - log_sum_exp`.
- More numerically stable than `log(softmax(x))`.

### Cross-Entropy
- Compute log-softmax of the logits.
- Use advanced indexing to select the log-probability of the correct class for each sample.
- Return the negative mean of these log-probabilities.

## Correctness
- **Numerical stability**: The max-subtraction trick prevents overflow when logits are large (e.g., 1000+) and underflow when they differ significantly.
- **Softmax invariance**: Subtracting a constant doesn't change the softmax output since it cancels in the numerator and denominator.
- **Log-softmax directly**: Computing log-softmax directly (not as `log(softmax(x))`) avoids taking log of very small numbers which would cause underflow.
- **Cross-entropy**: Using log-softmax internally is more stable than computing softmax probabilities and then taking their log.

## Complexity
- **Time**: \(O(N \times C)\) for all three functions, where \(N\) is batch size and \(C\) is number of classes. Each operation (max, exp, sum, log) is linear in the number of elements.
- **Space**: \(O(N \times C)\) for intermediate tensors (shifted values, exp values).

## Common Pitfalls
- **Forgetting numerical stability**: Computing `exp(x)` directly without subtracting max causes overflow for large x (e.g., `exp(1000)` = inf).
- **Using `log(softmax(x))`**: This is numerically unstable; use log-sum-exp trick instead for log-softmax.
- **Wrong dimension for max/sum**: Forgetting `keepdim=True` causes broadcasting issues.
- **Off-by-one in indexing**: Using `log_probs[:, targets]` instead of `log_probs[torch.arange(n), targets]` gives wrong shape.
- **Forgetting the negative sign**: Cross-entropy is the **negative** log-likelihood.
- **Not using `.values` on max result**: `torch.max()` returns a named tuple; need `.values` for the actual tensor.


