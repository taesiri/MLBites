## Approach

- **Forward pass**: Compute batch mean and variance along the batch dimension (axis=0), normalize inputs, then apply learned scale (gamma) and shift (beta).
- **Numerical stability**: Add a small epsilon to variance before taking the square root to avoid division by zero.
- **Cache intermediate values**: Store everything needed for the backward pass (x, normalized x, mean, variance, std, gamma).
- **Backward pass**: Apply the chain rule carefully. The tricky part is that mean and variance depend on all inputs in the batch.
- **Gradient for beta**: Simply sum dout over the batch dimension.
- **Gradient for gamma**: Sum of (dout × normalized x) over the batch.
- **Gradient for x**: Requires propagating gradients through normalization, variance, and mean computations.

## Math

### Forward Pass

Batch normalization transforms input \(x\) of shape \((N, D)\):

1. Compute batch mean:
\[
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
\]

2. Compute batch variance:
\[
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
\]

3. Normalize:
\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]

4. Scale and shift:
\[
y_i = \gamma \cdot \hat{x}_i + \beta
\]

### Backward Pass

Given upstream gradient \(\frac{\partial L}{\partial y}\):

1. Gradient w.r.t. beta:
\[
\frac{\partial L}{\partial \beta} = \sum_{i=1}^{N} \frac{\partial L}{\partial y_i}
\]

2. Gradient w.r.t. gamma:
\[
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{N} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i
\]

3. Gradient w.r.t. \(\hat{x}\):
\[
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma
\]

4. Gradient w.r.t. variance:
\[
\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^{N} \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu) \cdot \left(-\frac{1}{2}\right) (\sigma^2 + \epsilon)^{-3/2}
\]

5. Gradient w.r.t. mean:
\[
\frac{\partial L}{\partial \mu} = \sum_{i=1}^{N} \frac{\partial L}{\partial \hat{x}_i} \cdot \left(-\frac{1}{\sqrt{\sigma^2 + \epsilon}}\right)
\]

6. Gradient w.r.t. input:
\[
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{N} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{N}
\]

## Correctness

- Forward pass correctly normalizes each feature to have approximately zero mean and unit variance within the batch.
- The epsilon prevents division by zero when variance is very small.
- The backward pass correctly accounts for all dependencies: x affects the output directly via normalization, and indirectly via mean and variance.
- Gradients for gamma and beta are straightforward sums since they appear linearly in the output.

## Complexity

- **Time**: \(O(N \cdot D)\) for both forward and backward passes — linear in the size of the input.
- **Space**: \(O(N \cdot D)\) for caching intermediate values (x, x_norm, and gradients).

## Common Pitfalls

- Forgetting that mean and variance depend on all inputs, leading to incorrect dx.
- Using `axis=1` instead of `axis=0` for batch statistics (we reduce over the batch dimension).
- Not caching enough values for the backward pass (need x, x_norm, std, gamma).
- Incorrect power in variance gradient: it's \((\sigma^2 + \epsilon)^{-3/2}\), not \(-1/2\).
- Forgetting the \(2/N\) factor in the variance gradient or \(1/N\) factor in the mean gradient.
- Off-by-one errors in the variance formula (using \(N-1\) instead of \(N\) — we use population variance here).



