## Approach

- **Normalize over the last dimension**: Unlike BatchNorm which normalizes across samples, LayerNorm normalizes each sample independently over its feature dimension.
- **Compute mean and variance**: For each sample, compute the mean and variance of the features along the last dimension.
- **Numerical stability**: Add epsilon to variance before taking the square root.
- **Learnable parameters**: Apply a learnable scale (weight) and shift (bias) after normalization.
- **Parameter initialization**: Weight is initialized to ones, bias to zeros (identity transform initially).
- **Broadcasting**: The normalization works for any input shape as long as the last dimension matches `normalized_shape`.

## Math

Layer Normalization transforms input \(x\) where we normalize over the last dimension:

1. Compute mean over the last dimension:
\[
\mu = \frac{1}{D} \sum_{i=1}^{D} x_i
\]

2. Compute variance over the last dimension:
\[
\sigma^2 = \frac{1}{D} \sum_{i=1}^{D} (x_i - \mu)^2
\]

3. Normalize:
\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]

4. Scale and shift with learnable parameters:
\[
y_i = \gamma \cdot \hat{x}_i + \beta
\]

where \(\gamma\) (weight) and \(\beta\) (bias) are learnable parameters of shape \((D,)\).

## Correctness

- Each sample (along the last dimension) is normalized to have approximately zero mean and unit variance.
- The epsilon prevents division by zero when variance is very small.
- Learnable parameters allow the network to undo the normalization if needed.
- Works with any number of leading dimensions due to proper use of `keepdim=True`.

## Complexity

- **Time**: \(O(N)\) where N is the total number of elements — each element is touched a constant number of times.
- **Space**: \(O(D)\) for the learnable parameters (weight and bias).

## Common Pitfalls

- Forgetting `keepdim=True` when computing mean/variance, which breaks broadcasting.
- Using the wrong dimension for normalization (should be the last dimension, i.e., `dim=-1`).
- Not initializing weight to ones and bias to zeros.
- Using biased vs unbiased variance — LayerNorm uses the population variance (biased, N divisor).
- Forgetting to register weight and bias as `nn.Parameter` so they're trainable.

