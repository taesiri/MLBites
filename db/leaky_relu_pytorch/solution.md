# Solution: Leaky ReLU Activation Function

## Mathematical Formulation

The Leaky ReLU activation function is defined as:

\[
f(x) = 
\begin{cases}
x & \text{if } x > 0 \\
\alpha \cdot x & \text{if } x \leq 0
\end{cases}
\]

where \(\alpha\) is the `negative_slope` parameter (typically a small positive value like 0.01).

This can also be written as:

\[
f(x) = \max(0, x) + \alpha \cdot \min(0, x)
\]

or equivalently:

\[
f(x) = \max(\alpha \cdot x, x)
\]

## Approach
- Create a class that inherits from `nn.Module` and stores `negative_slope` in the constructor.
- In the `forward` method:
  - Use `torch.where()` to apply the conditional logic element-wise.
  - For elements where `x > 0`: return `x` unchanged.
  - For elements where `x <= 0`: return `negative_slope * x`.
- Alternatively, use `torch.maximum(negative_slope * x, x)` for a more concise implementation.

## Correctness
- The piecewise formula matches the standard Leaky ReLU definition.
- For positive inputs, the output equals the input (identity function).
- For non-positive inputs, the output is scaled by `negative_slope`, preserving the sign.
- The function is continuous everywhere (at x=0, both branches give 0).
- The function is differentiable everywhere except at x=0 (where the subgradient is used).
- Using `torch.where()` or `torch.maximum()` handles the conditional logic while maintaining gradient flow.
- Inheriting from `nn.Module` allows seamless integration with PyTorch's ecosystem.

## Complexity
- Time: \(O(n)\) where \(n\) is the number of elements in the input tensor. Each element is processed once.
- Space: \(O(n)\) for the output tensor.

## Common Pitfalls
- Forgetting to call `super().__init__()` in the constructor.
- Using `x >= 0` instead of `x > 0` for the positive case (both work, but convention varies).
- Applying `negative_slope` to positive values instead of negative ones.
- Using in-place operations that break autograd.
- Not storing `negative_slope` as an instance attribute.
- Forgetting that zero should be treated consistently (typically as non-positive).



