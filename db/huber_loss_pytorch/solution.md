# Solution: Huber Loss

## Mathematical Formulation

The Huber loss for a single residual \( r = \hat{y} - y \) (prediction minus target) is defined as:

\[
L_\delta(r) = 
\begin{cases}
\frac{1}{2} r^2 & \text{if } |r| \leq \delta \\
\delta \left( |r| - \frac{1}{2} \delta \right) & \text{if } |r| > \delta
\end{cases}
\]

For a batch of predictions \(\hat{\mathbf{y}}\) and targets \(\mathbf{y}\), the mean Huber loss is:

\[
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} L_\delta(\hat{y}_i - y_i)
\]

where:
- \(\delta\) is the threshold parameter that controls the transition between quadratic and linear regions
- \(r_i = \hat{y}_i - y_i\) is the residual for the \(i\)-th element
- \(n\) is the total number of elements

## Approach
- Create a class that inherits from `nn.Module` and stores `delta` in the constructor.
- In the `forward` method:
  - Compute the residual (difference) between predictions and targets: `residual = predictions - targets`.
  - Compute the absolute residual: `abs_residual = |residual|`.
  - For elements where `abs_residual <= delta`: use the quadratic formula `0.5 * residual^2`.
  - For elements where `abs_residual > delta`: use the linear formula `delta * (abs_residual - 0.5 * delta)`.
  - Use `torch.where()` to select between the two cases element-wise.
  - Return the mean of all element-wise losses.

## Correctness
- The piecewise formula matches the standard Huber loss definition.
- For small errors (within delta), the quadratic term provides smooth gradients.
- For large errors (beyond delta), the linear term reduces sensitivity to outliers.
- The function is continuous and differentiable at `abs_residual = delta` since both branches evaluate to `0.5 * delta^2` at that point.
- Using `torch.where()` handles the conditional logic correctly while maintaining gradient flow.
- Inheriting from `nn.Module` allows the loss to integrate seamlessly with PyTorch's ecosystem (e.g., model composition, device placement).

## Complexity
- Time: \(O(n)\) where \(n\) is the number of elements in the input tensors. Each element is processed once.
- Space: \(O(n)\) for intermediate tensors (residuals, abs_residuals, element-wise losses).

## Common Pitfalls
- Forgetting to call `super().__init__()` in the constructor.
- Forgetting the `0.5` coefficient in the quadratic term.
- Using `residual^2` without the `0.5` factor in the linear term offset (`0.5 * delta`).
- Confusing the condition: it should be `abs_residual <= delta` for quadratic, not `abs_residual < delta`.
- Applying reduction (mean) before computing the piecewise function instead of after.
- Using in-place operations that break autograd.
- Not using `self.delta` inside `forward` (using a hardcoded value instead).
