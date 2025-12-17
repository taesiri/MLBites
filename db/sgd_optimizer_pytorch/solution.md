## Mathematical Background

Stochastic Gradient Descent (SGD) is the foundational optimization algorithm in deep learning. The basic update rule moves parameters in the direction opposite to the gradient, scaled by a learning rate.

### Basic SGD Update

\[
\theta_{t+1} = \theta_t - \alpha \cdot g_t
\]

where:
- \( \theta_t \) is the parameter at step \( t \)
- \( \alpha \) is the learning rate
- \( g_t = \nabla_\theta \mathcal{L}(\theta_t) \) is the gradient

### Weight Decay (L2 Regularization)

Weight decay adds a penalty term proportional to the parameter magnitude, which discourages large weights:

\[
g_t \leftarrow g_t + \lambda \cdot \theta_t
\]

where \( \lambda \) is the weight decay coefficient. This is equivalent to adding \( \frac{\lambda}{2} \|\theta\|^2 \) to the loss function.

### Momentum

Momentum accelerates convergence by accumulating a velocity vector in the direction of persistent gradients:

\[
v_t = \mu \cdot v_{t-1} + g_t
\]

\[
\theta_{t+1} = \theta_t - \alpha \cdot v_t
\]

where:
- \( v_t \) is the velocity (momentum buffer)
- \( \mu \) is the momentum coefficient (typically 0.9)

Momentum helps:
- Dampen oscillations in high-curvature directions
- Accelerate movement along consistent gradient directions
- Escape shallow local minima

### Complete SGD Update (PyTorch Style)

Combining weight decay and momentum, the full update per parameter is:

\[
d_t = g_t + \lambda \cdot \theta_t
\]

\[
v_t = \mu \cdot v_{t-1} + d_t
\]

\[
\theta_{t+1} = \theta_t - \alpha \cdot v_t
\]

When \( \mu = 0 \), momentum is disabled and we use \( d_t \) directly instead of \( v_t \).

---

## Approach

- Store parameters and hyperparameters in `__init__`.
- If momentum is enabled, create a per-parameter buffer initialized to zeros.
- In `step()`, iterate over parameters:
  - Skip if gradient is `None`.
  - Apply weight decay to the gradient if enabled.
  - Update momentum buffer and use it as the effective gradient if momentum is enabled.
  - Update the parameter in-place.

## Correctness

- Weight decay matches the standard L2 regularization form used by `torch.optim.SGD`.
- Momentum buffers accumulate past gradients with exponential decay, matching SGD momentum behavior.
- Parameters with `grad is None` are skipped, consistent with PyTorch optimizers.

## Complexity

- **Time:** \( O\left(\sum_i |p_i|\right) \) per step.
- **Space:** \( O\left(\sum_i |p_i|\right) \) for momentum buffers when momentum is enabled.

## Common Pitfalls

- Forgetting to apply weight decay to the gradient before momentum.
- Updating momentum buffers with the wrong tensor (must use the current effective gradient).
- Performing updates without `torch.no_grad()` (accidentally tracking optimizer ops in the graph).
- Not skipping parameters with `grad is None`.
