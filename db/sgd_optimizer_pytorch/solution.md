## Mathematical Background

Stochastic Gradient Descent (SGD) is the foundational optimization algorithm in deep learning. The basic update rule moves parameters in the direction opposite to the gradient, scaled by a learning rate.

### Vanilla SGD

\[
\theta_{t+1} = \theta_t - \alpha \cdot g_t
\]

where:
- \( \theta_t \) is the parameter at step \( t \)
- \( \alpha \) is the learning rate
- \( g_t = \nabla_\theta \mathcal{L}(\theta_t) \) is the gradient

### Weight Decay (L2 Regularization)

Weight decay adds a penalty term proportional to the parameter magnitude, discouraging large weights:

\[
d_t = g_t + \lambda \cdot \theta_t
\]

where \( \lambda \) is the weight decay coefficient. This is equivalent to adding \( \frac{\lambda}{2} \|\theta\|^2 \) to the loss function. The update becomes:

\[
\theta_{t+1} = \theta_t - \alpha \cdot d_t
\]

Expanding:

\[
\theta_{t+1} = \theta_t - \alpha \cdot g_t - \alpha \lambda \cdot \theta_t = (1 - \alpha \lambda) \theta_t - \alpha \cdot g_t
\]

This shows that weight decay shrinks the parameters toward zero at each step.

---

## Approach

- Store parameters and hyperparameters in `__init__`.
- In `step()`, iterate over parameters:
  - Skip if gradient is `None`.
  - Apply weight decay to the gradient if enabled.
  - Update the parameter in-place: \( p \leftarrow p - \alpha \cdot d \).

## Correctness

- Weight decay matches the standard L2 regularization form used by `torch.optim.SGD`.
- Parameters with `grad is None` are skipped, consistent with PyTorch optimizers.

## Complexity

- **Time:** \( O\left(\sum_i |p_i|\right) \) per step.
- **Space:** \( O(1) \) additional space (updates are in-place).

## Common Pitfalls

- Forgetting to apply weight decay to the gradient when enabled.
- Performing updates without `torch.no_grad()` (accidentally tracking optimizer ops in the graph).
- Not skipping parameters with `grad is None`.
