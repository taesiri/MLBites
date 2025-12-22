## Mathematical Background

AdamW (Adam with Weight Decay) extends Adam by decoupling weight decay from the gradient update. The key difference is that weight decay is applied directly to the parameters rather than being added to the gradient.

### Algorithm

**Initialize:**
- First moment estimate: \( m_0 = 0 \)
- Second moment estimate: \( v_0 = 0 \)
- Timestep: \( t = 0 \)

**At each step \( t \):**

1. **Increment timestep:**
\[
t \leftarrow t + 1
\]

2. **Compute gradient:**
\[
g_t = \nabla_\theta f(\theta_{t-1})
\]

3. **Update biased first moment estimate (momentum):**
\[
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
\]

4. **Update biased second moment estimate (squared gradient):**
\[
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
\]

5. **Compute bias-corrected estimates:**
\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

6. **Update parameters (AdamW style):**
\[
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \cdot \lambda \cdot \theta_{t-1}
\]

where \( \lambda \) is the weight decay coefficient. The last term applies weight decay directly to the parameters, decoupled from the gradient-based update.

### PyTorch-Style Formulation

PyTorch fuses the bias correction into the step size for efficiency:

\[
\eta_t = \frac{\alpha \cdot \sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
\]

\[
\theta_t = \theta_{t-1} - \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \alpha \cdot \lambda \cdot \theta_{t-1}
\]

### Default Hyperparameters

- Learning rate: \( \alpha = 0.001 \)
- First moment decay: \( \beta_1 = 0.9 \)
- Second moment decay: \( \beta_2 = 0.999 \)
- Numerical stability: \( \epsilon = 10^{-8} \)
- Weight decay: \( \lambda = 0.01 \)

---

## Approach

- Keep per-parameter state: first moment `m` and second moment `v`, initialized to zeros.
- Maintain a step counter `t` starting at 0, incrementing once per `step()`.
- On each `step()`, for every parameter `p` with gradient `g = p.grad`:
  - Update moments: `m = β₁·m + (1-β₁)·g`, `v = β₂·v + (1-β₂)·g²`.
  - Apply PyTorch-style bias correction via the step size.
  - Update in-place: `p -= step_size · m / (√v + ε)` (Adam update).
  - Apply decoupled weight decay: `p -= lr · weight_decay · p`.

## Correctness

- The moment updates match Adam's definitions for exponential moving averages.
- Bias correction accounts for the fact that `m` and `v` start at zero, matching `torch.optim.AdamW`'s scaling.
- Weight decay is applied directly to parameters (decoupled), not added to gradients.
- Parameters with `grad is None` are skipped, consistent with typical PyTorch optimizer behavior.

## Complexity

- **Time:** \( O\left(\sum_i |p_i|\right) \) per step (touch each parameter element a constant number of times).
- **Space:** \( O\left(\sum_i |p_i|\right) \) for storing `m` and `v`.

## Common Pitfalls

- Forgetting bias correction (or applying it in a way that doesn't match PyTorch's update).
- Not using `torch.no_grad()` (accidentally building a graph during parameter updates).
- Applying weight decay to gradients instead of directly to parameters (this would be Adam, not AdamW).
- Failing to skip `grad=None` parameters.
- Mixing up `β₁` and `β₂`, or updating `v` with `g` instead of `g²`.
- Forgetting to apply weight decay even when gradient is zero.
