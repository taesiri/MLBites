## Mathematical Background

AdamW is a variant of Adam that fixes weight decay regularization. In the original Adam paper, weight decay was implemented as L2 regularization (adding \(\lambda \theta\) to the gradient), but this interacts poorly with adaptive learning rates. AdamW instead applies **decoupled weight decay** directly to the parameters.

### Adam with L2 Regularization (Incorrect)

The standard L2 approach modifies the gradient:
\[
g_t \leftarrow g_t + \lambda \theta_{t-1}
\]

This is problematic because the adaptive scaling in Adam reduces the effect of weight decay for parameters with large gradients.

### AdamW Algorithm (Decoupled Weight Decay)

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

3. **Apply decoupled weight decay:**
\[
\theta_{t-1} \leftarrow \theta_{t-1} \cdot (1 - \alpha \lambda)
\]

4. **Update biased first moment estimate:**
\[
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
\]

5. **Update biased second moment estimate:**
\[
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
\]

6. **Compute bias-corrected estimates:**
\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

7. **Update parameters:**
\[
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

### PyTorch-Style Formulation

PyTorch applies weight decay first, then computes the Adam update with fused bias correction:

\[
\theta \leftarrow \theta \cdot (1 - \alpha \lambda)
\]

\[
\eta_t = \frac{\alpha}{1 - \beta_1^t}
\]

\[
d_t = \frac{\sqrt{v_t}}{\sqrt{1 - \beta_2^t}} + \epsilon
\]

\[
\theta_t = \theta_{t-1} - \eta_t \cdot \frac{m_t}{d_t}
\]

### Default Hyperparameters

- Learning rate: \( \alpha = 0.001 \)
- First moment decay: \( \beta_1 = 0.9 \)
- Second moment decay: \( \beta_2 = 0.999 \)
- Numerical stability: \( \epsilon = 10^{-8} \)
- Weight decay: \( \lambda = 0.01 \)

---

## Approach

- Keep per-parameter state buffers: `m` (first moment) and `v` (second moment).
- Maintain a step counter `t` starting at 0, incrementing once per `step()`.
- For each parameter with gradient:
  - Apply **decoupled weight decay**: `p *= (1 - α·λ)`.
  - Update moments: `m = β₁·m + (1-β₁)·g`, `v = β₂·v + (1-β₂)·g²`.
  - Bias correction (PyTorch ordering).
  - Update: `p -= step_size · m / denom`.

## Correctness

- Weight decay is decoupled: it modifies parameters directly and does not change the gradient used for Adam's moment estimates.
- Moment updates match Adam's EMA definitions, and bias correction matches PyTorch's scaling.
- Parameters with `grad is None` are skipped, consistent with PyTorch optimizer behavior.

## Complexity

- **Time:** \( O\left(\sum_i |p_i|\right) \) per step.
- **Space:** \( O\left(\sum_i |p_i|\right) \) for storing `m` and `v`.

## Common Pitfalls

- Implementing classic L2 regularization (adding `λ·p` to the gradient) instead of decoupled decay.
- Forgetting bias correction or applying it in a different order than PyTorch.
- Applying weight decay to parameters that have `grad=None` (PyTorch skips those).
- Confusing AdamW's `weight_decay` parameter with Adam's—they have different semantics.
