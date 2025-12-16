Approach
- Keep per-parameter state buffers:
  - `m` (first moment / momentum)
  - `v` (second moment)
- Maintain a step counter `t` starting at 0 and increment once per `step()`.
- For each parameter with gradient:
  - Apply **decoupled weight decay**: `p *= (1 - lr * weight_decay)`.
  - Update moments:
    - `m = beta1*m + (1-beta1)*g`
    - `v = beta2*v + (1-beta2)*g*g`
  - Bias correction (PyTorch ordering):
    - `step_size = lr / (1 - beta1**t)`
    - `denom = sqrt(v) / sqrt(1 - beta2**t) + eps`
  - Update: `p -= step_size * m / denom`.

Correctness
- Weight decay is decoupled: it modifies parameters directly and does not change the gradient used for Adam’s moment estimates.
- Moment updates match Adam’s EMA definitions, and bias correction matches PyTorch’s scaling.
- Parameters with `grad is None` are skipped, consistent with PyTorch optimizer behavior.

Complexity
- Time: \(O(\sum_i |p_i|)\) per step.
- Space: \(O(\sum_i |p_i|)\) for storing `m` and `v`.

Common Pitfalls
- Implementing classic L2 regularization (adding `weight_decay * p` to the gradient) instead of decoupled decay.
- Forgetting bias correction or applying it in a different order than PyTorch (leading to tiny numeric mismatches).
- Applying weight decay to parameters that have `grad=None` (PyTorch skips those).


