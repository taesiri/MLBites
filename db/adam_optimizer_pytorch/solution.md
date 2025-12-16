Approach
- Keep per-parameter state: first moment `m` and second moment `v`, initialized to zeros.
- Maintain a step counter `t` starting at 0, incrementing once per `step()`.
- On each `step()`, for every parameter `p` with gradient `g = p.grad`:
  - Update moments: `m = beta1*m + (1-beta1)*g`, `v = beta2*v + (1-beta2)*g*g`.
  - Apply PyTorch-style bias correction via the step size:
    - `bias_correction1 = 1 - beta1**t`
    - `bias_correction2 = 1 - beta2**t`
    - `step_size = lr * sqrt(bias_correction2) / bias_correction1`
  - Update in-place: `p -= step_size * m / (sqrt(v) + eps)`.

Correctness
- The moment updates match Adam’s definitions for exponential moving averages.
- Bias correction accounts for the fact that `m` and `v` start at zero, matching `torch.optim.Adam`’s scaling.
- Parameters with `grad is None` are skipped, consistent with typical PyTorch optimizer behavior.

Complexity
- Time: \(O(\sum_i |p_i|)\) per step (touch each parameter element a constant number of times).
- Space: \(O(\sum_i |p_i|)\) for storing `m` and `v`.

Common Pitfalls
- Forgetting bias correction (or applying it in a way that doesn’t match PyTorch’s update).
- Not using `torch.no_grad()` (accidentally building a graph during parameter updates).
- Failing to skip `grad=None` parameters.
- Mixing up `beta1` and `beta2`, or updating `v` with `g` instead of `g*g`.


