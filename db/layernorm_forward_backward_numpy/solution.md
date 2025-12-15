Approach

- For each row of `x` (shape `(N, D)`), compute `mu = mean(x, axis=1)` and `var = var(x, axis=1)`.
- Compute `inv_std = 1 / sqrt(var + eps)` and `x_hat = (x - mu) * inv_std`.
- Apply affine transform: `y = x_hat * gamma + beta` (broadcast `gamma`, `beta` to `(1, D)`).
- Cache `(x_hat, inv_std, gamma)` for backward.
- Backward:
  - `dbeta = sum(dy, axis=0)`, `dgamma = sum(dy * x_hat, axis=0)`.
  - Let `dxhat = dy * gamma`.
  - For each row, use the simplified LayerNorm gradient:
    `dx = (inv_std / D) * (D*dxhat - sum(dxhat) - x_hat*sum(dxhat*x_hat))`.

Correctness

- `x_hat` has per-row mean 0 and variance 1 (up to `eps`), so normalization is correct.
- The affine step matches the definition of LayerNorm with learnable scale/shift.
- The backward formula is the standard closed-form derivative for normalization over features and matches finite-difference gradients.

Complexity

- Time: \(O(ND)\)
- Space: \(O(ND)\) for cached `x_hat` (plus \(O(N)\) for `inv_std`)

Common Pitfalls

- Normalizing over the wrong axis (LayerNorm is over features `D`, per sample).
- Forgetting to broadcast/reshape `gamma` and `beta` to `(1, D)`.
- Using `np.std` without keeping dimensions, causing shape bugs.
- Returning gradients with inconsistent shapes (accept `(D,)` or `(1, D)` but be consistent).

## Approach
- Compute per-row mean and variance across the feature dimension \(D\).
- Normalize: \(xhat = (x - \mu) / \sqrt{\sigma^2 + \epsilon}\).
- Scale/shift: \(y = xhat \cdot \gamma + \beta\) (broadcast \(\gamma,\beta\) over rows).
- For backward:
  - `dbeta = sum(dy, axis=0)`
  - `dgamma = sum(dy * xhat, axis=0)`
  - Propagate to `dx` by differentiating the normalization steps per row (vectorized).
- Optional verification: compare against the deterministic examples in `question.md`, or do a small finite-difference gradient check locally.

## Correctness
- Forward matches the LayerNorm definition with per-example statistics (axis `1` for `(N, D)`).
- Backward applies the chain rule through: scale/shift → normalization → mean/variance,
  producing gradients for `x`, `gamma`, and `beta` with the correct shapes.

## Complexity
- **Time**: \(O(ND)\)
- **Space**: \(O(ND)\) for cached intermediates

## Common Pitfalls
- Normalizing across the wrong axis (LayerNorm is per row here).
- Forgetting `keepdims=True`, causing broadcasting bugs.
- Returning `dgamma/dbeta` with shape `(1, D)` instead of `(D,)`.
- Mixing sample variance vs population variance (be consistent with `np.var` default).


