Approach

- Compute the forward pass with matrix multiply and broadcast bias: \(y = xW + b\).
- Save `(x, W, b)` in a cache for the backward pass.
- For a scalar loss \(L\) with upstream gradient `dy = dL/dy`:
  - `dx = dy @ W.T` (chain rule through the matmul)
  - `dW = x.T @ dy`
  - `db = sum(dy, axis=0)` (bias contributes equally to each row)

Correctness

- Forward is correct by definition of an affine transform applied to each batch row.
- Backward:
  - Each output element \(y_{n,m} = \sum_d x_{n,d} W_{d,m} + b_m\).
  - Differentiating gives:
    - \(\partial L/\partial x_{n,d} = \sum_m (\partial L/\partial y_{n,m}) W_{d,m}\) which is `dy @ W.T`.
    - \(\partial L/\partial W_{d,m} = \sum_n x_{n,d} (\partial L/\partial y_{n,m})\) which is `x.T @ dy`.
    - \(\partial L/\partial b_m = \sum_n \partial L/\partial y_{n,m}\) which is `sum(dy, axis=0)`.

Complexity

- Time: \(O(NDM)\) for both forward and backward (dominated by matrix multiplications).
- Space: \(O(NM)\) for `y` plus \(O(ND + DM)\) for cached inputs.

Common Pitfalls

- Using the wrong transpose in `dx` (should be `W.T`).
- Summing `db` over the wrong axis (should be over batch axis `0`).
- Returning gradients with incorrect shapes (e.g., `db` as `(1, M)` instead of `(M,)`).


