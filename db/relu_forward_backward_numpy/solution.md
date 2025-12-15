Approach

- Forward: compute `y = np.maximum(0, x)` elementwise.
- Cache a boolean mask `mask = (x > 0)` (we define the derivative at `x == 0` as 0).
- Backward: `dx = dy * mask`.

Correctness

- For each element \(x_i\), ReLU returns \(0\) when \(x_i \le 0\) and \(x_i\) when \(x_i > 0\), matching `np.maximum(0, x)`.
- The gradient is \(0\) for \(x_i \le 0\) and \(1\) for \(x_i > 0\); multiplying `dy` by the cached mask applies the chain rule elementwise.

Complexity

- Time: \(O(\text{numel}(x))\)
- Space: \(O(\text{numel}(x))\) for the output and cached mask

Common Pitfalls

- Using `x >= 0` for the mask; this makes the derivative at 0 equal to 1 (this question defines it as 0).
- Returning a mask of the wrong shape (must match `x`/`dy`).
- Forgetting to return a cache from forward, making backward impossible.


