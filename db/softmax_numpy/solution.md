## Approach
- Compute the maximum value along the target `axis` and subtract it from `x` to improve numerical stability.
- Exponentiate the shifted values.
- Normalize by dividing by the sum of exponentials along the same `axis` (use `keepdims=True` for broadcasting).

## Correctness
- Subtracting `max(x)` along `axis` does not change the output because for any constant \(c\),
  \(\text{softmax}(x) = \text{softmax}(x - c)\) when \(c\) is applied uniformly along the normalization axis.
- After exponentiation, dividing by the sum along `axis` ensures the outputs are non-negative and sum to `1` along `axis`.
- Using `keepdims=True` guarantees shapes broadcast correctly for any input dimensionality.

## Complexity
- Time: \(O(n)\) over the number of elements in `x` (one max, one exp, one sum, one divide).
- Space: \(O(n)\) for intermediate arrays (can be reduced with in-place patterns, but not necessary here).

## Common Pitfalls
- Forgetting the max-subtraction step (overflow for large logits like `1000`).
- Not using `keepdims=True`, causing broadcasting or shape bugs for multi-dimensional inputs.
- Summing over the wrong axis (or always using the last axis regardless of `axis` argument).


