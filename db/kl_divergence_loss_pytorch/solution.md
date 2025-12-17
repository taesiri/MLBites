# Solution: KL Divergence Loss

## Mathematical Formulation

The Kullback-Leibler divergence from distribution \( Q \) to distribution \( P \) is defined as:

\[
D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \sum_{x} P(x) \left( \log P(x) - \log Q(x) \right)
\]

Following PyTorch's convention where `input` contains \(\log Q\) and `target` contains \(P\):

\[
D_{KL} = \sum_{i} \text{target}_i \cdot \left( \log(\text{target}_i) - \text{input}_i \right)
\]

## Approach
- Create a class that inherits from `nn.Module` and stores the `reduction` mode in the constructor.
- In the `forward` method:
  - Compute element-wise KL divergence: `target * (log(target) - input)`.
  - Use `torch.xlogy(target, target)` to safely compute `target * log(target)`, which correctly handles the case where `target = 0` (returning 0 instead of NaN).
  - Apply the appropriate reduction based on `self.reduction`:
    - `"none"`: return element-wise values
    - `"sum"`: sum all elements
    - `"mean"`: mean of all elements
    - `"batchmean"`: sum divided by batch size (recommended for proper KL divergence)

## Correctness
- The formula matches the standard KL divergence definition.
- Using `torch.xlogy(a, b)` computes `a * log(b)` with the convention that `0 * log(0) = 0`, avoiding NaN values when target probabilities are zero.
- The implementation matches PyTorch's `nn.KLDivLoss` behavior exactly.
- `"batchmean"` is the mathematically correct reduction for KL divergence when dealing with probability distributions over batches, as it gives the average KL divergence per sample.

## Complexity
- Time: \(O(n)\) where \(n\) is the number of elements in the input tensors.
- Space: \(O(n)\) for the intermediate element-wise KL divergence tensor (when reduction is not `"none"`).

## Common Pitfalls
- Confusing the input convention: PyTorch expects `input` to be **log-probabilities** and `target` to be **probabilities**.
- Not handling the `0 * log(0) = 0` case properly, leading to NaN values.
- Using `torch.log(target)` directly instead of `torch.xlogy(target, target)` when target can contain zeros.
- Confusing `"mean"` and `"batchmean"`: `"batchmean"` divides by batch size, while `"mean"` divides by total number of elements.
- Forgetting to call `super().__init__()` in the constructor.

