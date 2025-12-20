## Background

Linear probing is a canonical evaluation protocol for representation learning. Given features \(\mathbf{X} \in \mathbb{R}^{N \times D}\) from a frozen encoder (e.g., CLIP) and labels \(\mathbf{y} \in \{0, \ldots, C-1\}^N\), we train a linear classifier:

\[
\hat{\mathbf{y}} = \arg\max \left( \mathbf{X} \mathbf{W}^\top + \mathbf{b} \right)
\]

where \(\mathbf{W} \in \mathbb{R}^{C \times D}\) and \(\mathbf{b} \in \mathbb{R}^C\).

The classifier is trained by minimizing cross-entropy loss with SGD:

\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(z_{i, y_i})}{\sum_{c=1}^{C} \exp(z_{i,c})}
\]

where \(z_i = \mathbf{x}_i \mathbf{W}^\top + \mathbf{b}\).

---

## Approach

- Initialize a single `nn.Linear(feature_dim, num_classes)` layer.
- Store the learning rate for manual SGD updates.
- In `fit()`:
  - Compute logits via forward pass through the linear layer.
  - Compute cross-entropy loss using `nn.CrossEntropyLoss`.
  - Backpropagate to compute gradients.
  - Manually update parameters: `param -= lr * param.grad`.
  - Zero gradients after each update.
  - Track and return per-epoch losses.
- In `predict()`:
  - Forward pass to get logits.
  - Return `argmax` along the class dimension.

## Correctness

- Using `nn.Linear` with `nn.CrossEntropyLoss` matches the standard formulation.
- Manual SGD (without momentum) is equivalent to `torch.optim.SGD` with `momentum=0`.
- The `@torch.no_grad()` decorator in `predict()` avoids unnecessary gradient tracking.
- Zeroing gradients after each step prevents accumulation across epochs.

## Complexity

- **Time per epoch:** \(O(N \cdot D \cdot C)\) for the matrix multiply, plus \(O(N \cdot C)\) for softmax/loss.
- **Space:** \(O(D \cdot C)\) for the linear layer's weights and bias.

## Common Pitfalls

- Forgetting to zero gradients (causes gradient accumulation and divergence).
- Using `loss.backward()` inside `torch.no_grad()` context (no gradients computed).
- Not detaching loss with `.item()` when storing (keeps computation graph alive).
- Confusing `dim=1` vs `dim=-1` in `argmax` (both work for 2D, but be explicit).
- Forgetting `@torch.no_grad()` in predict (wastes memory on inference).




