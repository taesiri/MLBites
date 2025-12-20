# Train GPT-2 on Dummy Data with Adam

## Approach

1. **Set seed** for reproducibility using `torch.manual_seed(seed)`.
2. **Build GPT-2 model** with standard components:
   - Token embeddings map discrete tokens to dense vectors
   - Positional embeddings encode position information (learned, not sinusoidal)
   - Stack N transformer blocks, each with pre-norm, causal attention, and MLP
   - Final layer norm followed by linear projection to vocabulary size
3. **Causal self-attention** uses a lower-triangular mask to prevent attending to future tokens.
4. **MLP** expands to 4x hidden dimension with GELU activation, then projects back.
5. **Create Adam optimizer** with the specified learning rate.
6. **Training loop**:
   - Generate random dummy tokens of shape `(batch_size, block_size + 1)`
   - Split into input `[:, :-1]` and target `[:, 1:]` (next-token prediction)
   - Compute cross-entropy loss averaged over all positions
   - Backpropagate and update parameters
   - Record loss for each step

## Correctness

- **Determinism**: Setting `torch.manual_seed` before model initialization and data generation ensures identical weights and data across runs with the same seed.
- **Causal masking**: The attention mask prevents position i from attending to positions j > i, ensuring autoregressive behavior.
- **Next-token prediction**: Input tokens at positions 0..T-1 predict tokens at positions 1..T, which is the standard language modeling objective.
- **Loss calculation**: Using `F.cross_entropy` with flattened logits and targets computes the average cross-entropy over all positions.

## Complexity

**Time per step:**
- Forward pass: O(B × T × d²) for embeddings/projections, O(B × T² × d) for attention
- Backward pass: Same order as forward
- Total per step: O(B × T² × d + B × T × d²) where B=batch_size, T=block_size, d=embed_dim

**Space:**
- Model parameters: O(vocab_size × d + T × d + L × d²) where L=num_layers
- Activations during training: O(B × T × d × L)

## Common Pitfalls

1. **Forgetting to set seed before model creation** — weights will be non-deterministic.
2. **Off-by-one errors in input/target split** — must use `[:, :-1]` for input and `[:, 1:]` for target.
3. **Not using causal mask** — model would "cheat" by attending to future tokens during training.
4. **Incorrect loss reshaping** — must flatten both logits and targets before cross_entropy.
5. **Forgetting `optimizer.zero_grad()`** — gradients accumulate across steps.
6. **Not calling `model.train()`** — dropout (if present) wouldn't be applied correctly.




