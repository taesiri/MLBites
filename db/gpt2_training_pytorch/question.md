# Train GPT-2 on Dummy Data with Adam

## Problem

Language models like GPT-2 are trained using next-token prediction: given a sequence of tokens, predict the next token at each position. This requires implementing a proper training loop with:
1. A simple GPT-2 style model (embeddings + transformer blocks + LM head)
2. Adam optimizer for parameter updates
3. Cross-entropy loss for next-token prediction
4. Dummy data generation for testing

## Task

Implement a function `train_gpt2` that:
1. Creates a minimal GPT-2 model with the given hyperparameters
2. Creates an Adam optimizer
3. Generates random dummy training data (token sequences)
4. Trains the model for a specified number of steps using next-token prediction loss
5. Returns the trained model and list of losses

The GPT-2 model should include:
- Token embeddings + learned positional embeddings
- N transformer blocks with causal self-attention and MLP
- Final layer norm and linear head for token prediction

## Function Signature

```python
def train_gpt2(
    vocab_size: int = 100,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    block_size: int = 32,
    batch_size: int = 4,
    num_steps: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[torch.nn.Module, list[float]]:
    """Train a minimal GPT-2 on dummy data for next-token prediction.

    Args:
        vocab_size: Size of the token vocabulary.
        embed_dim: Embedding and hidden dimension.
        num_heads: Number of attention heads per layer.
        num_layers: Number of transformer blocks.
        block_size: Maximum sequence length.
        batch_size: Number of sequences per training step.
        num_steps: Number of training iterations.
        lr: Learning rate for Adam optimizer.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (trained_model, losses) where losses is a list of
        float loss values, one per training step.
    """
```

## Inputs and Outputs

**Inputs:**
- `vocab_size` (int): Size of vocabulary, tokens are integers in `[0, vocab_size)`
- `embed_dim` (int): Model hidden dimension (must be divisible by `num_heads`)
- `num_heads` (int): Number of attention heads
- `num_layers` (int): Number of transformer blocks
- `block_size` (int): Maximum context length
- `batch_size` (int): Batch size for training
- `num_steps` (int): Number of training steps
- `lr` (float): Adam learning rate
- `seed` (int): Random seed for reproducibility

**Outputs:**
- `model` (nn.Module): Trained GPT-2 model
- `losses` (list[float]): List of training losses, length equals `num_steps`

## Constraints

- Must be solvable in 25–30 minutes.
- Interview-friendly: use a straightforward model and training loop.
- Dummy data: generate random token sequences of length `block_size + 1`, use first `block_size` tokens as input, last `block_size` tokens as targets (shifted by 1).
- Use cross-entropy loss averaged over all positions.
- Use standard Adam optimizer with provided learning rate.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- Must be deterministic given the seed.

## Examples

### Example 1 (basic training)

```python
import torch

model, losses = train_gpt2(
    vocab_size=50,
    embed_dim=32,
    num_heads=2,
    num_layers=1,
    block_size=16,
    batch_size=2,
    num_steps=10,
    lr=1e-3,
    seed=42,
)

# Model output shape should be correct
idx = torch.randint(0, 50, (1, 8))
logits = model(idx)
print(logits.shape)  # torch.Size([1, 8, 50])

# Should have 10 loss values
print(len(losses))  # 10

# Losses should be positive floats
print(all(loss > 0 for loss in losses))  # True
```

### Example 2 (loss should decrease with training)

```python
model, losses = train_gpt2(
    vocab_size=100,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    block_size=32,
    batch_size=8,
    num_steps=100,
    lr=3e-3,
    seed=123,
)

# With enough steps and appropriate lr, loss should generally decrease
# (comparing early vs late average)
early_avg = sum(losses[:10]) / 10
late_avg = sum(losses[-10:]) / 10
print(late_avg < early_avg)  # True (loss decreases)
```

### Example 3 (deterministic with seed)

```python
_, losses1 = train_gpt2(seed=999, num_steps=5)
_, losses2 = train_gpt2(seed=999, num_steps=5)

print(losses1 == losses2)  # True (deterministic)
```




