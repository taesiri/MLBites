# Sequence Padding and Masking for Transformers

## Problem
When processing batches of variable-length sequences in Transformers, we need to:
1. Pad sequences to a common length so they can be batched together
2. Create padding masks so the attention mechanism ignores padded positions
3. Create causal masks for autoregressive (decoder-style) attention

These utilities are foundational for any Transformer implementation.

## Task
Implement three utility functions for sequence padding and masking in PyTorch:

1. `pad_sequences`: Pad a list of variable-length 1D tensors to create a batch
2. `create_padding_mask`: Create a boolean mask indicating which positions are padding
3. `create_causal_mask`: Create a boolean mask for causal (autoregressive) attention

## Function Signature

```python
def pad_sequences(
    sequences: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and return lengths."""
    ...

def create_padding_mask(
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """Create a padding mask from sequence lengths."""
    ...

def create_causal_mask(
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a causal (future-blocking) mask for autoregressive attention."""
    ...
```

## Inputs and Outputs

### `pad_sequences`
- **inputs**:
  - `sequences`: list of 1D `torch.Tensor` of varying lengths (all same dtype)
  - `pad_value`: scalar value used for padding (default `0.0`)
- **outputs**:
  - `padded`: `torch.Tensor` of shape `(B, T)` where `B = len(sequences)` and `T = max sequence length`
  - `lengths`: `torch.Tensor` of shape `(B,)` containing the original length of each sequence

### `create_padding_mask`
- **inputs**:
  - `lengths`: 1D `torch.Tensor` of shape `(B,)` with sequence lengths
  - `max_len`: int, the padded sequence length
- **outputs**:
  - `mask`: boolean `torch.Tensor` of shape `(B, T)` where `True` = padding position (to be masked/ignored), `False` = valid position

### `create_causal_mask`
- **inputs**:
  - `seq_len`: int, the sequence length
  - `device`: optional torch device
- **outputs**:
  - `mask`: boolean `torch.Tensor` of shape `(T, T)` where `True` = future position (to be masked), `False` = valid (current or past) position. Position `(i, j)` is `True` if `j > i`.

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1: Padding sequences

```python
import torch

seqs = [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([4.0, 5.0]),
    torch.tensor([6.0]),
]
padded, lengths = pad_sequences(seqs, pad_value=0.0)
print(padded)
# tensor([[1., 2., 3.],
#         [4., 5., 0.],
#         [6., 0., 0.]])
print(lengths)
# tensor([3, 2, 1])
```

### Example 2: Creating padding mask

```python
import torch

lengths = torch.tensor([3, 2, 1])
mask = create_padding_mask(lengths, max_len=3)
print(mask)
# tensor([[False, False, False],
#         [False, False,  True],
#         [False,  True,  True]])
```

### Example 3: Creating causal mask

```python
import torch

mask = create_causal_mask(seq_len=4)
print(mask)
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])
```



