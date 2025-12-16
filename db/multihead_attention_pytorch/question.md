# Implement Multi-Head Self-Attention (PyTorch)

## Problem
Multi-head attention is a core building block in Transformers. In its simplest form (self-attention), it projects an input sequence into queries/keys/values, computes attention weights, and returns a weighted sum of values.

## Task
Implement a minimal, interview-friendly PyTorch module that performs **multi-head self-attention** over a batch of sequences.

## Function Signature

```python
class SimpleMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: `torch.Tensor` of shape `(B, T, E)` where:
    - `B` = batch size
    - `T` = sequence length
    - `E` = `embed_dim`
  - `key_padding_mask` (optional): boolean `torch.Tensor` of shape `(B, T)` where:
    - `True` means "this position is padding; ignore it as a key/value"
    - `False` means "valid token"
- **outputs**:
  - returns `torch.Tensor` of shape `(B, T, E)` (same shape as input)

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- You may assume:
  - `embed_dim % num_heads == 0`
  - for each query position, at least one key position is unmasked (to avoid all-`-inf` softmax)

## Examples

### Example 1 (identity projections, 1 head)
If Q=K=V=x and the output projection is identity, the module performs standard scaled dot-product self-attention.

```python
import torch

torch.set_printoptions(precision=4, sci_mode=False)

m = SimpleMultiheadAttention(embed_dim=2, num_heads=1, dropout_p=0.0, bias=False)
with torch.no_grad():
    # qkv_proj: stack [I; I; I] so Q=K=V=x
    m.qkv_proj.weight.zero_()
    m.qkv_proj.weight[0, 0] = 1.0
    m.qkv_proj.weight[1, 1] = 1.0
    m.qkv_proj.weight[2, 0] = 1.0
    m.qkv_proj.weight[3, 1] = 1.0
    m.qkv_proj.weight[4, 0] = 1.0
    m.qkv_proj.weight[5, 1] = 1.0
    m.out_proj.weight.copy_(torch.eye(2))

x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (B=1, T=2, E=2)
y = m(x)
print(y)
# expected approximately:
# tensor([[[0.6698, 0.3302],
#          [0.3302, 0.6698]]])
```

### Example 2 (padding mask ignores masked keys/values)

```python
import torch

m = SimpleMultiheadAttention(embed_dim=2, num_heads=1, dropout_p=0.0, bias=False)
with torch.no_grad():
    m.qkv_proj.weight.copy_(torch.tensor([
        [1.0, 0.0], [0.0, 1.0],  # Q
        [1.0, 0.0], [0.0, 1.0],  # K
        [1.0, 0.0], [0.0, 1.0],  # V
    ]))
    m.out_proj.weight.copy_(torch.eye(2))

x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
mask = torch.tensor([[False, True]])  # second position is padding; ignore as key/value
y = m(x, key_padding_mask=mask)
print(y)
# expected:
# tensor([[[1., 0.],
#          [1., 0.]]])
```


