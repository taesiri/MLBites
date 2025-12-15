# Multi-Head Attention (PyTorch)

## Problem
Multi-head attention is a core building block of Transformers. It computes attention weights between query/key vectors and uses them to mix value vectors.

## Task
Implement a **basic multi-head attention forward pass** in PyTorch (from scratch, no `torch.nn.MultiheadAttention`).

Your implementation must support:
- batched inputs (`(B, T, D)`)
- multi-head projection and recombination
- optional attention masking (`attn_mask`) and key padding masking (`key_padding_mask`)

## Function Signature
```python
import torch

def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    attn_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

## Inputs and Outputs
- **Inputs**:
  - `q`: `(B, Tq, D)` float tensor (queries)
  - `k`: `(B, Tk, D)` float tensor (keys)
  - `v`: `(B, Tk, D)` float tensor (values)
  - `w_q`, `w_k`, `w_v`, `w_o`: `(D, D)` float tensors (projection matrices)
  - `num_heads`: number of heads `H` (must divide `D`)
  - `attn_mask` (optional): `(Tq, Tk)` tensor
    - if `bool`: `True` means "masked out" (disallowed)
    - if `float`: additive mask (e.g., `0` for allowed, `-inf` for disallowed)
  - `key_padding_mask` (optional): `(B, Tk)` bool tensor where `True` means "this key position is padding; mask it out"
- **Output**:
  - `y`: `(B, Tq, D)` float tensor (attention output)

## Constraints
- Use **PyTorch only** (`torch`).
- Do **not** call `torch.nn.MultiheadAttention` or `torch.nn.functional.scaled_dot_product_attention`.
- Keep it interview-friendly (clean, mostly vectorized, 20–30 minutes).

## Examples
### Example 1 (uniform attention when scores are all zeros)
```python
import torch

B, Tq, Tk, D, H = 1, 2, 2, 4, 2
q = torch.zeros(B, Tq, D)
k = torch.zeros(B, Tk, D)
v = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0]]])

w_q = torch.eye(D)
w_k = torch.eye(D)
w_v = torch.eye(D)
w_o = torch.eye(D)

y = multihead_attention(q, k, v, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, num_heads=H)
# Expected: each query attends 50/50 to the two keys, so y is the mean of v:
# tensor([[[3., 4., 5., 6.],
#          [3., 4., 5., 6.]]])
```

### Example 2 (causal mask makes early positions attend to fewer keys)
```python
import torch

B, T, D, H = 1, 3, 2, 1
q = torch.zeros(B, T, D)
k = torch.zeros(B, T, D)
v = torch.tensor([[[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]]])

w_q = torch.eye(D)
w_k = torch.eye(D)
w_v = torch.eye(D)
w_o = torch.eye(D)

# True above diagonal => disallow attending to future positions
attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

y = multihead_attention(q, k, v, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, num_heads=H, attn_mask=attn_mask)

# Scores are still all zeros, but masking changes the allowed set:
# t=0: attend only to {0}        -> [1.0, 0.0]
# t=1: attend uniformly to {0,1} -> [0.5, 0.5]
# t=2: attend uniformly to {0,1,2} -> [2/3, 2/3]
# Expected:
# tensor([[[1.0000, 0.0000],
#          [0.5000, 0.5000],
#          [0.6667, 0.6667]]])
```


