# GPT-2 from Scratch

## Problem
GPT-2 is a decoder-only Transformer that uses causal (masked) self-attention to generate text. It consists of token and positional embeddings, stacked Transformer blocks (each with layer normalization, causal multi-head attention, and an MLP with GELU activation), and a final language modeling head.

## Task
Implement a minimal GPT-2 model in PyTorch. Your implementation should include:
1. **Token + Positional Embeddings**
2. **Transformer Block** with pre-normalization (LayerNorm before attention and MLP), causal self-attention, and an MLP with GELU
3. **Full GPT-2 Model** that stacks N blocks and outputs logits for next-token prediction

## Function Signature

```python
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout_p: float = 0.0,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class MLP(nn.Module):
    def __init__(self, embed_dim: int, dropout_p: float = 0.0) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout_p: float = 0.0,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout_p: float = 0.0,
    ) -> None: ...

    def forward(self, idx: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs
- **CausalSelfAttention**:
  - Input: `x` of shape `(B, T, C)` where B=batch, T=sequence length, C=embed_dim
  - Output: tensor of shape `(B, T, C)`

- **MLP**:
  - Input: `x` of shape `(B, T, C)`
  - Output: tensor of shape `(B, T, C)`

- **TransformerBlock**:
  - Input: `x` of shape `(B, T, C)`
  - Output: tensor of shape `(B, T, C)`

- **GPT2**:
  - Input: `idx` of shape `(B, T)` containing token indices (integers in `[0, vocab_size)`)
  - Output: logits of shape `(B, T, vocab_size)`

## Constraints
- Must be solvable in 25–30 minutes.
- Interview-friendly: focus on the core architecture, not production optimizations.
- Assume inputs satisfy the documented contract (e.g., sequence length ≤ block_size).
- Allowed libs: PyTorch (`torch`) and Python standard library.
- Use pre-normalization (LayerNorm before attention/MLP, not after).
- The MLP hidden dimension should be `4 * embed_dim` (standard GPT-2).
- Use GELU activation in the MLP.
- Causal mask ensures each position can only attend to previous positions (including itself).

## Examples

### Example 1 (forward pass shapes)

```python
import torch

model = GPT2(
    vocab_size=100,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    block_size=32,
    dropout_p=0.0,
)

idx = torch.randint(0, 100, (2, 10))  # (B=2, T=10)
logits = model(idx)
print(logits.shape)  # torch.Size([2, 10, 100])
```

### Example 2 (causal attention prevents future info leakage)

```python
import torch

torch.manual_seed(42)

attn = CausalSelfAttention(embed_dim=8, num_heads=2, block_size=4, dropout_p=0.0)
attn.eval()

x = torch.randn(1, 4, 8)  # (B=1, T=4, C=8)
y = attn(x)

# Changing future tokens should NOT affect past outputs
x2 = x.clone()
x2[0, 3, :] = 999.0  # modify last token
y2 = attn(x2)

# First 3 positions should be identical
print(torch.allclose(y[0, :3], y2[0, :3]))  # True
```

### Example 3 (MLP dimensions)

```python
import torch

mlp = MLP(embed_dim=16, dropout_p=0.0)
x = torch.randn(2, 5, 16)
y = mlp(x)
print(y.shape)  # torch.Size([2, 5, 16])
```




