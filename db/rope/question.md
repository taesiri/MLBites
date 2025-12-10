# Rotary Position Embedding (RoPE)

## Problem Statement

Implement **Rotary Position Embedding (RoPE)** from scratch. RoPE is used in LLaMA, Mistral, and other modern LLMs to encode positional information by rotating query and key vectors.

Your task is to:

1. Precompute rotation frequencies based on position
2. Apply rotation to query and key vectors
3. Implement the rotation using complex number formulation
4. Support variable sequence lengths

## Requirements

- Encode position through rotation in embedding space
- Apply to Q and K (not V) in attention
- Support caching of precomputed frequencies
- Handle both real and complex implementations

## Function Signature

```python
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials."""
    pass

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K."""
    pass
```

## RoPE Formula

For position m and dimension pair (2i, 2i+1):
```
θ_i = 10000^(-2i/d)

RoPE(x, m)_2i = x_2i * cos(m*θ_i) - x_{2i+1} * sin(m*θ_i)
RoPE(x, m)_{2i+1} = x_2i * sin(m*θ_i) + x_{2i+1} * cos(m*θ_i)
```

This is equivalent to complex multiplication:
```
(x_2i + i*x_{2i+1}) * (cos(m*θ_i) + i*sin(m*θ_i))
```

## Example

```python
batch, seq_len, n_heads, head_dim = 2, 10, 8, 64

freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=512)

q = torch.randn(batch, seq_len, n_heads, head_dim)
k = torch.randn(batch, seq_len, n_heads, head_dim)

q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis[:seq_len])
```

## Hints

- View tensor as pairs: reshape (..., head_dim) to (..., head_dim//2, 2)
- Use `torch.view_as_complex` and `torch.view_as_real` for efficiency
- Frequencies decrease for higher dimensions (captures different length patterns)
- RoPE allows attention to learn relative positions: q·k depends on (m-n)
