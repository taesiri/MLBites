# Flash Attention in Triton

## Problem Statement

Implement **Flash Attention** using Triton. Flash Attention is a memory-efficient attention algorithm that reduces memory from O(N²) to O(N) by computing attention in blocks without materializing the full attention matrix.

Your task is to:

1. Understand the tiled/blocked attention algorithm
2. Implement the forward pass in Triton
3. Apply online softmax (numerically stable)
4. Avoid materializing the N×N attention matrix

## Flash Attention Algorithm

```
For each query block Q_i:
    Initialize: m_i = -∞, l_i = 0, O_i = 0
    For each key/value block K_j, V_j:
        S_ij = Q_i @ K_j.T / sqrt(d)
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
        m_i, l_i = m_new, l_new
    O_i = O_i / l_i
```

## Function Signature

```python
@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pass

def flash_attention(q, k, v):
    """
    Flash Attention forward pass.
    
    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    pass
```

## Example

```python
import triton

batch, heads, seq_len, head_dim = 2, 8, 1024, 64

q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')

output = flash_attention(q, k, v)
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Tiling** | Process Q, K, V in blocks that fit in SRAM |
| **Online Softmax** | Compute softmax incrementally across K blocks |
| **No materialization** | Never store full N×N attention matrix |
| **IO-aware** | Minimize HBM reads/writes |

## Hints

- Use `tl.load`, `tl.store` for memory access
- `tl.max`, `tl.sum` for reductions
- Block sizes: typically 64-128 for BLOCK_M, BLOCK_N
- Track running max (m) and sum (l) for stable softmax
