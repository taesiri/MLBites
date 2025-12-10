"""
Rotary Position Embedding (RoPE) - Solution

Complete implementation of RoPE from scratch.
"""

import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials.
    
    Returns complex tensor of shape (max_seq_len, dim//2).
    """
    # Compute frequencies for each dimension pair
    # θ_i = 1 / (10000^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Create position indices [0, 1, ..., max_seq_len-1]
    pos = torch.arange(max_seq_len)
    
    # Outer product: each position times each frequency
    # Shape: (max_seq_len, dim//2)
    freqs = torch.outer(pos, freqs)
    
    # Convert to complex exponential: e^(i*θ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor from (seq, dim) to (1, seq, 1, dim)."""
    ndim = x.ndim
    assert ndim >= 2
    shape = [1] * (ndim - 2) + [freqs_cis.shape[0], freqs_cis.shape[1]]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K tensors."""
    # Reshape to (batch, seq, n_heads, head_dim//2, 2)
    xq_pairs = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_pairs = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # View as complex: (batch, seq, n_heads, head_dim//2)
    xq_complex = torch.view_as_complex(xq_pairs)
    xk_complex = torch.view_as_complex(xk_pairs)
    
    # Reshape freqs_cis for broadcasting: (1, seq, 1, head_dim//2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    
    # Apply rotation via complex multiplication
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis
    
    # Convert back to real and flatten
    xq_out = torch.view_as_real(xq_rotated).flatten(-2)
    xk_out = torch.view_as_real(xk_rotated).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings using real arithmetic (alternative to complex).
    
    This is mathematically equivalent but uses only real operations.
    """
    # Split into even and odd indices
    xq_even = xq[..., ::2]
    xq_odd = xq[..., 1::2]
    xk_even = xk[..., ::2]
    xk_odd = xk[..., 1::2]
    
    # Apply rotation
    xq_rotated_even = xq_even * cos - xq_odd * sin
    xq_rotated_odd = xq_even * sin + xq_odd * cos
    xk_rotated_even = xk_even * cos - xk_odd * sin
    xk_rotated_odd = xk_even * sin + xk_odd * cos
    
    # Interleave back
    xq_out = torch.stack([xq_rotated_even, xq_rotated_odd], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_rotated_even, xk_rotated_odd], dim=-1).flatten(-2)
    
    return xq_out, xk_out


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer('freqs_cis', freqs_cis)
    
    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0):
        """Apply RoPE to Q and K."""
        seq_len = xq.shape[1]
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        return apply_rotary_emb(xq, xk, freqs_cis)


class RoPEAttention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, embed_dim: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0):
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        xq, xk = self.rope(xq, xk, start_pos)
        
        # Attention
        xq = xq.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, xv)
        
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test precompute_freqs_cis
    head_dim = 64
    max_seq_len = 512
    
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
    print(f"Precomputed frequencies shape: {freqs_cis.shape}")
    
    # Test apply_rotary_emb
    batch, seq_len, n_heads = 2, 10, 8
    
    q = torch.randn(batch, seq_len, n_heads, head_dim)
    k = torch.randn(batch, seq_len, n_heads, head_dim)
    
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis[:seq_len])
    
    print(f"Q shape: {q.shape} -> {q_rot.shape}")
    print(f"K shape: {k.shape} -> {k_rot.shape}")
    
    # Test full attention with RoPE
    print("\n--- RoPE Attention ---")
    attn = RoPEAttention(embed_dim=512, n_heads=8)
    x = torch.randn(2, 10, 512)
    out = attn(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
