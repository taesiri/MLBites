"""
Rotary Position Embedding (RoPE) - Starting Point

Implement RoPE from scratch.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis = cos + i*sin).
    
    Args:
        dim: Dimension of embeddings (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
        
    Returns:
        Complex tensor of shape (max_seq_len, dim//2)
    """
    # TODO: Compute frequencies for each dimension pair
    # freqs = 1 / (theta^(2i/dim)) for i in 0..dim//2
    
    # TODO: Create position indices
    # pos = [0, 1, 2, ..., max_seq_len-1]
    
    # TODO: Compute outer product: pos * freqs
    # Shape: (max_seq_len, dim//2)
    
    # TODO: Convert to complex exponential (cos + i*sin)
    # freqs_cis = e^(i * pos * freqs) = cos(...) + i*sin(...)
    
    pass


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting with x.
    
    freqs_cis is (seq_len, head_dim//2)
    x is (batch, seq_len, n_heads, head_dim//2, 2) or complex
    
    Need to add dimensions for batch and n_heads.
    """
    # TODO: Add appropriate dimensions
    pass


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to Q and K tensors.
    
    Args:
        xq: Query tensor (batch, seq_len, n_heads, head_dim)
        xk: Key tensor (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed frequencies (seq_len, head_dim//2)
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # TODO: Reshape xq and xk to pairs of adjacent elements
    # (batch, seq, n_heads, head_dim) -> (batch, seq, n_heads, head_dim//2, 2)
    
    # TODO: View as complex numbers
    # xq_complex = view_as_complex(xq_pairs)
    
    # TODO: Reshape freqs_cis for broadcasting
    
    # TODO: Apply rotation via complex multiplication
    # xq_rotated = xq_complex * freqs_cis
    
    # TODO: Convert back to real and flatten
    
    pass


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        # TODO: Precompute and register frequencies
        pass
    
    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0):
        """Apply RoPE to Q and K."""
        # TODO: Get frequencies for current positions
        # TODO: Apply rotary embedding
        pass


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
    
    # Verify relative position property
    print("\nRelative position property:")
    print("Q[0] · K[0] should depend on relative position (0-0=0)")
