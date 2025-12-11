"""
Flash Attention in Triton - Starting Point

Implement Flash Attention using Triton.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available. Install with: pip install triton")


if TRITON_AVAILABLE:
    @triton.jit
    def flash_attention_kernel(
        Q, K, V, O,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """
        Flash Attention forward kernel.
        
        Each program instance handles one (batch, head, query_block).
        """
        # TODO: Get program IDs
        # pid_m = tl.program_id(0)  # query block index
        # pid_bh = tl.program_id(1)  # batch * head index
        
        # TODO: Compute batch and head indices
        
        # TODO: Initialize pointers to Q, K, V, O
        
        # TODO: Load query block Q[BLOCK_M, HEAD_DIM]
        
        # TODO: Initialize accumulators
        # m_i = -infinity (running max)
        # l_i = 0 (running sum)
        # o_i = 0 (output accumulator)
        
        # TODO: Loop over key/value blocks
        # for start_n in range(0, N_CTX, BLOCK_N):
        #     Load K block, V block
        #     Compute S = Q @ K.T / sqrt(d)
        #     Update m_new = max(m_i, rowmax(S))
        #     Compute P = exp(S - m_new)
        #     Update l_new = exp(m_i - m_new) * l_i + rowsum(P)
        #     Rescale o_i and add P @ V
        #     Update m_i, l_i
        
        # TODO: Normalize output: o_i = o_i / l_i
        
        # TODO: Store output
        
        pass


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Flash Attention forward pass.
    
    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
    
    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    # TODO: Check inputs, allocate output
    
    # TODO: Set up grid and launch kernel
    
    pass


def standard_attention(q, k, v):
    """Standard attention for comparison."""
    scale = q.shape[-1] ** -0.5
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    return attn @ v


if __name__ == "__main__":
    if not TRITON_AVAILABLE:
        print("Skipping - Triton not installed")
    elif not torch.cuda.is_available():
        print("Skipping - CUDA not available")
    else:
        torch.manual_seed(42)
        
        batch, heads, seq_len, head_dim = 2, 4, 256, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Flash attention
        output_flash = flash_attention(q, k, v)
        
        # Standard attention
        output_std = standard_attention(q, k, v)
        
        # Compare
        print(f"Flash output shape: {output_flash.shape}")
        print(f"Max diff: {(output_flash - output_std).abs().max():.6f}")
