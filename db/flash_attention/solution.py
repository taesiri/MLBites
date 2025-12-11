"""
Flash Attention in Triton - Solution

Complete Flash Attention implementation.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def flash_attention_fwd_kernel(
        Q, K, V, O,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        N_CTX,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        # Program IDs
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        
        # Compute batch and head
        num_heads = stride_qb // stride_qh
        batch_idx = pid_bh // num_heads
        head_idx = pid_bh % num_heads
        
        # Offsets for this block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)
        
        # Pointers
        q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + \
                 offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + \
                 offs_k[None, :] * stride_kk
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + \
                 offs_k[None, :] * stride_vk
        
        # Load Q block
        mask_m = offs_m < N_CTX
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        
        # Initialize accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        o_i = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        
        # Loop over K, V blocks
        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n_curr = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n_curr < N_CTX
            
            # Load K, V blocks
            k_ptrs_curr = k_ptrs + offs_n_curr[:, None] * stride_kn
            v_ptrs_curr = v_ptrs + offs_n_curr[:, None] * stride_vn
            
            k = tl.load(k_ptrs_curr, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs_curr, mask=mask_n[:, None], other=0.0)
            
            # Compute attention scores
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(mask_n[None, :], s, float('-inf'))
            
            # Online softmax
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)
            
            # Rescale and accumulate
            o_i = o_i * (tl.exp(m_i - m_new))[:, None] + tl.dot(p.to(v.dtype), v)
            
            m_i = m_new
            l_i = l_new
        
        # Normalize
        o_i = o_i / l_i[:, None]
        
        # Store output
        o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + \
                 offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(o_ptrs, o_i.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Flash Attention forward pass."""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    batch, heads, seq_len, head_dim = q.shape
    assert k.shape == v.shape == q.shape
    
    o = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    
    scale = head_dim ** -0.5
    
    flash_attention_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
    )
    
    return o


def standard_attention(q, k, v):
    """Standard attention for comparison."""
    scale = q.shape[-1] ** -0.5
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    return attn @ v


if __name__ == "__main__":
    if not TRITON_AVAILABLE:
        print("Triton not installed")
    elif not torch.cuda.is_available():
        print("CUDA not available")
    else:
        torch.manual_seed(42)
        
        batch, heads, seq_len, head_dim = 2, 4, 256, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        output_flash = flash_attention(q, k, v)
        output_std = standard_attention(q, k, v)
        
        print(f"Output shape: {output_flash.shape}")
        print(f"Max diff: {(output_flash - output_std).abs().max():.6f}")
        
        # Benchmark
        import time
        
        for name, fn in [("Standard", standard_attention), ("Flash", flash_attention)]:
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = fn(q, k, v)
            torch.cuda.synchronize()
            print(f"{name}: {(time.time() - start) * 10:.2f}ms")
