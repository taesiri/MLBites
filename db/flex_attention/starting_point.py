"""
FlexAttention Patterns - Starting Point

Practice implementing attention patterns with FlexAttention.
"""

import torch
import torch.nn.functional as F

# FlexAttention is available in PyTorch 2.5+
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    print("FlexAttention requires PyTorch 2.5+")


# ============ Attention Mask Functions ============

def causal_mask(b, h, q_idx, kv_idx):
    """
    Causal (autoregressive) mask.
    Query can only attend to positions <= its own.
    """
    # TODO: Return True if kv_idx <= q_idx
    pass


def sliding_window_mask(window_size: int):
    """
    Sliding window attention.
    Query attends to positions within window_size distance.
    """
    def mask(b, h, q_idx, kv_idx):
        # TODO: Return True if |q_idx - kv_idx| <= window_size
        pass
    return mask


def causal_sliding_window_mask(window_size: int):
    """
    Causal + sliding window (like Mistral).
    Combines both constraints.
    """
    def mask(b, h, q_idx, kv_idx):
        # TODO: Combine causal and sliding window
        pass
    return mask


def prefix_lm_mask(prefix_length: int):
    """
    Prefix LM: bidirectional for prefix, causal after.
    """
    def mask(b, h, q_idx, kv_idx):
        # TODO: Attend to prefix OR causal after prefix
        pass
    return mask


def local_global_mask(local_window: int, global_tokens: int):
    """
    Longformer-style: local window + global tokens.
    First `global_tokens` positions attend to all.
    """
    def mask(b, h, q_idx, kv_idx):
        # TODO: Local window OR is global token
        pass
    return mask


# ============ Score Modification Functions ============

def alibi_bias(num_heads: int):
    """
    ALiBi positional bias.
    Adds linear bias based on distance.
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        # TODO: Add -slope * |q_idx - kv_idx| to score
        # slope = 2^(-8/num_heads * (h+1))
        pass
    return score_mod


def relative_position_bias(max_distance: int = 128):
    """
    Relative position bias (T5-style).
    """
    def score_mod(score, b, h, q_idx, kv_idx):
        # TODO: Add learned bias based on relative position
        pass
    return score_mod


# ============ Standard Attention (fallback) ============

def standard_attention_with_mask(q, k, v, mask_fn):
    """Standard attention with custom mask (for testing without FlexAttention)."""
    B, H, N, D = q.shape
    scale = D ** -0.5
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Create mask
    q_idx = torch.arange(N, device=q.device).unsqueeze(1)
    kv_idx = torch.arange(N, device=q.device).unsqueeze(0)
    
    mask = mask_fn(0, 0, q_idx, kv_idx)
    scores = scores.masked_fill(~mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch, heads, seq_len, head_dim = 2, 8, 64, 32
    
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)
    
    # Test causal mask
    print("=== Causal Attention ===")
    if FLEX_AVAILABLE:
        q_cuda = q.cuda()
        k_cuda = k.cuda()
        v_cuda = v.cuda()
        output = flex_attention(q_cuda, k_cuda, v_cuda, score_mod=causal_mask)
        print(f"Output shape: {output.shape}")
    else:
        output = standard_attention_with_mask(q, k, v, causal_mask)
        print(f"Output shape (fallback): {output.shape}")
    
    # Test sliding window
    print("\n=== Sliding Window (size=16) ===")
    window_mask = sliding_window_mask(16)
    if FLEX_AVAILABLE:
        output = flex_attention(q_cuda, k_cuda, v_cuda, score_mod=window_mask)
        print(f"Output shape: {output.shape}")
    else:
        output = standard_attention_with_mask(q, k, v, window_mask)
        print(f"Output shape (fallback): {output.shape}")
