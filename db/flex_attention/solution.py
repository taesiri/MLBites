"""
FlexAttention Patterns - Solution

Complete FlexAttention pattern implementations.
"""

import math
import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False


# ============ Attention Mask Functions ============

def causal_mask(b, h, q_idx, kv_idx):
    """Causal (autoregressive) mask."""
    return q_idx >= kv_idx


def sliding_window_mask(window_size: int):
    """Sliding window attention."""
    def mask(b, h, q_idx, kv_idx):
        return torch.abs(q_idx - kv_idx) <= window_size
    return mask


def causal_sliding_window_mask(window_size: int):
    """Causal + sliding window."""
    def mask(b, h, q_idx, kv_idx):
        is_causal = q_idx >= kv_idx
        is_in_window = (q_idx - kv_idx) <= window_size
        return is_causal & is_in_window
    return mask


def prefix_lm_mask(prefix_length: int):
    """Prefix LM: bidirectional for prefix, causal after."""
    def mask(b, h, q_idx, kv_idx):
        is_prefix = kv_idx < prefix_length
        is_causal = q_idx >= kv_idx
        return is_prefix | is_causal
    return mask


def local_global_mask(local_window: int, global_tokens: int):
    """Longformer-style local + global attention."""
    def mask(b, h, q_idx, kv_idx):
        is_global_key = kv_idx < global_tokens
        is_global_query = q_idx < global_tokens
        is_local = torch.abs(q_idx - kv_idx) <= local_window
        return is_global_key | is_global_query | is_local
    return mask


def block_sparse_mask(block_size: int):
    """Block-sparse attention pattern."""
    def mask(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_block = kv_idx // block_size
        return q_block == kv_block
    return mask


def dilated_attention_mask(dilation: int, window_size: int):
    """Dilated attention with gaps."""
    def mask(b, h, q_idx, kv_idx):
        distance = torch.abs(q_idx - kv_idx)
        is_in_window = distance <= window_size * dilation
        is_dilated_position = (distance % dilation) == 0
        return is_in_window & is_dilated_position
    return mask


# ============ Score Modification Functions ============

def alibi_bias(num_heads: int):
    """ALiBi positional bias."""
    slopes = torch.pow(2.0, torch.arange(1, num_heads + 1) * (-8.0 / num_heads))
    
    def score_mod(score, b, h, q_idx, kv_idx):
        distance = torch.abs(q_idx - kv_idx)
        bias = -slopes[h] * distance
        return score + bias
    return score_mod


def create_alibi_bias_tensor(num_heads: int, seq_len: int, device='cpu'):
    """Pre-compute ALiBi bias tensor."""
    slopes = torch.pow(2.0, torch.arange(1, num_heads + 1, device=device) * (-8.0 / num_heads))
    positions = torch.arange(seq_len, device=device)
    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
    bias = slopes.unsqueeze(1).unsqueeze(2) * relative_positions.abs().unsqueeze(0)
    return -bias


def softcap_score(cap: float = 50.0):
    """Gemma2-style soft score capping."""
    def score_mod(score, b, h, q_idx, kv_idx):
        return cap * torch.tanh(score / cap)
    return score_mod


# ============ Combining Masks ============

def and_masks(*masks):
    """Combine masks with AND."""
    def combined(b, h, q_idx, kv_idx):
        result = masks[0](b, h, q_idx, kv_idx)
        for mask in masks[1:]:
            result = result & mask(b, h, q_idx, kv_idx)
        return result
    return combined


def or_masks(*masks):
    """Combine masks with OR."""
    def combined(b, h, q_idx, kv_idx):
        result = masks[0](b, h, q_idx, kv_idx)
        for mask in masks[1:]:
            result = result | mask(b, h, q_idx, kv_idx)
        return result
    return combined


# ============ Standard Attention (fallback) ============

def standard_attention_with_mask(q, k, v, mask_fn):
    """Standard attention with custom mask."""
    B, H, N, D = q.shape
    scale = D ** -0.5
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    q_idx = torch.arange(N, device=q.device).view(-1, 1)
    kv_idx = torch.arange(N, device=q.device).view(1, -1)
    mask = mask_fn(0, 0, q_idx, kv_idx)
    
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, 0.0)
    
    return torch.matmul(attn, v)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch, heads, seq_len, head_dim = 2, 8, 64, 32
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)
    
    # Test patterns using fallback
    patterns = [
        ("Causal", causal_mask),
        ("Sliding Window (16)", sliding_window_mask(16)),
        ("Causal + Window (32)", causal_sliding_window_mask(32)),
        ("Prefix LM (16)", prefix_lm_mask(16)),
        ("Local+Global", local_global_mask(8, 4)),
    ]
    
    for name, mask_fn in patterns:
        output = standard_attention_with_mask(q, k, v, mask_fn)
        print(f"{name}: output {output.shape}")
    
    # Combine patterns
    print("\n=== Combined Patterns ===")
    combined = and_masks(causal_mask, sliding_window_mask(16))
    output = standard_attention_with_mask(q, k, v, combined)
    print(f"Causal AND Window: {output.shape}")
