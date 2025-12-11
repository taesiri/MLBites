"""
KV Cache for LLM Inference - Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Key-Value cache for efficient autoregressive generation."""
    
    def __init__(self, max_batch_size: int, max_seq_len: int,
                 num_layers: int, num_heads: int, head_dim: int, 
                 device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.current_pos = 0
        
        cache_shape = (num_layers, max_batch_size, num_heads, max_seq_len, head_dim)
        self.cache_k = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.cache_v = torch.zeros(cache_shape, device=device, dtype=dtype)
    
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor,
               start_pos: int = None) -> tuple:
        """Update cache and return full K,V."""
        if start_pos is None:
            start_pos = self.current_pos
        
        seq_len = key.shape[2]
        
        # Store new K,V
        self.cache_k[layer_idx, :, :, start_pos:start_pos + seq_len] = key
        self.cache_v[layer_idx, :, :, start_pos:start_pos + seq_len] = value
        
        # Update position (only on last layer)
        if layer_idx == self.num_layers - 1:
            self.current_pos = start_pos + seq_len
        
        # Return cached K,V up to current position
        end_pos = start_pos + seq_len
        return (
            self.cache_k[layer_idx, :, :, :end_pos],
            self.cache_v[layer_idx, :, :, :end_pos]
        )
    
    def get(self, layer_idx: int) -> tuple:
        return (
            self.cache_k[layer_idx, :, :, :self.current_pos],
            self.cache_v[layer_idx, :, :, :self.current_pos]
        )
    
    def clear(self):
        self.current_pos = 0
        self.cache_k.zero_()
        self.cache_v.zero_()


class CausalAttentionWithCache(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, cache: KVCache = None,
                layer_idx: int = 0, start_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update cache if provided
        if cache is not None:
            k, v = cache.update(layer_idx, k, v, start_pos)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask (only needed for prefill, not decode)
        if T > 1:
            mask = torch.triu(torch.ones(T, k.size(2), device=x.device), diagonal=start_pos + 1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class TransformerBlockWithCache(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalAttentionWithCache(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
    
    def forward(self, x, cache=None, layer_idx=0, start_pos=0):
        x = x + self.attn(self.ln1(x), cache, layer_idx, start_pos)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTWithCache(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, max_seq_len: int = 2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlockWithCache(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
    
    def forward(self, idx: torch.Tensor, cache: KVCache = None, start_pos: int = 0):
        B, T = idx.shape
        
        x = self.tok_emb(idx) + self.pos_emb[:, start_pos:start_pos + T]
        
        for i, block in enumerate(self.blocks):
            x = block(x, cache, layer_idx=i, start_pos=start_pos)
        
        x = self.ln_f(x)
        return self.head(x)
    
    def create_cache(self, batch_size: int, max_len: int, device: str = 'cpu'):
        return KVCache(batch_size, max_len, self.num_layers, 
                       self.num_heads, self.head_dim, device)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    model = GPTWithCache(vocab_size=1000, embed_dim=256, num_heads=4, num_layers=4)
    cache = model.create_cache(batch_size=2, max_len=128)
    
    # Prefill
    prompt = torch.randint(0, 1000, (2, 10))
    logits = model(prompt, cache=cache, start_pos=0)
    print(f"Prefill: input {prompt.shape}, output {logits.shape}")
    
    # Decode
    for i in range(5):
        next_token = logits[:, -1:].argmax(dim=-1)
        logits = model(next_token, cache=cache, start_pos=10 + i)
        print(f"Decode {i}: input {next_token.shape}, output {logits.shape}")
