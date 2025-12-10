"""
GPT-2 from Scratch - Solution

Complete GPT-2 implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal (masked) self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer('mask', torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_dropout(self.proj(out))


class MLP(nn.Module):
    """Feed-forward with GELU."""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """GPT-2 block with Pre-LN."""
    
    def __init__(self, embed_dim: int, num_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 language model."""
    
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 1024,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_len = max_len
        
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, max_len, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying
        self.tok_emb.weight = self.head.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_len
        
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :T]
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None):
        """Autoregressively generate tokens."""
        for _ in range(max_new_tokens):
            # Crop to max_len
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        
        return idx


if __name__ == "__main__":
    torch.manual_seed(42)
    
    gpt = GPT2(
        vocab_size=1000,
        max_len=128,
        embed_dim=256,
        num_heads=4,
        num_layers=4
    )
    
    tokens = torch.randint(0, 1000, (2, 20))
    logits = gpt(tokens)
    
    print(f"Input: {tokens.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in gpt.parameters()):,}")
    
    prompt = torch.randint(0, 1000, (1, 5))
    generated = gpt.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"Generated: {generated.shape}")
