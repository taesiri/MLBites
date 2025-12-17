from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout_p: float = 0.0,
    ) -> None:
        """Causal multi-head self-attention.

        Args:
            embed_dim: Model dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            block_size: Maximum sequence length (for causal mask).
            dropout_p: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout_p)
        self.resid_dropout = nn.Dropout(dropout_p)

        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scale = self.head_dim**-0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, T, T)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim: int, dropout_p: float = 0.0) -> None:
        """Feed-forward network with GELU activation.

        Args:
            embed_dim: Input and output dimension.
            dropout_p: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout_p: float = 0.0,
    ) -> None:
        """A single Transformer block with pre-normalization.

        Args:
            embed_dim: Model dimension.
            num_heads: Number of attention heads.
            block_size: Maximum sequence length.
            dropout_p: Dropout probability.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout_p)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Transformer block.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout_p: float = 0.0,
    ) -> None:
        """Minimal GPT-2 model.

        Args:
            vocab_size: Size of vocabulary.
            embed_dim: Model dimension.
            num_heads: Number of attention heads per layer.
            num_layers: Number of Transformer blocks.
            block_size: Maximum sequence length.
            dropout_p: Dropout probability.
        """
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout_p)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, block_size, dropout_p)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for language modeling.

        Args:
            idx: Token indices of shape (B, T).

        Returns:
            Logits of shape (B, T, vocab_size).
        """
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, C)
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.pos_emb(pos)  # (T, C)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

