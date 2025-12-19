from __future__ import annotations

import torch
import torch.nn as nn


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
        # TODO: store embed_dim, num_heads, head_dim = embed_dim // num_heads
        # TODO: create combined QKV projection: nn.Linear(embed_dim, 3 * embed_dim)
        # TODO: create output projection: nn.Linear(embed_dim, embed_dim)
        # TODO: create dropout layers
        # TODO: register a causal mask buffer (lower triangular)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        # TODO: get B, T, C from x.shape
        # TODO: compute Q, K, V via projection
        # TODO: reshape for multi-head: (B, T, C) -> (B, num_heads, T, head_dim)
        # TODO: compute attention scores: Q @ K^T / sqrt(head_dim)
        # TODO: apply causal mask (set future positions to -inf)
        # TODO: softmax and dropout
        # TODO: apply attention to V
        # TODO: reshape back to (B, T, C) and project output
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, embed_dim: int, dropout_p: float = 0.0) -> None:
        """Feed-forward network with GELU activation.

        Args:
            embed_dim: Input and output dimension.
            dropout_p: Dropout probability.
        """
        super().__init__()
        # TODO: create fc1: nn.Linear(embed_dim, 4 * embed_dim)
        # TODO: create fc2: nn.Linear(4 * embed_dim, embed_dim)
        # TODO: GELU activation
        # TODO: dropout
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        # TODO: fc1 -> GELU -> fc2 -> dropout
        raise NotImplementedError


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
        # TODO: create LayerNorm for attention (ln1)
        # TODO: create CausalSelfAttention
        # TODO: create LayerNorm for MLP (ln2)
        # TODO: create MLP
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Transformer block.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        # TODO: x = x + attn(ln1(x))  [pre-norm + residual]
        # TODO: x = x + mlp(ln2(x))   [pre-norm + residual]
        raise NotImplementedError


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
        # TODO: create token embedding: nn.Embedding(vocab_size, embed_dim)
        # TODO: create position embedding: nn.Embedding(block_size, embed_dim)
        # TODO: create dropout
        # TODO: create nn.ModuleList of TransformerBlocks
        # TODO: create final LayerNorm
        # TODO: create language model head: nn.Linear(embed_dim, vocab_size, bias=False)
        # Optional: tie weights between token embedding and lm_head
        raise NotImplementedError

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for language modeling.

        Args:
            idx: Token indices of shape (B, T).

        Returns:
            Logits of shape (B, T, vocab_size).
        """
        # TODO: get B, T from idx.shape
        # TODO: create position indices: torch.arange(T, device=idx.device)
        # TODO: compute token embeddings + position embeddings
        # TODO: apply dropout
        # TODO: pass through all transformer blocks
        # TODO: apply final layer norm
        # TODO: project to vocabulary logits
        raise NotImplementedError


