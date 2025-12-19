from __future__ import annotations

import torch
import torch.nn as nn


def train_gpt2(
    vocab_size: int = 100,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    block_size: int = 32,
    batch_size: int = 4,
    num_steps: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[nn.Module, list[float]]:
    """Train a minimal GPT-2 on dummy data for next-token prediction.

    Args:
        vocab_size: Size of the token vocabulary.
        embed_dim: Embedding and hidden dimension.
        num_heads: Number of attention heads per layer.
        num_layers: Number of transformer blocks.
        block_size: Maximum sequence length.
        batch_size: Number of sequences per training step.
        num_steps: Number of training iterations.
        lr: Learning rate for Adam optimizer.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (trained_model, losses) where losses is a list of
        float loss values, one per training step.

    Implementation steps:
        1. Set random seed for reproducibility
        2. Build a GPT-2 model:
           - Token embedding (vocab_size -> embed_dim)
           - Positional embedding (block_size -> embed_dim)
           - N transformer blocks (LayerNorm + CausalAttention + MLP)
           - Final LayerNorm + linear head to vocab_size
        3. Create Adam optimizer
        4. Training loop for num_steps:
           - Generate random dummy tokens of shape (batch_size, block_size + 1)
           - Input: tokens[:, :-1], Target: tokens[:, 1:]
           - Forward pass, compute cross-entropy loss
           - Backward pass, optimizer step
           - Record loss
        5. Return model and losses
    """
    # TODO: implement
    raise NotImplementedError


