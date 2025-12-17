from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(
        self, embed_dim: int, num_heads: int, block_size: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim**-0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    """Feed-forward network with GELU."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(self, embed_dim: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """Minimal GPT-2 model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, block_size)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)


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
    """
    torch.manual_seed(seed)

    # Build model
    model = GPT2(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
    )
    model.train()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []

    for _ in range(num_steps):
        # Generate dummy data: random token sequences
        tokens = torch.randint(0, vocab_size, (batch_size, block_size + 1))
        inputs = tokens[:, :-1]   # (B, block_size)
        targets = tokens[:, 1:]   # (B, block_size)

        # Forward pass
        logits = model(inputs)  # (B, block_size, vocab_size)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses

