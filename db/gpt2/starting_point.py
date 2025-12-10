"""
GPT-2 from Scratch - Starting Point

Implement GPT-2 from scratch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal (masked) self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        # TODO: Create Q, K, V projections
        # TODO: Output projection
        # TODO: Register causal mask buffer
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        """
        # TODO: Project to Q, K, V
        # TODO: Split into heads
        # TODO: Compute attention scores
        # TODO: Apply causal mask
        # TODO: Softmax and apply to V
        # TODO: Concatenate heads and project
        pass


class MLP(nn.Module):
    """Feed-forward network with GELU."""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # TODO: Two linear layers with 4x expansion
        # TODO: GELU activation
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TransformerBlock(nn.Module):
    """GPT-2 transformer block (Pre-LN)."""
    
    def __init__(self, embed_dim: int, num_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        # TODO: LayerNorms, attention, MLP
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pre-LN architecture
        # x = x + attn(ln(x))
        # x = x + mlp(ln(x))
        pass


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
        # TODO: Token embeddings
        # TODO: Position embeddings
        # TODO: Transformer blocks
        # TODO: Final LayerNorm
        # TODO: Output projection (weight tying optional)
        pass
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Token indices (batch, seq_len)
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        # TODO: Embed tokens and positions
        # TODO: Pass through transformer blocks
        # TODO: Final norm and project to vocab
        pass
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """Autoregressively generate tokens."""
        # TODO: Loop max_new_tokens times:
        #   - Forward pass (only care about last position)
        #   - Sample next token
        #   - Append to sequence
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create small GPT-2
    gpt = GPT2(
        vocab_size=1000,
        max_len=128,
        embed_dim=256,
        num_heads=4,
        num_layers=4
    )
    
    # Test forward
    tokens = torch.randint(0, 1000, (2, 20))
    logits = gpt(tokens)
    
    print(f"Input: {tokens.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in gpt.parameters()):,}")
    
    # Test generation
    prompt = torch.randint(0, 1000, (1, 5))
    generated = gpt.generate(prompt, max_new_tokens=10)
    print(f"Generated: {generated.shape}")
