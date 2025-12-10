# GPT-2 from Scratch

## Problem Statement

Implement **GPT-2** (Generative Pre-trained Transformer 2) from scratch. GPT-2 is a decoder-only transformer for autoregressive language modeling.

Your task is to:

1. Implement causal self-attention (masked)
2. Build transformer decoder blocks
3. Add token and position embeddings
4. Implement autoregressive text generation

## Requirements

- Causal masking to prevent attending to future tokens
- LayerNorm before attention (Pre-LN)
- GELU activation in MLP
- Support for text generation with sampling

## GPT-2 Architecture

```
Token Embeddings + Position Embeddings
    ↓
Transformer Block × N:
    LayerNorm → Causal Self-Attention → Residual
    LayerNorm → MLP (GELU) → Residual
    ↓
LayerNorm → Linear (vocab_size)
```

## Function Signature

```python
class GPT2(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, embed_dim: int, 
                 num_heads: int, num_layers: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for next token prediction."""
        pass
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """Autoregressively generate tokens."""
        pass
```

## Example

```python
gpt = GPT2(
    vocab_size=50257,
    max_len=1024,
    embed_dim=768,
    num_heads=12,
    num_layers=12
)

# Forward pass
tokens = torch.randint(0, 50257, (1, 10))
logits = gpt(tokens)  # (1, 10, 50257)

# Generate
generated = gpt.generate(tokens, max_new_tokens=20)
```

## Hints

- Causal mask: `torch.triu(torch.ones(T, T), diagonal=1)`
- Use learned position embeddings, not sinusoidal
- Apply dropout to embeddings, attention, and MLP
- Top-k or nucleus sampling for better generation
