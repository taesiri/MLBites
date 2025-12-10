# RNN Text Classifier

## Problem Statement

Implement a **Recurrent Neural Network (RNN)** for sequence classification using PyTorch. RNNs are designed to process sequential data by maintaining a hidden state that captures information from previous time steps.

Your task is to:

1. Build an RNN-based classifier using LSTM or GRU layers
2. Include an embedding layer for input tokens
3. Use the final hidden state for classification
4. Implement the forward pass correctly handling variable length sequences

## Requirements

- Use an embedding layer (`nn.Embedding`) for input tokens
- Use `nn.LSTM` or `nn.GRU` for the recurrent layer
- Support bidirectional RNNs (optional but recommended)
- Use the final hidden state (or pooled states) for classification

## Function Signature

```python
class RNNClassifier(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = False
    ):
        """Initialize RNN classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden state
            num_classes: Number of output classes
            num_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Token indices of shape (batch, seq_len)
            
        Returns:
            Class logits of shape (batch, num_classes)
        """
        pass
```

## Example

```python
import torch

# Create RNN classifier
model = RNNClassifier(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    num_classes=2,
    bidirectional=True
)

# Generate random input (batch of 4, sequence length 50)
x = torch.randint(0, 10000, (4, 50))

# Forward pass
logits = model(x)

print(f"Input shape: {x.shape}")       # (4, 50)
print(f"Output shape: {logits.shape}") # (4, 2)
```

## Hints

- LSTM returns `(output, (h_n, c_n))` where `h_n` is the final hidden state
- For bidirectional, final hidden state has shape `(num_layers * 2, batch, hidden_dim)`
- Concatenate the forward and backward hidden states for bidirectional
- The `output` tensor contains hidden states for all time steps
