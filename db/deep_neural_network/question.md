# Deep Neural Network (MLP)

## Problem Statement

Implement a **Deep Neural Network** (Multi-Layer Perceptron) from scratch using PyTorch. MLPs are the foundation of deep learning, consisting of multiple fully connected layers with non-linear activations.

Your task is to:

1. Build a flexible MLP with configurable depth and width
2. Add dropout for regularization
3. Use proper weight initialization
4. Train on a classification or regression task

## Requirements

- Support variable number of hidden layers
- Use ReLU activation (or configurable)
- Include dropout for regularization
- Apply proper weight initialization

## Function Signature

```python
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.5,
        activation: str = 'relu'
    ):
        """Create a deep neural network."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

## Example

```python
# Create a 3-hidden-layer MLP
model = MLP(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    dropout=0.5
)

x = torch.randn(32, 784)
output = model(x)  # (32, 10)
```

## Architecture

```
Input(784) -> FC(512) -> ReLU -> Dropout
          -> FC(256) -> ReLU -> Dropout
          -> FC(128) -> ReLU -> Dropout
          -> FC(10) -> Output
```

## Hints

- Use `nn.Sequential` or `nn.ModuleList` for dynamic layers
- Xavier/Kaiming initialization helps with deep networks
- Dropout should only be applied during training
- Consider using batch normalization between layers
