# Linear Regression with PyTorch

## Problem Statement

Implement a simple **Linear Regression** model using PyTorch. Your task is to:

1. Create a linear regression model class that inherits from `nn.Module`
2. Implement the forward pass
3. Train the model on synthetic data
4. Make predictions

## Requirements

- Use `torch.nn.Linear` for the linear layer
- Implement Mean Squared Error (MSE) loss
- Use Stochastic Gradient Descent (SGD) optimizer
- Train for a specified number of epochs

## Function Signature

```python
class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """Initialize the linear regression model."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01
) -> list[float]:
    """Train the model and return the loss history."""
    pass
```

## Example

```python
import torch

# Generate synthetic data: y = 2x + 3
X = torch.randn(100, 1)
y = 2 * X + 3 + torch.randn(100, 1) * 0.1  # Add some noise

# Create and train model
model = LinearRegression(input_dim=1, output_dim=1)
losses = train_model(model, X, y, epochs=100, lr=0.01)

# The model should learn weights close to 2 and bias close to 3
print(f"Weight: {model.linear.weight.item():.2f}")  # ~2.0
print(f"Bias: {model.linear.bias.item():.2f}")      # ~3.0
```

## Hints

- Remember to zero the gradients before each backward pass
- Use `loss.backward()` to compute gradients
- Use `optimizer.step()` to update parameters
