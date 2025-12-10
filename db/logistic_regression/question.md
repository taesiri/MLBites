# Logistic Regression with PyTorch

## Problem Statement

Implement a **Logistic Regression** classifier using PyTorch for binary classification. Your task is to:

1. Create a logistic regression model class that inherits from `nn.Module`
2. Implement the forward pass with sigmoid activation
3. Train the model using Binary Cross Entropy loss
4. Make predictions and compute accuracy

## Requirements

- Use `torch.nn.Linear` for the linear layer
- Apply sigmoid activation for binary classification
- Use Binary Cross Entropy (BCE) loss
- Use Stochastic Gradient Descent (SGD) optimizer

## Function Signature

```python
class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int):
        """Initialize the logistic regression model."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation."""
        pass

def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.1
) -> list[float]:
    """Train the model and return the loss history."""
    pass

def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Make binary predictions (0 or 1)."""
    pass

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute classification accuracy."""
    pass
```

## Example

```python
import torch

# Generate synthetic binary classification data
torch.manual_seed(42)
X = torch.randn(200, 2)
y = ((X[:, 0] + X[:, 1]) > 0).float().unsqueeze(1)

# Create and train model
model = LogisticRegression(input_dim=2)
losses = train_model(model, X, y, epochs=100, lr=0.1)

# Make predictions
predictions = predict(model, X)
acc = accuracy(predictions, y)
print(f"Accuracy: {acc:.2%}")  # Should be > 90%
```

## Hints

- Sigmoid function outputs values between 0 and 1
- Use `torch.sigmoid()` or `nn.Sigmoid()`
- For predictions, threshold at 0.5: values >= 0.5 → 1, else → 0
- Use `nn.BCELoss()` for binary cross entropy
