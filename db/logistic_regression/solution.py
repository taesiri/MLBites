"""
Logistic Regression with PyTorch - Solution

Complete implementation of a logistic regression classifier for binary classification.
"""

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Logistic regression classifier for binary classification."""
    
    def __init__(self, input_dim: int):
        """
        Initialize the logistic regression model.
        
        Args:
            input_dim: Number of input features
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sigmoid activation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probabilities tensor of shape (batch_size, 1)
        """
        return torch.sigmoid(self.linear(x))


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.1
) -> list[float]:
    """
    Train the logistic regression model.
    
    Args:
        model: The logistic regression model
        X: Input features tensor
        y: Target labels tensor (0 or 1)
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of loss values for each epoch
    """
    # Define the loss function (Binary Cross Entropy)
    criterion = nn.BCELoss()
    
    # Define the optimizer (SGD)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass - compute probabilities
        y_pred = model(X)
        
        # Compute the loss
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Append loss to losses list
        losses.append(loss.item())
    
    return losses


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Make binary predictions.
    
    Args:
        model: Trained logistic regression model
        X: Input features tensor
        
    Returns:
        Binary predictions tensor (0 or 1)
    """
    with torch.no_grad():
        probabilities = model(X)
        predictions = (probabilities >= 0.5).float()
    return predictions


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_pred: Predicted labels
        y_true: True labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    correct = (y_pred == y_true).sum().item()
    total = y_true.numel()
    return correct / total


if __name__ == "__main__":
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
    
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Accuracy: {acc:.2%}")
