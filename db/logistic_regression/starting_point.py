"""
Logistic Regression with PyTorch - Starting Point

Implement a logistic regression classifier for binary classification.
Fill in the TODO sections to complete the implementation.
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
        # TODO: Create a linear layer (output dimension should be 1 for binary classification)
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sigmoid activation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probabilities tensor of shape (batch_size, 1)
        """
        # TODO: Apply linear transformation followed by sigmoid activation
        pass


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
    # TODO: Define the loss function (Binary Cross Entropy)
    
    # TODO: Define the optimizer (SGD)
    
    losses = []
    
    for epoch in range(epochs):
        # TODO: Forward pass - compute probabilities
        
        # TODO: Compute the loss
        
        # TODO: Backward pass (zero gradients, compute gradients, update parameters)
        
        # TODO: Append loss to losses list
        pass
    
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
    # TODO: Get probabilities from model and threshold at 0.5
    pass


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_pred: Predicted labels
        y_true: True labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # TODO: Compute the fraction of correct predictions
    pass


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
