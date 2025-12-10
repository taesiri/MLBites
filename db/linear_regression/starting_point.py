"""
Linear Regression with PyTorch - Starting Point

Implement a simple linear regression model using PyTorch.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    """Simple linear regression model."""
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the linear regression model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super().__init__()
        # TODO: Create a linear layer using nn.Linear
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # TODO: Implement the forward pass
        pass


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01
) -> list[float]:
    """
    Train the linear regression model.
    
    Args:
        model: The linear regression model
        X: Input features tensor
        y: Target values tensor
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of loss values for each epoch
    """
    # TODO: Define the loss function (MSE)
    
    # TODO: Define the optimizer (SGD)
    
    losses = []
    
    for epoch in range(epochs):
        # TODO: Forward pass - compute predictions
        
        # TODO: Compute the loss
        
        # TODO: Backward pass
        # Hint: Don't forget to zero the gradients!
        
        # TODO: Update parameters
        
        # TODO: Append loss to losses list
        pass
    
    return losses


if __name__ == "__main__":
    # Generate synthetic data: y = 2x + 3
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y = 2 * X + 3 + torch.randn(100, 1) * 0.1
    
    # Create and train model
    model = LinearRegression(input_dim=1, output_dim=1)
    losses = train_model(model, X, y, epochs=100, lr=0.01)
    
    # Print results
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Learned weight: {model.linear.weight.item():.2f}")
    print(f"Learned bias: {model.linear.bias.item():.2f}")
