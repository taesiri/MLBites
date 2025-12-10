"""
Linear Regression with PyTorch - Solution

Complete implementation of a simple linear regression model using PyTorch.
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
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.linear(x)


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
    # Define the loss function (MSE)
    criterion = nn.MSELoss()
    
    # Define the optimizer (SGD)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass - compute predictions
        y_pred = model(X)
        
        # Compute the loss
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Compute gradients
        
        # Update parameters
        optimizer.step()
        
        # Append loss to losses list
        losses.append(loss.item())
    
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
