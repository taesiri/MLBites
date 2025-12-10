"""
CNN Image Classifier - Starting Point

Implement a CNN for image classification using PyTorch.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize CNN classifier.
        
        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        
        # TODO: Define first convolutional block
        # Input: 1 channel, Output: 32 channels, Kernel: 3x3
        
        # TODO: Define second convolutional block
        # Input: 32 channels, Output: 64 channels, Kernel: 3x3
        
        # TODO: Define pooling layer
        
        # TODO: Define fully connected layers
        # Calculate the size after convolutions (input: 28x28)
        # After conv1 + pool: 28 -> 26 -> 13
        # After conv2 + pool: 13 -> 11 -> 5
        # Final feature size: 64 * 5 * 5 = 1600
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            
        Returns:
            Class logits of shape (batch, num_classes)
        """
        # TODO: First conv block (conv -> relu -> pool)
        
        # TODO: Second conv block (conv -> relu -> pool)
        
        # TODO: Flatten the features
        
        # TODO: Fully connected layers
        
        pass


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    lr: float = 0.001
) -> list[float]:
    """
    Train the CNN model.
    
    Args:
        model: The CNN model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of average loss per epoch
    """
    # TODO: Define loss function (Cross Entropy)
    
    # TODO: Define optimizer (Adam)
    
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # TODO: Zero gradients
            
            # TODO: Forward pass
            
            # TODO: Compute loss
            
            # TODO: Backward pass
            
            # TODO: Update weights
            
            pass
        
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return epoch_losses


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create model
    model = CNNClassifier(num_classes=10)
    
    # Test with random MNIST-like input
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
