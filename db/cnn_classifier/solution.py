"""
CNN Image Classifier - Solution

Complete implementation of a CNN for image classification using PyTorch.
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
        
        # First convolutional block
        # Input: 1 channel, Output: 32 channels, Kernel: 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        
        # Second convolutional block
        # Input: 32 channels, Output: 64 channels, Kernel: 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Calculate size: 28 -> conv1(26) -> pool(13) -> conv2(11) -> pool(5)
        # Final: 64 * 5 * 5 = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            
        Returns:
            Class logits of shape (batch, num_classes)
        """
        # First conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the features
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


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
    # Define loss function (Cross Entropy)
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
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
