"""
Mixed Precision Training - Starting Point

Implement mixed precision training using torch.cuda.amp.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train_with_mixed_precision(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda',
    epochs: int = 1
) -> list[float]:
    """
    Train model using mixed precision.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs
        
    Returns:
        List of average losses per epoch
    """
    model.train()
    epoch_losses = []
    
    # TODO: Create GradScaler for gradient scaling
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # TODO: Zero gradients
            
            # TODO: Forward pass with autocast
            # with autocast():
            #     outputs = model(inputs)
            #     loss = criterion(outputs, targets)
            
            # TODO: Backward pass with scaled gradients
            # scaler.scale(loss).backward()
            
            # TODO: Optimizer step with scaler
            # scaler.step(optimizer)
            
            # TODO: Update scaler
            # scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return epoch_losses


def evaluate_with_mixed_precision(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> float:
    """
    Evaluate model using mixed precision.
    
    Note: GradScaler is not needed for evaluation.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # TODO: Forward pass with autocast
            # with autocast():
            #     outputs = model(inputs)
            #     loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def check_tensor_dtype(model: nn.Module, x: torch.Tensor, device: str = 'cuda'):
    """Check tensor dtypes during autocast to understand what's happening."""
    model.to(device)
    x = x.to(device)
    
    print("Outside autocast:")
    print(f"  Input dtype: {x.dtype}")
    
    # TODO: Check dtype inside autocast
    # with autocast():
    #     print("Inside autocast:")
    #     print(f"  Input dtype: {x.dtype}")  # Still FP32
    #     ...
    
    pass


if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("Note: Mixed precision is most beneficial on CUDA-enabled GPUs")
    
    # Create dummy data
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Train with mixed precision
    print("\nTraining with mixed precision...")
    losses = train_with_mixed_precision(
        model, loader, optimizer, criterion, 
        device=device, epochs=3
    )
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
