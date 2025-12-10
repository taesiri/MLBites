"""
Mixed Precision Training - Solution

Complete implementation of mixed precision training using torch.cuda.amp.
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
    """Train model using mixed precision."""
    model.train()
    epoch_losses = []
    
    # Create GradScaler for gradient scaling
    scaler = GradScaler()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Optimizer step with scaler (handles unscaling internally)
            scaler.step(optimizer)
            
            # Update scaler for next iteration
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return epoch_losses


def train_with_gradient_clipping(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_grad_norm: float = 1.0,
    device: str = 'cuda',
    epochs: int = 1
) -> list[float]:
    """Train with mixed precision AND gradient clipping."""
    model.train()
    epoch_losses = []
    scaler = GradScaler()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients (now in FP32)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Step with scaler (skips if gradients are inf/nan)
            scaler.step(optimizer)
            scaler.update()
            
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
    """Evaluate model using mixed precision."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Autocast for faster inference
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def check_tensor_dtype(model: nn.Module, x: torch.Tensor, device: str = 'cuda'):
    """Check tensor dtypes during autocast."""
    model.to(device)
    x = x.to(device)
    
    print("Outside autocast:")
    print(f"  Input dtype: {x.dtype}")
    print(f"  Weight dtype: {model.fc1.weight.dtype}")
    
    with autocast():
        print("\nInside autocast:")
        print(f"  Input dtype: {x.dtype}")  # Still FP32 (inputs not auto-converted)
        
        # Linear operations become FP16
        hidden = model.fc1(x)
        print(f"  After Linear: {hidden.dtype}")  # FP16 (automatic)
        
        # But softmax stays FP32 for stability
        probs = torch.softmax(hidden, dim=-1)
        print(f"  After Softmax: {probs.dtype}")  # FP32 (automatic)


if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("Note: Mixed precision is most beneficial on CUDA-enabled GPUs")
        print("Running on CPU for demonstration...\n")
    
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
    print("Training with mixed precision...")
    losses = train_with_mixed_precision(
        model, loader, optimizer, criterion, 
        device=device, epochs=3
    )
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    
    # Check dtypes
    if device == 'cuda':
        print("\nChecking tensor dtypes:")
        check_tensor_dtype(model, torch.randn(1, 784), device)
