"""
Save and Load PyTorch Models - Starting Point

Learn to save and load models properly.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
    """
    # TODO: Create checkpoint dictionary with:
    # - 'epoch': epoch number
    # - 'model_state_dict': model.state_dict()
    # - 'optimizer_state_dict': optimizer.state_dict()
    # - 'loss': loss value
    
    # TODO: Save with torch.save
    
    pass


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> dict:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    # TODO: Load checkpoint with torch.load
    # Use map_location for device handling
    
    # TODO: Load model state dict
    
    # TODO: Load optimizer state dict if provided
    
    pass


def save_for_inference(model: nn.Module, path: str):
    """
    Save model for inference only (just the weights).
    
    Args:
        model: Model to save
        path: Path to save weights
    """
    # TODO: Save only the model's state dict
    pass


def load_for_inference(model: nn.Module, path: str, device: str = 'cpu'):
    """
    Load model for inference.
    
    Args:
        model: Model to load weights into
        path: Path to weights file
        device: Device to load to
    """
    # TODO: Load state dict with device mapping
    
    # TODO: Load into model
    
    # TODO: Set to eval mode
    
    pass


if __name__ == "__main__":
    import os
    
    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate some training
    print("Simulating training...")
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    for epoch in range(5):
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Save checkpoint
    print("\nSaving checkpoint...")
    save_checkpoint(model, optimizer, epoch=5, loss=loss.item(), path='/tmp/checkpoint.pt')
    
    # Create new model and load checkpoint
    print("Loading checkpoint into new model...")
    new_model = SimpleModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    
    checkpoint = load_checkpoint('/tmp/checkpoint.pt', new_model, new_optimizer)
    print(f"Loaded from epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
    
    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
        if not torch.equal(p1, p2):
            print(f"Mismatch in {n1}!")
            break
    else:
        print("All weights match!")
    
    # Clean up
    os.remove('/tmp/checkpoint.pt')
