"""
Save and Load PyTorch Models - Solution

Complete implementation of model saving and loading.
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
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> dict:
    """Load a training checkpoint."""
    # Load with device mapping
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_for_inference(model: nn.Module, path: str):
    """Save model weights for inference only."""
    torch.save(model.state_dict(), path)


def load_for_inference(model: nn.Module, path: str, device: str = 'cpu'):
    """Load model for inference."""
    # Load state dict with device mapping
    state_dict = torch.load(path, map_location=device)
    
    # Load into model
    model.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()


def save_entire_model(model: nn.Module, path: str):
    """
    Save the entire model (not recommended for production).
    
    This saves the model class definition along with weights.
    Requires the exact same class to be available when loading.
    """
    torch.save(model, path)


def load_entire_model(path: str, device: str = 'cpu') -> nn.Module:
    """Load an entire model (requires exact same class definition)."""
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def export_to_torchscript(model: nn.Module, example_input: torch.Tensor, path: str):
    """
    Export model to TorchScript for production deployment.
    
    TorchScript models can be loaded in C++ and don't require Python.
    """
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(path)


def load_torchscript(path: str, device: str = 'cpu') -> torch.jit.ScriptModule:
    """Load a TorchScript model."""
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model


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
    
    # Test TorchScript export
    print("\nExporting to TorchScript...")
    export_to_torchscript(model, torch.randn(1, 10), '/tmp/model_scripted.pt')
    
    scripted_model = load_torchscript('/tmp/model_scripted.pt')
    scripted_output = scripted_model(x)
    original_output = model(x)
    print(f"TorchScript output matches: {torch.allclose(scripted_output, original_output)}")
    
    # Clean up
    os.remove('/tmp/checkpoint.pt')
    os.remove('/tmp/model_scripted.pt')
