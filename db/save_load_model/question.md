# Save and Load PyTorch Models

## Problem Statement

Learn to properly **save and load PyTorch models** for inference and resuming training. This is essential for deploying models and creating checkpoints.

Your task is to:

1. Save and load model state_dict (recommended approach)
2. Create training checkpoints with optimizer state
3. Save the entire model (understand the limitations)
4. Handle device mapping when loading

## Requirements

- Use `torch.save` and `torch.load`
- Save state_dict for flexibility
- Include optimizer state for training resumption
- Handle CPU/GPU device mapping

## Function Signatures

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save a training checkpoint."""
    pass

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> dict:
    """Load a training checkpoint."""
    pass

def save_for_inference(model: nn.Module, path: str):
    """Save model for inference only."""
    pass

def load_for_inference(model: nn.Module, path: str, device: str = 'cpu'):
    """Load model for inference."""
    pass
```

## Example

```python
# During training - save checkpoint
save_checkpoint(model, optimizer, epoch=10, loss=0.5, path='checkpoint.pt')

# Resume training
checkpoint = load_checkpoint('checkpoint.pt', model, optimizer)
start_epoch = checkpoint['epoch'] + 1

# For deployment - save for inference
save_for_inference(model, 'model.pt')

# Load for inference
load_for_inference(model, 'model.pt', device='cuda')
model.eval()
```

## Hints

- `model.state_dict()` returns a dictionary of parameters
- `model.load_state_dict(state_dict)` loads parameters
- Use `map_location` parameter in `torch.load` for device mapping
- Always call `model.eval()` for inference
- Consider using `.pt` or `.pth` extension for consistency
