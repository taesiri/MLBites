# Mixed Precision Training

## Problem Statement

Implement **Mixed Precision Training** using `torch.cuda.amp`. Mixed precision uses FP16 for faster computation while maintaining FP32 for numerical stability where needed, providing significant speedups on modern GPUs.

Your task is to:

1. Use `autocast` for automatic FP16/FP32 selection
2. Use `GradScaler` to prevent gradient underflow
3. Implement a training loop with mixed precision
4. Handle edge cases and loss scaling

## Requirements

- Use `torch.cuda.amp.autocast` for forward pass
- Use `torch.cuda.amp.GradScaler` for gradient scaling
- Properly handle the backward pass and optimizer step
- Support both training and evaluation modes

## Function Signature

```python
def train_with_mixed_precision(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda',
    epochs: int = 1
) -> list[float]:
    """Train model using mixed precision."""
    pass

def evaluate_with_mixed_precision(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> float:
    """Evaluate model using mixed precision."""
    pass
```

## Example

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Mixed Precision Benefits

| Benefit | Description |
|---------|-------------|
| Speed | 2-3x faster forward/backward on Tensor Cores |
| Memory | ~50% less memory for activations |
| Accuracy | Same accuracy as FP32 when done correctly |

## Hints

- `autocast` automatically chooses precision per operation
- `GradScaler` prevents FP16 gradients from underflowing to zero
- Some operations (softmax, log) always run in FP32 for stability
- Use `with autocast(enabled=False):` to disable locally
