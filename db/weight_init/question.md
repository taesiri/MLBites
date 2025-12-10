# Weight Initialization for CNNs

## Problem Statement

Implement common **weight initialization** strategies for neural networks. Proper initialization is crucial for training deep networks effectively.

Your task is to:

1. Implement Xavier/Glorot initialization (for tanh/sigmoid)
2. Implement Kaiming/He initialization (for ReLU)
3. Apply initialization to different layer types
4. Understand when to use each method

## Requirements

- Implement both uniform and normal variants
- Apply correctly to Conv2d and Linear layers
- Handle bias initialization
- Create a function to initialize entire models

## Function Signature

```python
def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0):
    """Xavier uniform initialization."""
    pass

def kaiming_normal_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    """Kaiming normal initialization."""
    pass

def init_weights(model: nn.Module, init_type: str = 'kaiming'):
    """Initialize all weights in a model."""
    pass
```

## Initialization Formulas

**Xavier (for tanh/sigmoid):**
```
Var(W) = 2 / (fan_in + fan_out)
Uniform: [-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))]
```

**Kaiming (for ReLU):**
```
Var(W) = 2 / fan_in  (or fan_out)
Normal: N(0, √(2/fan_in))
```

## Example

```python
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),
)

init_weights(model, init_type='kaiming')
```

## Hints

- `fan_in` = number of input connections
- `fan_out` = number of output connections
- For Conv2d: fan_in = in_channels * kernel_h * kernel_w
- Use `nn.init.calculate_gain()` for activation-specific gains
