# Activation Functions from Scratch

## Problem Statement

Implement various **activation functions** from scratch using PyTorch. Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

Your task is to implement:

1. **Classic activations**: ReLU, Leaky ReLU, Sigmoid, Tanh
2. **Modern activations**: GELU, SiLU/Swish, Mish
3. **Gated activations**: GLU (Gated Linear Unit), SwiGLU

## Requirements

- Do **NOT** use built-in activation functions like `F.relu`, `F.gelu`, etc.
- Implement both forward pass and understand the derivatives
- Support both functional and module-based implementations

## Function Signatures

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU: max(0, x)"""
    pass

def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """Leaky ReLU: max(negative_slope * x, x)"""
    pass

def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU: x * Φ(x) where Φ is the CDF of standard normal"""
    pass

def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish/SiLU: x * sigmoid(x)"""
    pass

def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish: x * tanh(softplus(x))"""
    pass

class SwiGLU(nn.Module):
    """SwiGLU: Swish(xW) * (xV) - used in LLaMA"""
    pass
```

## Example

```python
import torch

x = torch.randn(4, 10)

# Classic activations
print(f"ReLU: {relu(x).shape}")
print(f"Sigmoid: {sigmoid(x).shape}")

# Modern activations
print(f"GELU: {gelu(x).shape}")
print(f"Swish: {swish(x).shape}")

# Gated activation
swiglu = SwiGLU(10, 20)
print(f"SwiGLU: {swiglu(x).shape}")
```

## Activation Formulas

| Activation | Formula |
|------------|---------|
| ReLU | `max(0, x)` |
| Leaky ReLU | `max(αx, x)` |
| Sigmoid | `1 / (1 + exp(-x))` |
| Tanh | `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` |
| GELU | `x * 0.5 * (1 + erf(x / √2))` |
| Swish/SiLU | `x * sigmoid(x)` |
| Mish | `x * tanh(softplus(x))` |
| SwiGLU | `Swish(xW) ⊙ (xV)` |

## Hints

- GELU can be approximated: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
- Softplus: `ln(1 + exp(x))`
- Gated activations split or double the hidden dimension
