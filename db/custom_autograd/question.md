# Custom Autograd Function

## Problem Statement

Implement a **custom autograd function** in PyTorch by extending `torch.autograd.Function`. This allows you to define custom forward and backward passes for operations not in PyTorch.

Your task is to:

1. Implement SiLU/Swish activation with custom backward
2. Understand the autograd.Function interface
3. Save tensors for backward pass correctly
4. Verify gradients match PyTorch's implementation

## Requirements

- Extend `torch.autograd.Function`
- Implement both `forward` and `backward` static methods
- Use `ctx.save_for_backward()` to save tensors
- Support gradient checking

## Function Signature

```python
class SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: SiLU(x) = x * sigmoid(x)."""
        pass
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass: compute gradient."""
        pass
```

## SiLU Gradient

```
SiLU(x) = x * σ(x)

d/dx [x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x))
                = σ(x) * (1 + x * (1 - σ(x)))
                = σ(x) + SiLU(x) * (1 - σ(x))
```

## Example

```python
silu = SiLUFunction.apply  # Get callable

x = torch.randn(10, requires_grad=True)
y = silu(x)
y.sum().backward()

print(x.grad)

# Verify with gradcheck
torch.autograd.gradcheck(silu, (x,))
```

## Hints

- `ctx.save_for_backward(*tensors)` saves tensors for backward
- `ctx.saved_tensors` retrieves them in backward
- Return None for inputs that don't need gradients
- Use `@staticmethod` decorator for both methods
