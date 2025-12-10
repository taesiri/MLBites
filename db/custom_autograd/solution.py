"""
Custom Autograd Function - Solution

Complete implementation of custom autograd functions.
"""

import torch
import torch.nn as nn


class SiLUFunction(torch.autograd.Function):
    """Custom autograd function for SiLU/Swish activation."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        sigmoid = torch.sigmoid(x)
        output = x * sigmoid
        ctx.save_for_backward(x, sigmoid)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, sigmoid = ctx.saved_tensors
        # d/dx [x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x))
        grad_input = sigmoid + x * sigmoid * (1 - sigmoid)
        return grad_output * grad_input


class ReLUFunction(torch.autograd.Function):
    """Custom ReLU."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input


class LeakyReLUFunction(torch.autograd.Function):
    """Custom Leaky ReLU."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.negative_slope = negative_slope
        return torch.where(x >= 0, x, negative_slope * x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        grad_input = torch.where(x >= 0, grad_output, ctx.negative_slope * grad_output)
        return grad_input, None  # None for negative_slope


class GELUFunction(torch.autograd.Function):
    """Custom GELU activation."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        import math
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        ctx.save_for_backward(x, cdf)
        return x * cdf
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        import math
        x, cdf = ctx.saved_tensors
        pdf = torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        grad_input = cdf + x * pdf
        return grad_output * grad_input


class ClampFunction(torch.autograd.Function):
    """Clamp with straight-through gradient."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        return x.clamp(min=min_val, max=max_val)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: pass gradient through unchanged
        return grad_output, None, None


# Module wrappers
class SiLU(nn.Module):
    def forward(self, x):
        return SiLUFunction.apply(x)


class CustomReLU(nn.Module):
    def forward(self, x):
        return ReLUFunction.apply(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test SiLU
    x = torch.randn(10, requires_grad=True, dtype=torch.float64)
    silu = SiLUFunction.apply
    y = silu(x)
    y.sum().backward()
    
    print(f"Input: {x[:5].tolist()}")
    print(f"Gradient: {x.grad[:5].tolist()}")
    
    # Gradient check
    print("\nGradient checks:")
    for name, func in [("SiLU", SiLUFunction.apply), 
                       ("ReLU", ReLUFunction.apply),
                       ("GELU", GELUFunction.apply)]:
        x = torch.randn(5, requires_grad=True, dtype=torch.float64)
        check = torch.autograd.gradcheck(func, (x,))
        print(f"  {name}: {check}")
    
    # Compare with PyTorch
    print("\nCompare SiLU with PyTorch:")
    x = torch.randn(5, requires_grad=True, dtype=torch.float64)
    x_pt = x.detach().clone().requires_grad_(True)
    
    y_custom = SiLUFunction.apply(x)
    y_custom.sum().backward()
    
    y_pt = torch.nn.functional.silu(x_pt)
    y_pt.sum().backward()
    
    print(f"Outputs match: {torch.allclose(y_custom, y_pt)}")
    print(f"Gradients match: {torch.allclose(x.grad, x_pt.grad)}")
