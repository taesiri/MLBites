"""
Custom Autograd Function - Starting Point

Implement custom autograd functions.
"""

import torch
import torch.nn as nn


class SiLUFunction(torch.autograd.Function):
    """Custom autograd function for SiLU/Swish activation."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SiLU(x) = x * sigmoid(x).
        
        Args:
            ctx: Context object to save tensors for backward
            x: Input tensor
        
        Returns:
            Output tensor
        """
        # TODO: Compute sigmoid
        
        # TODO: Compute SiLU output
        
        # TODO: Save tensors needed for backward
        # ctx.save_for_backward(x, sigmoid)
        
        pass
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass.
        
        Args:
            ctx: Context with saved tensors
            grad_output: Gradient of loss w.r.t. output
        
        Returns:
            Gradient of loss w.r.t. input
        """
        # TODO: Retrieve saved tensors
        # x, sigmoid = ctx.saved_tensors
        
        # TODO: Compute gradient
        # d/dx [x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x))
        
        # TODO: Chain rule: grad_input = grad_output * local_gradient
        
        pass


class ReLUFunction(torch.autograd.Function):
    """Custom ReLU for demonstration."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # TODO: Compute ReLU
        # TODO: Save mask for backward
        pass
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO: Gradient is 1 where x > 0, else 0
        pass


class ClampFunction(torch.autograd.Function):
    """Custom clamp with straight-through gradient."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        # TODO: Clamp values
        pass
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # TODO: Straight-through estimator
        # Return (grad_input, None, None) - None for scalar inputs
        pass


# Create module wrapper
class SiLU(nn.Module):
    """SiLU as nn.Module using custom Function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SiLUFunction.apply(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test custom SiLU
    x = torch.randn(10, requires_grad=True, dtype=torch.float64)
    
    silu = SiLUFunction.apply
    y = silu(x)
    y.sum().backward()
    
    print(f"Input: {x[:5].tolist()}")
    print(f"Output: {y[:5].tolist()}")
    print(f"Gradient: {x.grad[:5].tolist()}")
    
    # Gradient check
    print("\nGradient check:")
    x = torch.randn(5, requires_grad=True, dtype=torch.float64)
    check = torch.autograd.gradcheck(silu, (x,))
    print(f"Gradient check passed: {check}")
    
    # Compare with PyTorch
    print("\nCompare with PyTorch SiLU:")
    x_pt = x.detach().clone().requires_grad_(True)
    y_pt = torch.nn.functional.silu(x_pt)
    y_pt.sum().backward()
    
    print(f"Custom grad: {x.grad}")
    print(f"PyTorch grad: {x_pt.grad}")
    print(f"Match: {torch.allclose(x.grad, x_pt.grad)}")
