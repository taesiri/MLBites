"""
Activation Functions from Scratch - Starting Point

Implement various activation functions using PyTorch.
Fill in the TODO sections to complete the implementation.
"""

import math
import torch
import torch.nn as nn


# =============================================================================
# Classic Activation Functions
# =============================================================================

def relu(x: torch.Tensor) -> torch.Tensor:
    """
    ReLU (Rectified Linear Unit): max(0, x)
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    # TODO: Implement ReLU
    pass


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Leaky ReLU: max(negative_slope * x, x)
    
    Args:
        x: Input tensor
        negative_slope: Slope for negative values
        
    Returns:
        Activated tensor
    """
    # TODO: Implement Leaky ReLU
    pass


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid: 1 / (1 + exp(-x))
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor (values in [0, 1])
    """
    # TODO: Implement sigmoid
    pass


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor (values in [-1, 1])
    """
    # TODO: Implement tanh
    pass


# =============================================================================
# Modern Activation Functions
# =============================================================================

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU (Gaussian Error Linear Unit): x * Φ(x)
    
    Used in BERT, GPT-2, and many modern Transformers.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    # TODO: Implement GELU
    # Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
    # Approximate: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    pass


def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish/SiLU: x * sigmoid(x)
    
    Used in EfficientNet and many modern architectures.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    # TODO: Implement Swish
    pass


def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Mish: x * tanh(softplus(x))
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    # TODO: Implement Mish
    # softplus(x) = ln(1 + exp(x))
    pass


# =============================================================================
# Gated Activation Functions
# =============================================================================

class GLU(nn.Module):
    """Gated Linear Unit: splits input and applies gating."""
    
    def __init__(self, dim: int = -1):
        """
        Args:
            dim: Dimension to split on
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split input in half and apply gating.
        
        GLU(x) = a * sigmoid(b) where [a, b] = split(x)
        """
        # TODO: Implement GLU
        pass


class SwiGLU(nn.Module):
    """
    SwiGLU activation used in LLaMA and PaLM.
    
    SwiGLU(x) = Swish(xW) * (xV)
    """
    
    def __init__(self, in_features: int, hidden_features: int):
        """
        Args:
            in_features: Input dimension
            hidden_features: Hidden dimension
        """
        super().__init__()
        # TODO: Create two linear projections (W and V)
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation."""
        # TODO: Implement SwiGLU
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    x = torch.randn(4, 10)
    
    # Test classic activations
    print("Classic Activations:")
    print(f"  ReLU output range: [{relu(x).min():.2f}, {relu(x).max():.2f}]")
    print(f"  Sigmoid output range: [{sigmoid(x).min():.2f}, {sigmoid(x).max():.2f}]")
    
    # Test modern activations
    print("\nModern Activations:")
    print(f"  GELU output range: [{gelu(x).min():.2f}, {gelu(x).max():.2f}]")
    print(f"  Swish output range: [{swish(x).min():.2f}, {swish(x).max():.2f}]")
    
    # Test gated activations
    print("\nGated Activations:")
    swiglu = SwiGLU(10, 20)
    print(f"  SwiGLU output shape: {swiglu(x).shape}")
