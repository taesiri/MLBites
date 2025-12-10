"""
Activation Functions from Scratch - Solution

Complete implementation of various activation functions.
"""

import math
import torch
import torch.nn as nn


# =============================================================================
# Classic Activation Functions
# =============================================================================

def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU (Rectified Linear Unit): max(0, x)"""
    return torch.maximum(x, torch.zeros_like(x))


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """Leaky ReLU: max(negative_slope * x, x)"""
    return torch.where(x > 0, x, negative_slope * x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + torch.exp(-x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


# =============================================================================
# Modern Activation Functions
# =============================================================================

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU (Gaussian Error Linear Unit): x * Φ(x)
    Using the exact formula with erf.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_approximate(x: torch.Tensor) -> torch.Tensor:
    """GELU using tanh approximation (faster, used in some implementations)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish/SiLU: x * sigmoid(x)"""
    return x * sigmoid(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    """Softplus: ln(1 + exp(x))"""
    # Numerically stable version
    return torch.where(x > 20, x, torch.log1p(torch.exp(x)))


def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish: x * tanh(softplus(x))"""
    return x * torch.tanh(softplus(x))


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU: x if x > 0 else alpha * (exp(x) - 1)"""
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))


def selu(x: torch.Tensor) -> torch.Tensor:
    """SELU: Self-Normalizing ELU with specific alpha and scale."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))


# =============================================================================
# Gated Activation Functions
# =============================================================================

class GLU(nn.Module):
    """Gated Linear Unit: splits input and applies gating."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GLU(x) = a * sigmoid(b) where [a, b] = split(x)"""
        a, b = x.chunk(2, dim=self.dim)
        return a * sigmoid(b)


class SwiGLU(nn.Module):
    """SwiGLU activation used in LLaMA and PaLM."""
    
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w = nn.Linear(in_features, hidden_features, bias=False)
        self.v = nn.Linear(in_features, hidden_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU(x) = Swish(xW) * (xV)"""
        return swish(self.w(x)) * self.v(x)


class GeGLU(nn.Module):
    """GeGLU: GELU-gated linear unit."""
    
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w = nn.Linear(in_features, hidden_features, bias=False)
        self.v = nn.Linear(in_features, hidden_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GeGLU(x) = GELU(xW) * (xV)"""
        return gelu(self.w(x)) * self.v(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    x = torch.randn(4, 10)
    
    # Test classic activations
    print("Classic Activations:")
    print(f"  ReLU output range: [{relu(x).min():.2f}, {relu(x).max():.2f}]")
    print(f"  Leaky ReLU output range: [{leaky_relu(x).min():.2f}, {leaky_relu(x).max():.2f}]")
    print(f"  Sigmoid output range: [{sigmoid(x).min():.2f}, {sigmoid(x).max():.2f}]")
    print(f"  Tanh output range: [{tanh(x).min():.2f}, {tanh(x).max():.2f}]")
    
    # Test modern activations
    print("\nModern Activations:")
    print(f"  GELU output range: [{gelu(x).min():.2f}, {gelu(x).max():.2f}]")
    print(f"  Swish output range: [{swish(x).min():.2f}, {swish(x).max():.2f}]")
    print(f"  Mish output range: [{mish(x).min():.2f}, {mish(x).max():.2f}]")
    
    # Verify against PyTorch
    print("\nVerification against PyTorch:")
    print(f"  GELU matches: {torch.allclose(gelu(x), torch.nn.functional.gelu(x), atol=1e-5)}")
    print(f"  Swish matches: {torch.allclose(swish(x), torch.nn.functional.silu(x), atol=1e-5)}")
    
    # Test gated activations
    print("\nGated Activations:")
    swiglu = SwiGLU(10, 20)
    geglu = GeGLU(10, 20)
    print(f"  SwiGLU output shape: {swiglu(x).shape}")
    print(f"  GeGLU output shape: {geglu(x).shape}")
