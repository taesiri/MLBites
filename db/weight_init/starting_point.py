"""
Weight Initialization - Starting Point

Implement weight initialization strategies for neural networks.
"""

import math
import torch
import torch.nn as nn


def calculate_fan_in_out(tensor: torch.Tensor) -> tuple[int, int]:
    """
    Calculate fan_in and fan_out for a weight tensor.
    
    For Linear: (in_features, out_features)
    For Conv2d: (in_channels * kH * kW, out_channels * kH * kW)
    """
    # TODO: Calculate fan_in and fan_out based on tensor dimensions
    pass


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0):
    """
    Xavier/Glorot uniform initialization.
    
    Designed for tanh/sigmoid activations.
    W ~ U[-a, a] where a = gain * sqrt(6 / (fan_in + fan_out))
    """
    # TODO: Calculate fan_in, fan_out
    # TODO: Compute bound a
    # TODO: Fill tensor with uniform values
    pass


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0):
    """Xavier/Glorot normal initialization."""
    # TODO: W ~ N(0, gain * sqrt(2 / (fan_in + fan_out)))
    pass


def kaiming_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    """
    Kaiming/He uniform initialization.
    
    Designed for ReLU activations.
    """
    # TODO: Calculate fan based on mode
    # TODO: Get gain for nonlinearity
    # TODO: Compute bound and fill
    pass


def kaiming_normal_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    """Kaiming/He normal initialization."""
    # TODO: W ~ N(0, sqrt(2 / fan))
    pass


def init_weights(model: nn.Module, init_type: str = 'kaiming'):
    """
    Initialize all weights in a model.
    
    Args:
        model: Neural network module
        init_type: 'xavier', 'kaiming', or 'normal'
    """
    # TODO: Iterate through modules
    # TODO: Apply appropriate initialization based on layer type
    pass


if __name__ == "__main__":
    # Test initialization
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 32 * 32, 10)
    )
    
    print("Before initialization:")
    print(f"Conv1 weight std: {model[0].weight.std():.4f}")
    
    init_weights(model, init_type='kaiming')
    
    print("\nAfter Kaiming initialization:")
    print(f"Conv1 weight std: {model[0].weight.std():.4f}")
