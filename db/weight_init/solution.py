"""
Weight Initialization - Solution

Complete implementation of weight initialization strategies.
"""

import math
import torch
import torch.nn as nn


def calculate_fan_in_out(tensor: torch.Tensor) -> tuple[int, int]:
    """Calculate fan_in and fan_out for a weight tensor."""
    dimensions = tensor.dim()
    
    if dimensions < 2:
        raise ValueError("Fan in/out requires at least 2D tensor")
    
    if dimensions == 2:  # Linear layer
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:  # Conv layers
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = tensor[0][0].numel()
        
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out


def calculate_gain(nonlinearity: str, param=None) -> float:
    """Calculate gain for different activations."""
    gains = {
        'linear': 1.0,
        'sigmoid': 1.0,
        'tanh': 5/3,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01)**2)),
        'selu': 3/4,
    }
    return gains.get(nonlinearity, 1.0)


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0):
    """Xavier/Glorot uniform initialization."""
    fan_in, fan_out = calculate_fan_in_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    
    with torch.no_grad():
        tensor.uniform_(-a, a)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0):
    """Xavier/Glorot normal initialization."""
    fan_in, fan_out = calculate_fan_in_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def kaiming_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    """Kaiming/He uniform initialization."""
    fan_in, fan_out = calculate_fan_in_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


def kaiming_normal_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu'):
    """Kaiming/He normal initialization."""
    fan_in, fan_out = calculate_fan_in_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def init_weights(model: nn.Module, init_type: str = 'kaiming'):
    """Initialize all weights in a model."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            if init_type == 'kaiming':
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                xavier_uniform_(m.weight)
            else:
                nn.init.normal_(m.weight, 0, 0.01)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.Linear):
            if init_type == 'kaiming':
                kaiming_normal_(m.weight)
            elif init_type == 'xavier':
                xavier_uniform_(m.weight)
            else:
                nn.init.normal_(m.weight, 0, 0.01)
            
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Test initialization
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10)
    )
    
    print("Before initialization:")
    print(f"Conv1 weight std: {model[0].weight.std():.4f}")
    
    init_weights(model, init_type='kaiming')
    
    print("\nAfter Kaiming initialization:")
    print(f"Conv1 weight std: {model[0].weight.std():.4f}")
    
    # Compare with PyTorch built-in
    conv = nn.Conv2d(3, 64, 3)
    print(f"\nPyTorch default std: {conv.weight.std():.4f}")
    
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    print(f"PyTorch kaiming std: {conv.weight.std():.4f}")
