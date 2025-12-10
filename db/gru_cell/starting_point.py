"""
GRU Cell from Scratch - Starting Point

Implement a GRU cell from scratch.
Fill in the TODO sections to complete the implementation.
"""

import math
import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """GRU cell implemented from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # TODO: Create weight matrices
        # Combined weights for reset and update gates: (2 * hidden_size, input_size + hidden_size)
        # Weights for candidate: (hidden_size, input_size + hidden_size)
        
        # TODO: Create bias terms
        
        # TODO: Initialize weights
        pass
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        GRU cell forward pass.
        
        Args:
            x: Input at current timestep (batch, input_size)
            h: Hidden state from previous timestep (batch, hidden_size)
            
        Returns:
            New hidden state
        """
        # TODO: Compute reset and update gates
        # r = sigmoid(W_r · [h, x])
        # z = sigmoid(W_z · [h, x])
        
        # TODO: Compute candidate hidden state
        # h̃ = tanh(W_h · [r * h, x])
        
        # TODO: Interpolate between old and new
        # h_new = (1 - z) * h + z * h̃
        
        pass


class GRU(nn.Module):
    """Full GRU layer that processes sequences."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # TODO: Create GRU cells for each layer
        pass
    
    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        """
        Process full sequence.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            h: Initial hidden state
            
        Returns:
            output: (batch, seq_len, hidden_size)
            h_n: Final hidden state
        """
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test GRU cell
    gru_cell = GRUCell(input_size=10, hidden_size=20)
    
    x = torch.randn(32, 10)
    h = torch.zeros(32, 20)
    
    h_new = gru_cell(x, h)
    
    print(f"Input shape: {x.shape}")
    print(f"Output hidden shape: {h_new.shape}")
    
    # Compare with PyTorch
    pytorch_gru = nn.GRUCell(10, 20)
    print(f"\nNumber of parameters (ours): {sum(p.numel() for p in gru_cell.parameters())}")
    print(f"Number of parameters (PyTorch): {sum(p.numel() for p in pytorch_gru.parameters())}")
