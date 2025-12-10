"""
LSTM Cell from Scratch - Starting Point

Implement an LSTM cell from scratch.
Fill in the TODO sections to complete the implementation.
"""

import math
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """LSTM cell implemented from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize LSTM cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # TODO: Create weight matrices
        # Option 1: Separate weights for each gate
        # Option 2: Combined weights (more efficient)
        # W_ih: (4 * hidden_size, input_size)
        # W_hh: (4 * hidden_size, hidden_size)
        
        # TODO: Create bias terms
        
        # TODO: Initialize weights
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LSTM cell forward pass.
        
        Args:
            x: Input at current timestep (batch, input_size)
            state: Tuple of (h, c) from previous timestep
            
        Returns:
            Tuple of new (h, c)
        """
        h, c = state
        
        # TODO: Compute gates
        # gates = x @ W_ih.T + h @ W_hh.T + bias
        
        # TODO: Split into forget, input, cell, output gates
        # i, f, g, o = gates.chunk(4, dim=1)
        
        # TODO: Apply activations
        # i, f, o = sigmoid(...)
        # g = tanh(...)
        
        # TODO: Compute new cell state
        # c_new = f * c + i * g
        
        # TODO: Compute new hidden state
        # h_new = o * tanh(c_new)
        
        pass


class LSTM(nn.Module):
    """Full LSTM layer that processes sequences."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # TODO: Create LSTM cells for each layer
        pass
    
    def forward(self, x: torch.Tensor, state: tuple = None):
        """
        Process full sequence.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            state: Initial state tuple (h_0, c_0)
            
        Returns:
            output: (batch, seq_len, hidden_size)
            (h_n, c_n): Final states
        """
        # TODO: Initialize state if None
        
        # TODO: Process sequence step by step
        
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test LSTM cell
    lstm_cell = LSTMCell(input_size=10, hidden_size=20)
    
    x = torch.randn(32, 10)
    h = torch.zeros(32, 20)
    c = torch.zeros(32, 20)
    
    h_new, c_new = lstm_cell(x, (h, c))
    
    print(f"Input shape: {x.shape}")
    print(f"Output hidden shape: {h_new.shape}")
    print(f"Output cell shape: {c_new.shape}")
    
    # Compare with PyTorch
    pytorch_lstm = nn.LSTMCell(10, 20)
    print(f"\nNumber of parameters (ours): {sum(p.numel() for p in lstm_cell.parameters())}")
    print(f"Number of parameters (PyTorch): {sum(p.numel() for p in pytorch_lstm.parameters())}")
