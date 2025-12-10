"""
RNN Cell from Scratch - Starting Point

Implement RNN and LSTM cells without using built-in recurrent modules.
Fill in the TODO sections to complete the implementation.
"""

import math
import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    """Vanilla RNN cell implemented from scratch."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize vanilla RNN cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # TODO: Create weight matrices for input-to-hidden
        # w_ih: (hidden_size, input_size)
        
        # TODO: Create weight matrices for hidden-to-hidden
        # w_hh: (hidden_size, hidden_size)
        
        # TODO: Create bias terms
        # b_ih: (hidden_size,)
        # b_hh: (hidden_size,)
        
        # TODO: Initialize weights (use Xavier/Glorot initialization)
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Single RNN cell forward step.
        
        Args:
            x: Input at current timestep (batch, input_size)
            h: Hidden state from previous timestep (batch, hidden_size)
            
        Returns:
            New hidden state (batch, hidden_size)
        """
        # TODO: Compute new hidden state
        # h_new = tanh(x @ W_ih.T + h @ W_hh.T + b_ih + b_hh)
        
        pass


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
        
        # TODO: Create weight matrices for all gates (input, forget, cell, output)
        # Option 1: Separate weights for each gate
        # Option 2: Combined weight matrix for efficiency
        # w_ih: (4 * hidden_size, input_size) - combines all input weights
        # w_hh: (4 * hidden_size, hidden_size) - combines all hidden weights
        
        # TODO: Create bias terms
        
        # TODO: Initialize weights
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single LSTM cell forward step.
        
        Args:
            x: Input at current timestep (batch, input_size)
            state: Tuple of (h, c) from previous timestep
                h: hidden state (batch, hidden_size)
                c: cell state (batch, hidden_size)
            
        Returns:
            Tuple of new (h, c)
        """
        h, c = state
        
        # TODO: Compute all gates
        # gates = x @ W_ih.T + h @ W_hh.T + bias
        
        # TODO: Split gates into i, f, g, o
        
        # TODO: Apply activations
        # i, f, o = sigmoid(...)
        # g = tanh(...)
        
        # TODO: Compute new cell state
        # c_new = f * c + i * g
        
        # TODO: Compute new hidden state
        # h_new = o * tanh(c_new)
        
        pass


def process_sequence(cell, x_seq, h_init, c_init=None):
    """
    Process a full sequence through an RNN/LSTM cell.
    
    Args:
        cell: RNN or LSTM cell
        x_seq: Input sequence (batch, seq_len, input_size)
        h_init: Initial hidden state
        c_init: Initial cell state (for LSTM only)
        
    Returns:
        List of hidden states for each timestep
    """
    # TODO: Iterate through sequence and collect hidden states
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch_size = 4
    seq_len = 10
    input_size = 8
    hidden_size = 16
    
    # Test Vanilla RNN
    print("Testing Vanilla RNN Cell...")
    rnn_cell = VanillaRNNCell(input_size, hidden_size)
    x = torch.randn(batch_size, input_size)
    h = torch.zeros(batch_size, hidden_size)
    h_new = rnn_cell(x, h)
    print(f"Input shape: {x.shape}")
    print(f"Output hidden shape: {h_new.shape}")
    
    # Test LSTM
    print("\nTesting LSTM Cell...")
    lstm_cell = LSTMCell(input_size, hidden_size)
    h = torch.zeros(batch_size, hidden_size)
    c = torch.zeros(batch_size, hidden_size)
    h_new, c_new = lstm_cell(x, (h, c))
    print(f"Output hidden shape: {h_new.shape}")
    print(f"Output cell shape: {c_new.shape}")
