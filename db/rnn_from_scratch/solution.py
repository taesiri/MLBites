"""
RNN Cell from Scratch - Solution

Complete implementation of RNN and LSTM cells from scratch.
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
        
        # Weight matrices
        self.w_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        
        # Bias terms
        self.b_ih = nn.Parameter(torch.empty(hidden_size))
        self.b_hh = nn.Parameter(torch.empty(hidden_size))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.w_ih, self.w_hh]:
            nn.init.uniform_(weight, -stdv, stdv)
        for bias in [self.b_ih, self.b_hh]:
            nn.init.zeros_(bias)
    
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
        # Compute new hidden state
        # h_new = tanh(x @ W_ih.T + h @ W_hh.T + b_ih + b_hh)
        h_new = torch.tanh(
            x @ self.w_ih.T + h @ self.w_hh.T + self.b_ih + self.b_hh
        )
        return h_new


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
        
        # Combined weight matrices for all gates (i, f, g, o)
        # More efficient than separate matrices
        self.w_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        
        # Bias terms
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.w_ih, self.w_hh]:
            nn.init.uniform_(weight, -stdv, stdv)
        for bias in [self.b_ih, self.b_hh]:
            nn.init.zeros_(bias)
    
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
        
        # Compute all gates at once
        gates = x @ self.w_ih.T + h @ self.w_hh.T + self.b_ih + self.b_hh
        
        # Split into individual gates
        # Order: input, forget, cell (g), output
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate (candidate)
        o = torch.sigmoid(o)  # Output gate
        
        # Compute new cell state
        c_new = f * c + i * g
        
        # Compute new hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


def process_sequence(cell, x_seq, h_init, c_init=None):
    """
    Process a full sequence through an RNN/LSTM cell.
    
    Args:
        cell: RNN or LSTM cell
        x_seq: Input sequence (batch, seq_len, input_size)
        h_init: Initial hidden state
        c_init: Initial cell state (for LSTM only)
        
    Returns:
        Tuple of (hidden_states, final_state)
        - hidden_states: (seq_len, batch, hidden_size)
        - final_state: final hidden (and cell for LSTM) state
    """
    batch_size, seq_len, input_size = x_seq.shape
    hidden_states = []
    
    h = h_init
    if c_init is not None:
        c = c_init
        state = (h, c)
    
    for t in range(seq_len):
        x_t = x_seq[:, t, :]  # (batch, input_size)
        
        if c_init is not None:  # LSTM
            h, c = cell(x_t, (h, c))
            state = (h, c)
        else:  # Vanilla RNN
            h = cell(x_t, h)
            state = h
        
        hidden_states.append(h)
    
    # Stack hidden states: (seq_len, batch, hidden_size)
    hidden_states = torch.stack(hidden_states, dim=0)
    
    return hidden_states, state


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
    
    # Compare with PyTorch's built-in
    rnn_builtin = nn.RNNCell(input_size, hidden_size)
    
    # Test LSTM
    print("\nTesting LSTM Cell...")
    lstm_cell = LSTMCell(input_size, hidden_size)
    h = torch.zeros(batch_size, hidden_size)
    c = torch.zeros(batch_size, hidden_size)
    h_new, c_new = lstm_cell(x, (h, c))
    print(f"Output hidden shape: {h_new.shape}")
    print(f"Output cell shape: {c_new.shape}")
    
    # Test sequence processing
    print("\nTesting Sequence Processing...")
    x_seq = torch.randn(batch_size, seq_len, input_size)
    h_init = torch.zeros(batch_size, hidden_size)
    c_init = torch.zeros(batch_size, hidden_size)
    
    hidden_states, final_state = process_sequence(lstm_cell, x_seq, h_init, c_init)
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Final hidden shape: {final_state[0].shape}")
