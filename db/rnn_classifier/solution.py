"""
RNN Text Classifier - Solution

Complete implementation of an RNN for sequence classification using PyTorch.
"""

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """RNN-based classifier for sequence data."""
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5
    ):
        """
        Initialize RNN classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden state
            num_classes: Number of output classes
            num_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        # If bidirectional, input size is hidden_dim * 2
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN.
        
        Args:
            x: Token indices of shape (batch, seq_len)
            
        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Embed the input tokens
        # embedded: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # Pass through LSTM
        # output: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        output, (h_n, c_n) = self.lstm(embedded)
        
        # Get the final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n[-2] is forward final, h_n[-1] is backward final
            hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            # Just use the last hidden state
            hidden = h_n[-1, :, :]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Pass through fully connected layer
        logits = self.fc(hidden)
        
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create RNN classifier
    model = RNNClassifier(
        vocab_size=10000,
        embed_dim=128,
        hidden_dim=256,
        num_classes=2,
        bidirectional=True
    )
    
    # Generate random input (batch of 4, sequence length 50)
    x = torch.randint(0, 10000, (4, 50))
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
