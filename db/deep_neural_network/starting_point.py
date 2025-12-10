"""
Deep Neural Network (MLP) - Starting Point

Implement a deep neural network with configurable architecture.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.5,
        activation: str = 'relu'
    ):
        """
        Create a deep neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output features/classes
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        
        # TODO: Choose activation function based on string
        # if activation == 'relu': act_fn = nn.ReLU()
        
        # TODO: Build network layers dynamically
        # Iterate through hidden_dims and create:
        # - Linear layer
        # - Activation
        # - Dropout
        
        # TODO: Add final output layer (no activation)
        
        # TODO: Apply weight initialization
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # TODO: Pass through all layers
        pass
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        # TODO: Initialize each linear layer properly
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)
        pass


def train_mlp(
    model: nn.Module,
    train_loader,
    epochs: int = 10,
    lr: float = 0.001
):
    """Train the MLP model."""
    # TODO: Define optimizer and loss
    
    # TODO: Training loop
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create MLP
    model = MLP(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        dropout=0.5
    )
    
    print("Model architecture:")
    print(model)
    
    # Test forward pass
    x = torch.randn(32, 784)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
