"""
Deep Neural Network (MLP) - Solution

Complete implementation of a configurable deep neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.5,
        activation: str = 'relu',
        use_batch_norm: bool = False
    ):
        """
        Create a deep neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output features/classes
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh', 'leaky_relu')
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Choose activation function
        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }
        act_fn = activations.get(activation, nn.ReLU)
        
        # Build network layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Add final output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Apply weight initialization
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ResidualMLP(nn.Module):
    """MLP with residual connections for deeper networks."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def _make_block(self, dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        
        return self.output_proj(x)


def train_mlp(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs: int = 10,
    lr: float = 0.001,
    device: str = 'cpu'
):
    """Train the MLP model."""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")


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
    
    # Test Residual MLP
    print("\n" + "="*50)
    print("Residual MLP:")
    res_model = ResidualMLP(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        num_layers=4
    )
    print(res_model)
    
    output = res_model(x)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in res_model.parameters()):,}")
