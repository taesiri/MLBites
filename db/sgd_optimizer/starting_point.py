"""
SGD Optimizer from Scratch - Starting Point

Implement SGD with momentum from scratch.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn


class SGD:
    """SGD optimizer with momentum, implemented from scratch."""
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        """
        Initialize SGD optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            momentum: Momentum factor (0 = no momentum)
            weight_decay: Weight decay (L2 penalty)
            nesterov: Whether to use Nesterov momentum
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        
        # TODO: Initialize velocity for each parameter (for momentum)
        self.velocity = {}
        
        pass
    
    def zero_grad(self):
        """Zero out all parameter gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """Perform a single optimization step."""
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # TODO: Apply weight decay
            # grad = grad + weight_decay * param
            
            # TODO: Apply momentum if enabled
            # v = momentum * v + grad
            
            # TODO: Apply Nesterov momentum if enabled
            # update = momentum * v + grad
            
            # TODO: Update parameter
            # param = param - lr * update
            
            pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create simple model and data
    model = nn.Linear(10, 1)
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Test SGD with momentum
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    
    print("Training with custom SGD (momentum=0.9)...")
    for epoch in range(100):
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
