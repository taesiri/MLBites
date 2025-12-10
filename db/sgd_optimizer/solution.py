"""
SGD Optimizer from Scratch - Solution

Complete implementation of SGD with momentum.
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
        
        # Initialize velocity for each parameter
        self.velocity = {param: torch.zeros_like(param.data) for param in self.params}
    
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
            
            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Apply momentum if enabled
            if self.momentum != 0:
                v = self.velocity[param]
                v.mul_(self.momentum).add_(grad)
                
                if self.nesterov:
                    # Nesterov: use look-ahead gradient
                    update = grad.add(v, alpha=self.momentum)
                else:
                    update = v
            else:
                update = grad
            
            # Update parameter
            param.data.add_(update, alpha=-self.lr)


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
    
    # Test Nesterov momentum
    print("\nTraining with Nesterov momentum...")
    model2 = nn.Linear(10, 1)
    optimizer2 = SGD(model2.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    
    for epoch in range(100):
        output = model2(x)
        loss = criterion(output, y)
        
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
