"""
Adam Optimizer from Scratch - Solution

Complete implementation of Adam and AdamW optimizers.
"""

import math
import torch
import torch.nn as nn


class Adam:
    """Adam optimizer implemented from scratch."""
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (β1, β2)
            eps: Small constant for numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state for each parameter
        self.state = {}
        for param in self.params:
            self.state[param] = {
                'm': torch.zeros_like(param.data),  # First moment
                'v': torch.zeros_like(param.data),  # Second moment
                't': 0  # Step count
            }
    
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
            state = self.state[param]
            
            # Increment step count
            state['t'] += 1
            t = state['t']
            
            # Apply L2 weight decay (add to gradient)
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Get moments
            m, v = state['m'], state['v']
            
            # Update first moment (m)
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # Update second moment (v)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # Compute bias-corrected moments
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Update parameters
            param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


class AdamW:
    """AdamW optimizer with decoupled weight decay."""
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        """Initialize AdamW optimizer."""
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.state = {}
        for param in self.params:
            self.state[param] = {
                'm': torch.zeros_like(param.data),
                'v': torch.zeros_like(param.data),
                't': 0
            }
    
    def zero_grad(self):
        """Zero out all parameter gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """Perform optimization step with decoupled weight decay."""
        for param in self.params:
            if param.grad is None:
                continue
            
            # Apply decoupled weight decay FIRST (key difference from Adam)
            if self.weight_decay != 0:
                param.data.mul_(1 - self.lr * self.weight_decay)
            
            grad = param.grad.data
            state = self.state[param]
            
            state['t'] += 1
            t = state['t']
            
            m, v = state['m'], state['v']
            
            # Update moments (without weight decay in gradient)
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Update parameters
            param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create simple model and data
    model = nn.Linear(10, 1)
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Test Adam
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training with custom Adam...")
    for epoch in range(100):
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Compare with PyTorch's Adam
    print("\nComparing with PyTorch Adam...")
    model2 = nn.Linear(10, 1)
    torch_optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
    
    for epoch in range(100):
        output = model2(x)
        loss = criterion(output, y)
        
        torch_optimizer.zero_grad()
        loss.backward()
        torch_optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
