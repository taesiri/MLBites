"""
Adam Optimizer from Scratch - Starting Point

Implement the Adam optimizer without using built-in optimizers.
Fill in the TODO sections to complete the implementation.
"""

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
        
        # TODO: Initialize state for each parameter
        # Each parameter needs: m (first moment), v (second moment), t (step count)
        self.state = {}
        
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
            
            # TODO: Get or initialize state for this parameter
            
            # TODO: Apply weight decay if specified (L2 regularization)
            # For regular Adam: add weight_decay * param to gradient
            # For AdamW: apply directly to parameter
            
            # TODO: Update first moment (m)
            # m = β1 * m + (1 - β1) * grad
            
            # TODO: Update second moment (v)
            # v = β2 * v + (1 - β2) * grad²
            
            # TODO: Compute bias-corrected moments
            # m_hat = m / (1 - β1^t)
            # v_hat = v / (1 - β2^t)
            
            # TODO: Update parameters
            # param = param - lr * m_hat / (sqrt(v_hat) + eps)
            
            pass


class AdamW(Adam):
    """AdamW optimizer with decoupled weight decay."""
    
    def step(self):
        """Perform optimization step with decoupled weight decay."""
        for param in self.params:
            if param.grad is None:
                continue
            
            # TODO: Apply decoupled weight decay first
            # param = param - lr * weight_decay * param
            
            # TODO: Then apply Adam update (without weight decay in gradient)
            
            pass


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
