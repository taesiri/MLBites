"""
Softmax from Scratch - Starting Point

Implement softmax function from scratch using PyTorch.
Fill in the TODO sections to complete the implementation.
"""

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute numerically stable softmax.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax
        
    Returns:
        Softmax probabilities (same shape as input)
    """
    # TODO: Subtract max for numerical stability
    # Hint: Use torch.max(x, dim=dim, keepdim=True)
    
    # TODO: Compute exp of shifted values
    
    # TODO: Normalize by sum
    # Hint: sum along the same dimension
    
    pass


def softmax_with_temperature(
    x: torch.Tensor, 
    temperature: float = 1.0, 
    dim: int = -1
) -> torch.Tensor:
    """
    Compute softmax with temperature scaling.
    
    Higher temperature -> more uniform distribution
    Lower temperature -> more peaked distribution
    
    Args:
        x: Input tensor
        temperature: Temperature parameter (default 1.0)
        dim: Dimension along which to compute softmax
        
    Returns:
        Softmax probabilities
    """
    # TODO: Scale logits by temperature
    # Hint: Divide logits by temperature before softmax
    
    # TODO: Apply softmax
    
    pass


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log softmax in a numerically stable way.
    
    This is useful for computing cross-entropy loss.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute log softmax
        
    Returns:
        Log softmax values
    """
    # TODO: Implement log softmax
    # Hint: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test basic softmax
    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"Input logits: {logits}")
    print(f"Softmax probabilities: {probs}")
    print(f"Sum (should be 1.0): {probs.sum():.6f}")
    
    # Compare with PyTorch
    expected = torch.softmax(logits, dim=-1)
    print(f"Matches PyTorch: {torch.allclose(probs, expected)}")
    
    # Test with temperature
    print("\nTemperature scaling:")
    print(f"T=0.5 (peaked): {softmax_with_temperature(logits, 0.5)}")
    print(f"T=1.0 (normal): {softmax_with_temperature(logits, 1.0)}")
    print(f"T=2.0 (uniform): {softmax_with_temperature(logits, 2.0)}")
    
    # Test batch processing
    batch = torch.randn(4, 10)
    batch_probs = softmax(batch, dim=-1)
    print(f"\nBatch row sums: {batch_probs.sum(dim=-1)}")
