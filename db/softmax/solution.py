"""
Softmax from Scratch - Solution

Complete implementation of softmax function from scratch.
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
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    
    # Compute exp of shifted values
    exp_x = torch.exp(x_shifted)
    
    # Normalize by sum
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    probs = exp_x / sum_exp
    
    return probs


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
    # Scale logits by temperature
    x_scaled = x / temperature
    
    # Apply softmax
    return softmax(x_scaled, dim=dim)


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
    # log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    log_sum_exp = torch.log(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True))
    
    return x_shifted - log_sum_exp


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
    
    # Test log softmax
    log_probs = log_softmax(logits)
    expected_log = torch.log_softmax(logits, dim=-1)
    print(f"\nLog softmax matches PyTorch: {torch.allclose(log_probs, expected_log)}")
    
    # Test numerical stability with large values
    large_logits = torch.tensor([1000.0, 1001.0, 1002.0])
    large_probs = softmax(large_logits)
    print(f"\nLarge logits: {large_logits}")
    print(f"Stable softmax: {large_probs}")
    print(f"No NaN or Inf: {not (torch.isnan(large_probs).any() or torch.isinf(large_probs).any())}")
