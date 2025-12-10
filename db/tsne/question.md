# t-SNE from Scratch

## Problem Statement

Implement **t-Distributed Stochastic Neighbor Embedding (t-SNE)** from scratch. t-SNE is a dimensionality reduction technique particularly suited for visualizing high-dimensional data.

Your task is to:

1. Compute pairwise affinities in high-dimensional space (Gaussian)
2. Compute pairwise affinities in low-dimensional space (Student's t-distribution)
3. Minimize KL divergence between distributions using gradient descent

## Requirements

- Do **NOT** use sklearn's TSNE
- Implement the core t-SNE algorithm
- Use gradient descent for optimization
- Handle perplexity parameter

## Function Signature

```python
class TSNE:
    def __init__(self, n_components: int = 2, perplexity: float = 30.0, 
                 learning_rate: float = 200.0, n_iter: int = 1000):
        pass
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Reduce dimensionality of X."""
        pass
```

## t-SNE Algorithm

```
1. Compute pairwise affinities p_ij in high-D (Gaussian, based on perplexity)
2. Initialize Y randomly in low-D
3. Repeat for n_iter:
   a. Compute pairwise affinities q_ij in low-D (Student's t)
   b. Compute gradient of KL divergence
   c. Update Y using gradient descent with momentum
```

## Key Formulas

**High-dimensional affinity:**
```
p_j|i = exp(-||x_i - x_j||² / 2σ_i²) / Σ_k exp(-||x_i - x_k||² / 2σ_i²)
p_ij = (p_j|i + p_i|j) / 2n
```

**Low-dimensional affinity (Student's t with 1 DOF):**
```
q_ij = (1 + ||y_i - y_j||²)^(-1) / Σ_k,l (1 + ||y_k - y_l||²)^(-1)
```

## Hints

- σ_i is set based on perplexity using binary search
- Perplexity ≈ 2^H(P_i) where H is entropy
- Use momentum and learning rate annealing for better results
- Early exaggeration helps initial cluster formation
