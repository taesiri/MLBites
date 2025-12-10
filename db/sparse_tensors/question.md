# Sparse Tensors in PyTorch

## Problem Statement

Learn to work with **Sparse Tensors** in PyTorch for memory-efficient operations on data with many zeros (graphs, NLP embeddings, recommender systems).

Your task is to:

1. Create sparse tensors in COO and CSR formats
2. Convert between sparse and dense representations
3. Perform sparse matrix operations
4. Build sparse embeddings for large vocabularies

## Sparse Formats

| Format | Full Name | Best For |
|--------|-----------|----------|
| COO | Coordinate | Construction, conversion |
| CSR | Compressed Sparse Row | Row slicing, SpMM |
| CSC | Compressed Sparse Column | Column slicing |

## Function Signature

```python
def create_sparse_coo(indices: torch.Tensor, values: torch.Tensor, size: tuple) -> torch.Tensor:
    """Create COO sparse tensor."""
    pass

def sparse_matmul(sparse_A: torch.Tensor, dense_B: torch.Tensor) -> torch.Tensor:
    """Sparse-dense matrix multiplication."""
    pass

class SparseEmbedding(nn.Module):
    """Memory-efficient embedding for large vocabularies."""
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        pass
```

## Example

```python
# Create sparse tensor from coordinates
indices = torch.tensor([[0, 1, 2], [1, 0, 2]])  # (row, col) pairs
values = torch.tensor([3.0, 4.0, 5.0])
size = (3, 3)

sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
dense_tensor = sparse_tensor.to_dense()

# Sparse matrix multiplication
result = torch.sparse.mm(sparse_tensor, dense_matrix)
```

## Use Cases

- **Graph Neural Networks**: Adjacency matrices are sparse
- **NLP**: Bag-of-words, TF-IDF vectors
- **Recommender Systems**: User-item interaction matrices
- **Scientific Computing**: Finite element matrices

## Hints

- Use `coalesce()` to combine duplicate indices
- `to_sparse()` converts dense to sparse
- Sparse gradients: set `sparse=True` in embedding
- `torch.sparse.mm` for sparse × dense multiplication
