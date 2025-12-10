"""
Sparse Tensors in PyTorch - Starting Point

Learn to work with sparse tensors.
"""

import torch
import torch.nn as nn


def create_sparse_coo(indices: torch.Tensor, values: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Create a COO format sparse tensor.
    
    Args:
        indices: (2, nnz) tensor of row, col indices
        values: (nnz,) tensor of values
        size: (rows, cols) tuple
    
    Returns:
        Sparse COO tensor
    """
    # TODO: Create sparse tensor
    # torch.sparse_coo_tensor(indices, values, size)
    pass


def sparse_to_dense(sparse_tensor: torch.Tensor) -> torch.Tensor:
    """Convert sparse tensor to dense."""
    # TODO: Use .to_dense()
    pass


def dense_to_sparse(dense_tensor: torch.Tensor) -> torch.Tensor:
    """Convert dense tensor to sparse."""
    # TODO: Use .to_sparse()
    pass


def sparse_matmul(sparse_A: torch.Tensor, dense_B: torch.Tensor) -> torch.Tensor:
    """
    Sparse-dense matrix multiplication.
    
    Args:
        sparse_A: Sparse matrix (M, K)
        dense_B: Dense matrix (K, N)
    
    Returns:
        Dense result (M, N)
    """
    # TODO: Use torch.sparse.mm
    pass


class SparseLinear(nn.Module):
    """Linear layer with sparse weights."""
    
    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.9):
        super().__init__()
        # TODO: Create sparse weight matrix
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Sparse matrix multiplication
        pass


class SparseEmbedding(nn.Module):
    """
    Sparse gradient embedding for large vocabularies.
    
    Useful when vocab is large but batch only uses small subset.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        # TODO: nn.Embedding with sparse=True
        pass
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create sparse tensor
    indices = torch.tensor([[0, 1, 2], [1, 0, 2]])
    values = torch.tensor([3.0, 4.0, 5.0])
    
    sparse = create_sparse_coo(indices, values, (3, 3))
    print("Sparse tensor:")
    print(sparse)
    
    # Convert to dense
    dense = sparse_to_dense(sparse)
    print("\nDense representation:")
    print(dense)
    
    # Sparse matmul
    B = torch.randn(3, 4)
    result = sparse_matmul(sparse, B)
    print(f"\nSparse @ Dense: {result.shape}")
