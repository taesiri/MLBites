"""
Sparse Tensors in PyTorch - Solution

Complete sparse tensor operations.
"""

import torch
import torch.nn as nn


def create_sparse_coo(indices: torch.Tensor, values: torch.Tensor, size: tuple) -> torch.Tensor:
    """Create a COO format sparse tensor."""
    return torch.sparse_coo_tensor(indices, values, size)


def create_sparse_csr(crow_indices: torch.Tensor, col_indices: torch.Tensor, 
                      values: torch.Tensor, size: tuple) -> torch.Tensor:
    """Create a CSR format sparse tensor."""
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size)


def sparse_to_dense(sparse_tensor: torch.Tensor) -> torch.Tensor:
    """Convert sparse tensor to dense."""
    return sparse_tensor.to_dense()


def dense_to_sparse(dense_tensor: torch.Tensor, layout: str = 'coo') -> torch.Tensor:
    """Convert dense tensor to sparse."""
    if layout == 'coo':
        return dense_tensor.to_sparse()
    elif layout == 'csr':
        return dense_tensor.to_sparse_csr()
    else:
        raise ValueError(f"Unknown layout: {layout}")


def sparse_matmul(sparse_A: torch.Tensor, dense_B: torch.Tensor) -> torch.Tensor:
    """Sparse-dense matrix multiplication."""
    return torch.sparse.mm(sparse_A, dense_B)


def create_random_sparse(size: tuple, sparsity: float = 0.9) -> torch.Tensor:
    """Create random sparse tensor with given sparsity."""
    dense = torch.randn(size)
    mask = torch.rand(size) > sparsity
    dense = dense * mask.float()
    return dense.to_sparse()


class SparseLinear(nn.Module):
    """Linear layer with sparse weights."""
    
    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.9, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create sparse weight
        weight = torch.randn(out_features, in_features)
        mask = torch.rand_like(weight) > sparsity
        weight = weight * mask.float()
        self.weight = nn.Parameter(weight.to_sparse())
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # weight: sparse (out_features, in_features)
        out = torch.sparse.mm(self.weight, x.T).T
        if self.bias is not None:
            out = out + self.bias
        return out


class SparseEmbedding(nn.Module):
    """Sparse gradient embedding for large vocabularies."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, 
            padding_idx=padding_idx, 
            sparse=True  # Sparse gradients
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)


def sparse_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                     mask: torch.Tensor = None, top_k: int = None) -> torch.Tensor:
    """
    Sparse attention that only attends to top-k or masked positions.
    
    Args:
        query, key, value: (batch, seq, dim)
        mask: Sparse mask indicating which positions to attend
        top_k: Only attend to top-k positions per query
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
    
    if top_k is not None:
        # Keep only top-k scores per query
        topk_values, topk_indices = scores.topk(top_k, dim=-1)
        sparse_scores = torch.zeros_like(scores).fill_(float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_values)
        scores = sparse_scores
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create sparse tensor
    indices = torch.tensor([[0, 1, 2], [1, 0, 2]])
    values = torch.tensor([3.0, 4.0, 5.0])
    
    sparse = create_sparse_coo(indices, values, (3, 3))
    print("Sparse tensor:")
    print(sparse)
    
    dense = sparse_to_dense(sparse)
    print("\nDense:")
    print(dense)
    
    # Matmul
    B = torch.randn(3, 4)
    result = sparse_matmul(sparse.coalesce(), B)
    print(f"\nSparse @ Dense: {result.shape}")
    
    # Sparse embedding
    print("\n--- Sparse Embedding ---")
    emb = SparseEmbedding(10000, 256)
    ids = torch.randint(0, 10000, (32, 10))
    out = emb(ids)
    print(f"Embedding output: {out.shape}")
    
    # Check gradient sparsity
    loss = out.sum()
    loss.backward()
    print(f"Gradient is sparse: {emb.embedding.weight.grad.is_sparse}")
