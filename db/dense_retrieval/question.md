# Dense Retrieval System

## Problem Statement

Build a **Dense Retrieval System** using PyTorch for semantic search. Dense retrieval uses neural network embeddings to find similar documents/passages, unlike sparse methods (BM25) that use term matching.

Your task is to:

1. Create an encoder model for text/documents
2. Build an index of document embeddings
3. Query the index with semantic similarity
4. Implement efficient approximate nearest neighbor search

## Requirements

- Encode documents into dense vectors
- Use cosine similarity for retrieval
- Support batch encoding for efficiency
- Optionally integrate with FAISS for fast search

## Function Signature

```python
class DenseRetriever:
    def __init__(self, encoder: nn.Module, embed_dim: int):
        pass
    
    def index(self, documents: list[str]):
        """Build index from documents."""
        pass
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return top-k (doc_id, score) pairs."""
        pass

class BiEncoder(nn.Module):
    """Dual encoder for query and document."""
    def __init__(self, hidden_size: int = 768):
        pass
    
    def encode_query(self, query: str) -> torch.Tensor:
        pass
    
    def encode_document(self, doc: str) -> torch.Tensor:
        pass
```

## Example

```python
# Create retriever
encoder = BiEncoder(hidden_size=768)
retriever = DenseRetriever(encoder, embed_dim=768)

# Index documents
documents = ["The cat sat on the mat", "Dogs are loyal pets", ...]
retriever.index(documents)

# Search
results = retriever.search("feline animals", top_k=5)
for doc_id, score in results:
    print(f"{score:.3f}: {documents[doc_id]}")
```

## Retrieval Methods

| Method | Description |
|--------|-------------|
| Exact | Brute-force cosine similarity (slow, accurate) |
| FAISS IVF | Inverted file index (faster, approximate) |
| HNSW | Hierarchical navigable small world graphs |
| Product Quantization | Compressed vectors for memory efficiency |

## Hints

- Normalize embeddings for cosine similarity
- Use mean pooling over token embeddings
- FAISS: `faiss.IndexFlatIP` for inner product (cosine on normalized)
- Batch queries for efficiency
