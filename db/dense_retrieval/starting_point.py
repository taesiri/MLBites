"""
Dense Retrieval System - Starting Point

Build a dense retrieval system for semantic search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):
    """Simple text encoder (for demonstration)."""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        # TODO: Embedding layer
        # TODO: Encoder (LSTM, Transformer, etc.)
        # TODO: Pooling and projection
        pass
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text to dense vector.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
        Returns:
            Embeddings (batch, embed_dim)
        """
        # TODO: Embed, encode, pool
        pass


class DenseRetriever:
    """Dense retrieval system."""
    
    def __init__(self, encoder: nn.Module, embed_dim: int):
        """
        Args:
            encoder: Text encoder model
            embed_dim: Embedding dimension
        """
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.index = None  # Document embeddings
        self.documents = []
    
    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        # TODO: Tokenize texts
        # TODO: Encode with model
        # TODO: Normalize embeddings
        pass
    
    def build_index(self, documents: list[str]):
        """Build index from documents."""
        # TODO: Store documents
        # TODO: Encode all documents
        # TODO: Store embeddings as index
        pass
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Search for similar documents.
        
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        # TODO: Encode query
        # TODO: Compute similarity with all documents
        # TODO: Return top-k results
        pass


class DenseRetrieverFAISS:
    """Dense retriever with FAISS for fast search."""
    
    def __init__(self, encoder: nn.Module, embed_dim: int):
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.index = None
        self.documents = []
    
    def build_index(self, documents: list[str]):
        """Build FAISS index."""
        # TODO: Install faiss-cpu or faiss-gpu
        # import faiss
        # self.index = faiss.IndexFlatIP(self.embed_dim)
        # self.index.add(embeddings.numpy())
        pass
    
    def search(self, query: str, top_k: int = 10):
        """Search using FAISS."""
        # TODO: Encode query
        # TODO: Search with FAISS
        # distances, indices = self.index.search(query_emb, top_k)
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create simple encoder
    encoder = SimpleEncoder(vocab_size=1000, embed_dim=256)
    retriever = DenseRetriever(encoder, embed_dim=256)
    
    # Demo documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing deals with text data",
        "Computer vision focuses on image understanding",
        "Reinforcement learning learns from rewards",
    ]
    
    # Build index
    retriever.build_index(documents)
    
    # Search
    query = "neural networks and AI"
    results = retriever.search(query, top_k=3)
    
    print(f"Query: {query}")
    print("\nTop results:")
    for doc_id, score in results:
        print(f"  {score:.3f}: {documents[doc_id]}")
