"""
Dense Retrieval System - Solution

Complete dense retrieval implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):
    """Simple text encoder."""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        output, (h, _) = self.lstm(x)
        # Mean pooling
        pooled = output.mean(dim=1)
        return self.proj(pooled)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder."""
    
    def __init__(self, vocab_size: int = 30000, embed_dim: int = 768, num_heads: int = 12, num_layers: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, 4*embed_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(input_ids) + self.pos_embedding[:, :input_ids.size(1)]
        
        if attention_mask is not None:
            # Convert to transformer mask format
            mask = attention_mask == 0
        else:
            mask = None
        
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        # Mean pooling (ignoring padding)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        return x


class DenseRetriever:
    """Dense retrieval with exact search."""
    
    def __init__(self, encoder: nn.Module, embed_dim: int, device: str = 'cpu'):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.embed_dim = embed_dim
        self.device = device
        self.index = None
        self.documents = []
    
    def _simple_tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        """Simple character-level tokenization for demo."""
        tokens = [ord(c) % 1000 for c in text.lower()[:max_len]]
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens).unsqueeze(0)
    
    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to normalized embeddings."""
        embeddings = []
        for text in texts:
            input_ids = self._simple_tokenize(text).to(self.device)
            emb = self.encoder(input_ids)
            embeddings.append(emb)
        
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    def build_index(self, documents: list[str]):
        """Build index from documents."""
        self.documents = documents
        self.index = self.encode(documents)
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search for similar documents."""
        query_emb = self.encode([query])
        
        # Cosine similarity (embeddings are normalized)
        similarities = (query_emb @ self.index.T).squeeze(0)
        
        # Get top-k
        scores, indices = similarities.topk(min(top_k, len(self.documents)))
        
        return [(idx.item(), score.item()) for idx, score in zip(indices, scores)]


class DenseRetrieverFAISS:
    """Dense retriever with FAISS for fast approximate search."""
    
    def __init__(self, encoder: nn.Module, embed_dim: int, device: str = 'cpu'):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.embed_dim = embed_dim
        self.device = device
        self.faiss_index = None
        self.documents = []
    
    def _simple_tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        tokens = [ord(c) % 1000 for c in text.lower()[:max_len]]
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens).unsqueeze(0)
    
    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        embeddings = []
        for text in texts:
            input_ids = self._simple_tokenize(text).to(self.device)
            emb = self.encoder(input_ids)
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0)
        return F.normalize(embeddings, dim=-1)
    
    def build_index(self, documents: list[str], use_ivf: bool = False, nlist: int = 100):
        """Build FAISS index."""
        try:
            import faiss
        except ImportError:
            print("FAISS not installed. Using exact search fallback.")
            self.documents = documents
            self.index_embeddings = self.encode(documents)
            return
        
        self.documents = documents
        embeddings = self.encode(documents).cpu().numpy()
        
        if use_ivf and len(documents) > nlist:
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embed_dim, nlist)
            self.faiss_index.train(embeddings)
        else:
            # Flat index for exact search
            self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
        
        self.faiss_index.add(embeddings)
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        query_emb = self.encode([query]).cpu().numpy()
        
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_emb, top_k)
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        else:
            # Fallback without FAISS
            similarities = (torch.from_numpy(query_emb) @ self.index_embeddings.T).squeeze(0)
            scores, indices = similarities.topk(min(top_k, len(self.documents)))
            return [(idx.item(), score.item()) for idx, score in zip(indices, scores)]


if __name__ == "__main__":
    torch.manual_seed(42)
    
    encoder = SimpleEncoder(vocab_size=1000, embed_dim=256)
    retriever = DenseRetriever(encoder, embed_dim=256)
    
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing deals with text data",
        "Computer vision focuses on image understanding",
        "Reinforcement learning learns from rewards",
        "Transformers revolutionized NLP with attention mechanisms",
        "CNNs are great for image classification tasks",
        "RNNs and LSTMs handle sequential data",
    ]
    
    retriever.build_index(documents)
    
    query = "neural networks and AI"
    results = retriever.search(query, top_k=3)
    
    print(f"Query: {query}")
    print("\nTop results:")
    for doc_id, score in results:
        print(f"  {score:.3f}: {documents[doc_id]}")
