"""
Word2Vec from Scratch - Starting Point
"""

import torch
import torch.nn as nn
from collections import Counter


class SkipGram(nn.Module):
    """Skip-gram Word2Vec model."""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # TODO: Center word embeddings
        # TODO: Context word embeddings
        pass
    
    def forward(self, center: torch.Tensor, context: torch.Tensor, negatives: torch.Tensor = None):
        """
        Compute negative sampling loss.
        
        Args:
            center: Center word indices (batch,)
            context: Positive context indices (batch,)
            negatives: Negative sample indices (batch, num_neg)
        """
        # TODO: Get embeddings
        # TODO: Compute positive score
        # TODO: Compute negative scores
        # TODO: Return loss
        pass


class CBOW(nn.Module):
    """Continuous Bag of Words model."""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # TODO: Embeddings
        pass
    
    def forward(self, context: torch.Tensor, center: torch.Tensor, negatives: torch.Tensor = None):
        """
        Args:
            context: Context word indices (batch, context_size)
            center: Center word index (batch,)
        """
        # TODO: Average context embeddings
        # TODO: Compute loss with negative sampling
        pass


class Word2Vec:
    """Word2Vec trainer."""
    
    def __init__(self, sentences: list, embed_dim: int = 100, window: int = 5, min_count: int = 5):
        self.embed_dim = embed_dim
        self.window = window
        
        # TODO: Build vocabulary
        # TODO: Create word -> index mapping
        # TODO: Initialize model
        pass
    
    def _build_vocab(self, sentences: list, min_count: int):
        # TODO: Count words, filter by min_count
        pass
    
    def _generate_training_data(self, sentences: list):
        # TODO: Generate (center, context) pairs
        pass
    
    def _sample_negatives(self, batch_size: int, num_neg: int) -> torch.Tensor:
        # TODO: Sample negatives based on word frequency
        pass
    
    def train(self, epochs: int = 5, batch_size: int = 256, neg_samples: int = 5):
        # TODO: Training loop
        pass
    
    def get_embedding(self, word: str) -> torch.Tensor:
        pass
    
    def most_similar(self, word: str, k: int = 10) -> list:
        # TODO: Find most similar words by cosine similarity
        pass


if __name__ == "__main__":
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "in", "the", "park"],
        ["cats", "and", "dogs", "are", "pets"],
    ]
    
    w2v = Word2Vec(sentences, embed_dim=50, window=2)
    w2v.train(epochs=10)
    
    print("Similar to 'cat':", w2v.most_similar("cat", k=3))
