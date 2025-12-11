"""
Bag-of-Words and TF-IDF - Starting Point

Implement BoW and TF-IDF from scratch.
"""

import math
import torch
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    # TODO: Split text into tokens
    pass


class BagOfWords:
    """Bag-of-Words text representation."""
    
    def __init__(self):
        self.vocab = {}  # word -> index
        self.vocab_size = 0
    
    def fit(self, documents: list[str]):
        """
        Build vocabulary from documents.
        
        Args:
            documents: List of text documents
        """
        # TODO: Tokenize all documents
        # TODO: Build vocabulary (unique words)
        # TODO: Create word -> index mapping
        pass
    
    def transform(self, documents: list[str]) -> torch.Tensor:
        """
        Convert documents to BoW vectors.
        
        Args:
            documents: List of text documents
            
        Returns:
            Tensor of shape (n_docs, vocab_size)
        """
        # TODO: For each document:
        #   - Tokenize
        #   - Count word occurrences
        #   - Create vector
        pass
    
    def fit_transform(self, documents: list[str]) -> torch.Tensor:
        self.fit(documents)
        return self.transform(documents)


class TFIDF:
    """TF-IDF text representation."""
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.idf = {}  # word -> IDF value
    
    def fit(self, documents: list[str]):
        """
        Compute IDF values from corpus.
        
        IDF(t) = log(N / df(t))
        where df(t) = number of documents containing term t
        """
        # TODO: Build vocabulary
        # TODO: Compute document frequency for each word
        # TODO: Compute IDF values
        pass
    
    def transform(self, documents: list[str]) -> torch.Tensor:
        """
        Convert documents to TF-IDF vectors.
        
        TF(t, d) = count(t in d) / total_words(d)
        TF-IDF = TF × IDF
        """
        # TODO: For each document:
        #   - Compute term frequencies
        #   - Multiply by IDF
        pass
    
    def fit_transform(self, documents: list[str]) -> torch.Tensor:
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between vectors."""
    # TODO: Implement cosine similarity
    pass


if __name__ == "__main__":
    # Sample documents
    documents = [
        "the cat sat on the mat",
        "the dog ate my homework",
        "cats and dogs are pets",
        "I love my cat",
    ]
    
    # Bag of Words
    print("=== Bag of Words ===")
    bow = BagOfWords()
    bow_vectors = bow.fit_transform(documents)
    print(f"Vocabulary size: {bow.vocab_size}")
    print(f"Vector shape: {bow_vectors.shape}")
    print(f"First document vector: {bow_vectors[0]}")
    
    # TF-IDF
    print("\n=== TF-IDF ===")
    tfidf = TFIDF()
    tfidf_vectors = tfidf.fit_transform(documents)
    print(f"Vector shape: {tfidf_vectors.shape}")
    print(f"First document vector: {tfidf_vectors[0]}")
    
    # Similarity
    print("\n=== Document Similarity ===")
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
            print(f"Doc {i} vs Doc {j}: {sim:.3f}")
