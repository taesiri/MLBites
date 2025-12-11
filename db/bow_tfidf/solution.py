"""
Bag-of-Words and TF-IDF - Solution

Complete implementation of BoW and TF-IDF.
"""

import math
import torch
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase and split on non-alphanumeric."""
    import re
    return re.findall(r'\b\w+\b', text.lower())


class BagOfWords:
    """Bag-of-Words text representation."""
    
    def __init__(self, max_features: int = None):
        self.vocab = {}
        self.vocab_size = 0
        self.max_features = max_features
    
    def fit(self, documents: list[str]):
        """Build vocabulary from documents."""
        word_counts = Counter()
        
        for doc in documents:
            tokens = tokenize(doc)
            word_counts.update(tokens)
        
        # Get most common words if max_features is set
        if self.max_features:
            words = [w for w, _ in word_counts.most_common(self.max_features)]
        else:
            words = list(word_counts.keys())
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(words))}
        self.vocab_size = len(self.vocab)
        
        return self
    
    def transform(self, documents: list[str]) -> torch.Tensor:
        """Convert documents to BoW vectors."""
        vectors = torch.zeros(len(documents), self.vocab_size)
        
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            for token in tokens:
                if token in self.vocab:
                    vectors[i, self.vocab[token]] += 1
        
        return vectors
    
    def fit_transform(self, documents: list[str]) -> torch.Tensor:
        self.fit(documents)
        return self.transform(documents)
    
    def get_feature_names(self) -> list[str]:
        """Return vocabulary words in order."""
        return [w for w, _ in sorted(self.vocab.items(), key=lambda x: x[1])]


class TFIDF:
    """TF-IDF text representation."""
    
    def __init__(self, max_features: int = None, smooth_idf: bool = True):
        self.vocab = {}
        self.vocab_size = 0
        self.idf = None
        self.max_features = max_features
        self.smooth_idf = smooth_idf
    
    def fit(self, documents: list[str]):
        """Compute IDF values from corpus."""
        n_docs = len(documents)
        word_doc_count = Counter()
        word_counts = Counter()
        
        for doc in documents:
            tokens = tokenize(doc)
            word_counts.update(tokens)
            unique_tokens = set(tokens)
            word_doc_count.update(unique_tokens)
        
        # Build vocabulary
        if self.max_features:
            words = [w for w, _ in word_counts.most_common(self.max_features)]
        else:
            words = list(word_counts.keys())
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(words))}
        self.vocab_size = len(self.vocab)
        
        # Compute IDF
        self.idf = torch.zeros(self.vocab_size)
        for word, idx in self.vocab.items():
            df = word_doc_count.get(word, 0)
            if self.smooth_idf:
                # Smooth IDF: log((N + 1) / (df + 1)) + 1
                self.idf[idx] = math.log((n_docs + 1) / (df + 1)) + 1
            else:
                self.idf[idx] = math.log(n_docs / (df + 1e-10))
        
        return self
    
    def transform(self, documents: list[str]) -> torch.Tensor:
        """Convert documents to TF-IDF vectors."""
        vectors = torch.zeros(len(documents), self.vocab_size)
        
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            n_tokens = len(tokens)
            
            if n_tokens == 0:
                continue
            
            # Compute TF
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token in self.vocab:
                    tf = count / n_tokens
                    idx = self.vocab[token]
                    vectors[i, idx] = tf * self.idf[idx]
        
        # L2 normalize (optional but common)
        norms = vectors.norm(dim=1, keepdim=True)
        vectors = vectors / (norms + 1e-10)
        
        return vectors
    
    def fit_transform(self, documents: list[str]) -> torch.Tensor:
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between vectors."""
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-10)


def pairwise_cosine_similarity(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix."""
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-10)
    return X_norm @ X_norm.T


if __name__ == "__main__":
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
    print(f"Vocabulary: {bow.get_feature_names()}")
    print(f"Vector shape: {bow_vectors.shape}")
    print(f"Doc 0: {bow_vectors[0].tolist()}")
    
    # TF-IDF
    print("\n=== TF-IDF ===")
    tfidf = TFIDF()
    tfidf_vectors = tfidf.fit_transform(documents)
    print(f"Vector shape: {tfidf_vectors.shape}")
    
    # Similarity matrix
    print("\n=== Similarity Matrix ===")
    sim_matrix = pairwise_cosine_similarity(tfidf_vectors)
    print(sim_matrix)
    
    # Most similar documents
    print("\n=== Most Similar Pairs ===")
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = sim_matrix[i, j].item()
            print(f"'{documents[i][:30]}...' vs '{documents[j][:30]}...': {sim:.3f}")
