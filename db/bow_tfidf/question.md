# Bag-of-Words and TF-IDF

## Problem Statement

Implement **Bag-of-Words (BoW)** and **TF-IDF** text representations from scratch. These are foundational NLP techniques for converting text to numerical vectors.

Your task is to:

1. Build a vocabulary from a corpus
2. Implement Bag-of-Words encoding
3. Implement TF-IDF weighting
4. Support document similarity computation

## Definitions

**Bag-of-Words (BoW):**
Count of each word in the document (ignores order).

**TF-IDF (Term Frequency - Inverse Document Frequency):**
```
TF(t, d) = count(t in d) / total_words(d)
IDF(t) = log(N / df(t))  where df(t) = docs containing t
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

## Function Signature

```python
class BagOfWords:
    def __init__(self):
        pass
    
    def fit(self, documents: list[str]):
        """Build vocabulary from documents."""
        pass
    
    def transform(self, documents: list[str]) -> torch.Tensor:
        """Convert documents to BoW vectors."""
        pass

class TFIDF:
    def __init__(self):
        pass
    
    def fit(self, documents: list[str]):
        """Compute IDF values from corpus."""
        pass
    
    def transform(self, documents: list[str]) -> torch.Tensor:
        """Convert documents to TF-IDF vectors."""
        pass
```

## Example

```python
documents = [
    "the cat sat on the mat",
    "the dog ate my homework",
    "cats and dogs are pets"
]

# Bag of Words
bow = BagOfWords()
bow.fit(documents)
vectors = bow.transform(documents)
# Shape: (3, vocab_size)

# TF-IDF
tfidf = TFIDF()
tfidf.fit(documents)
vectors = tfidf.transform(documents)
```

## Hints

- Tokenize by splitting on whitespace (simple) or use regex
- Use set for vocabulary, dict for word→index mapping
- IDF: add 1 to denominator to avoid division by zero
- Sparse representation is more memory efficient for large vocab
