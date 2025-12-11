# Word2Vec from Scratch

## Problem Statement

Implement **Word2Vec** from scratch. Word2Vec learns dense word embeddings by predicting context words (Skip-gram) or predicting a word from its context (CBOW).

Your task is to:

1. Implement Skip-gram model
2. Implement CBOW model  
3. Use negative sampling for efficiency
4. Train on a text corpus

## Word2Vec Models

**Skip-gram:** Given center word, predict context words
```
"the [cat] sat" → predict: the, sat
```

**CBOW:** Given context words, predict center word
```
"the ??? sat" → predict: cat
```

## Function Signature

```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        pass
    
    def forward(self, center: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute similarity between center and context words."""
        pass

class Word2Vec:
    def __init__(self, sentences: list[list[str]], embed_dim: int = 100, window: int = 5):
        pass
    
    def train(self, epochs: int = 5, neg_samples: int = 5):
        pass
    
    def get_embedding(self, word: str) -> torch.Tensor:
        pass
    
    def most_similar(self, word: str, k: int = 10) -> list[str]:
        pass
```

## Negative Sampling Loss

Instead of softmax over all words:
```
L = -log σ(v_c · v_w) - Σ log σ(-v_c · v_neg)
```

## Example

```python
sentences = [["the", "cat", "sat"], ["dogs", "are", "pets"], ...]

w2v = Word2Vec(sentences, embed_dim=100, window=5)
w2v.train(epochs=5)

similar = w2v.most_similar("cat", k=5)
# ['dog', 'kitten', 'pet', ...]
```

## Hints

- `nn.Embedding` for word embeddings
- Sample negatives based on word frequency (unigram^0.75)
- Use `torch.sigmoid` for negative sampling
- Subsampling frequent words helps training
