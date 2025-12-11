"""
Word2Vec from Scratch - Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import random


class SkipGram(nn.Module):
    """Skip-gram with negative sampling."""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        
        nn.init.xavier_uniform_(self.center_embed.weight)
        nn.init.zeros_(self.context_embed.weight)
    
    def forward(self, center, context, negatives=None):
        center_emb = self.center_embed(center)  # (batch, dim)
        context_emb = self.context_embed(context)  # (batch, dim)
        
        pos_score = torch.sum(center_emb * context_emb, dim=1)
        pos_loss = F.logsigmoid(pos_score)
        
        if negatives is not None:
            neg_emb = self.context_embed(negatives)  # (batch, num_neg, dim)
            neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
            neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        else:
            neg_loss = 0
        
        return -(pos_loss + neg_loss).mean()


class Word2Vec:
    def __init__(self, sentences, embed_dim=100, window=5, min_count=5):
        self.embed_dim = embed_dim
        self.window = window
        self.vocab = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        self._build_vocab(sentences, min_count)
        self.model = SkipGram(len(self.vocab), embed_dim)
        self.training_data = self._generate_training_data(sentences)
        
        # Negative sampling distribution (unigram^0.75)
        counts = torch.tensor([self.word_counts.get(self.idx2word[i], 1) for i in range(len(self.vocab))])
        self.neg_dist = (counts ** 0.75) / (counts ** 0.75).sum()
    
    def _build_vocab(self, sentences, min_count):
        for sent in sentences:
            self.word_counts.update(sent)
        
        vocab_words = [w for w, c in self.word_counts.items() if c >= min_count]
        self.vocab = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word = {i: w for w, i in self.vocab.items()}
    
    def _generate_training_data(self, sentences):
        pairs = []
        for sent in sentences:
            indices = [self.vocab[w] for w in sent if w in self.vocab]
            for i, center in enumerate(indices):
                start = max(0, i - self.window)
                end = min(len(indices), i + self.window + 1)
                for j in range(start, end):
                    if i != j:
                        pairs.append((center, indices[j]))
        return pairs
    
    def _sample_negatives(self, batch_size, num_neg):
        return torch.multinomial(self.neg_dist, batch_size * num_neg, replacement=True).view(batch_size, num_neg)
    
    def train(self, epochs=5, batch_size=256, neg_samples=5, lr=0.025):
        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            random.shuffle(self.training_data)
            total_loss = 0
            
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i+batch_size]
                center = torch.tensor([p[0] for p in batch])
                context = torch.tensor([p[1] for p in batch])
                negatives = self._sample_negatives(len(batch), neg_samples)
                
                optimizer.zero_grad()
                loss = self.model(center, context, negatives)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss / (len(self.training_data) // batch_size):.4f}")
    
    def get_embedding(self, word):
        if word not in self.vocab:
            return None
        return self.model.center_embed.weight[self.vocab[word]].detach()
    
    def most_similar(self, word, k=10):
        if word not in self.vocab:
            return []
        
        word_emb = self.get_embedding(word)
        all_emb = self.model.center_embed.weight.detach()
        
        sims = F.cosine_similarity(word_emb.unsqueeze(0), all_emb)
        topk = sims.topk(k + 1)
        
        return [(self.idx2word[i.item()], s.item()) for i, s in zip(topk.indices[1:], topk.values[1:])]


if __name__ == "__main__":
    sentences = [["the", "cat", "sat", "on", "mat"]] * 100 + \
                [["the", "dog", "ran", "in", "park"]] * 100 + \
                [["cat", "and", "dog", "are", "pets"]] * 50
    
    w2v = Word2Vec(sentences, embed_dim=50, window=2, min_count=1)
    w2v.train(epochs=20, lr=0.01)
    
    print("\nSimilar to 'cat':", w2v.most_similar("cat", k=3))
    print("Similar to 'dog':", w2v.most_similar("dog", k=3))
