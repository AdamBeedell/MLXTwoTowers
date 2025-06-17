import torch
from torch import nn as nn
from torch.nn import functional as F


class Tower(nn.Module):
    def __init__(self, embeddings, embed_dim, dropout_rate):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embeddings)
        self.embedding.weight.requires_grad = self.train_embeddings()
        shrink = int(embed_dim / 2)
        self.inp = nn.Linear(embed_dim, shrink)
        self.middle = nn.Linear(shrink, embed_dim)
        self.hidden = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.inp.weight)
        nn.init.xavier_uniform_(self.middle.weight)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, x):
        emb = self.embedding(x)
        batch_size, seq_len, embed_dim = emb.size()
        h = torch.zeros(batch_size, embed_dim, device=emb.device)
        for t in range(seq_len):
            token_emb = emb[:, t, :]
            lin = torch.relu(self.inp(token_emb))
            lin = torch.relu(self.middle(lin))
            h = torch.tanh(lin + self.hidden(h))
        dropout = self.dropout(h)
        return F.normalize(dropout, p=2, dim=1)


class QueryTower(Tower):
    def __init__(self, embeddings, embed_dim, dropout_rate):
        super().__init__(embeddings, embed_dim, dropout_rate)

    @classmethod
    def train_embeddings(cls):
        return True


class DocTower(Tower):
    def __init__(self, embeddings, embed_dim, dropout_rate):
        super().__init__(embeddings, embed_dim, dropout_rate)

    @classmethod
    def train_embeddings(cls):
        return False
