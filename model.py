import torch
from torch import nn as nn
from torch.nn import functional as F


class Tower(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input = nn.Linear(embed_dim, embed_dim)
        self.hidden = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        emb = self.embedding(x)
        batch_size, seq_len, embed_dim = emb.size()
        h = torch.zeros(batch_size, embed_dim, device=emb.device)
        for t in range(seq_len):
            token_emb = emb[:, t, :]
            h = torch.tanh(self.input(token_emb) + self.hidden(h))
        dropout = self.dropout(h)
        return F.normalize(dropout, p=2, dim=1)
