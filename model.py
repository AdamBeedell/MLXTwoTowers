from torch import nn as nn
from torch.nn import functional as F


class Tower(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        return F.normalize(x, p=2, dim=1)
