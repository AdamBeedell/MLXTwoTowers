from torch import nn as nn
from torch.nn import functional as F


class Tower(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        x_sum = (emb * mask).sum(dim=1)
        x_len = mask.sum(dim=1).clamp(min=1e-9)
        avg = x_sum / x_len
        return F.normalize(avg, p=2, dim=1)
