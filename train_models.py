import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

class DocTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

queryTower = QueryTower()
docTower = DocTower()

query = torch.randn(1, 10)
pos = torch.randn(1, 10)
neg = torch.randn(1, 10)

query = queryTower(query)
pos = docTower(pos)
neg = docTower(neg)

dst_pos = F.cosine_similarity(query, pos)
dst_neg = F.cosine_similarity(query, neg)
dst_diff = dst_pos - dst_neg
dst_mrg = torch.tensor(0.2)

loss = torch.max(torch.tensor(0.0), dst_diff + dst_mrg)
loss.backward()