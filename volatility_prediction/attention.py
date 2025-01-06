import torch
import torch.nn as nn
import torch.nn.functional as F

n_stocks = 112


class TimeAttention(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.weights = nn.Parameter(torch.zeros(steps))

    def forward(self, x):
        # x: (b, st, t, f)
        attn = F.softmax(self.weights, 0)
        x = torch.einsum("b s t f, t -> b s f", x, attn)
        return x


class StockAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((n_stocks, n_stocks)))
        self.bias = nn.Parameter(torch.zeros(n_stocks))
        self.fc_combine = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x: (b, st, t, f)
        attn = F.softmax(self.weight + self.bias[None, :], dim=-1)  # (st, st)
        y = torch.einsum("b i ..., j i -> b j ...", x, attn)
        x = torch.cat((x, y), -1)
        x = self.fc_combine(x)
        return x
