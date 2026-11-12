import torch
import torch.nn as nn
from block.DenseNet import ResidualDenseBlock_out as DB


class Noise_INN_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__()
        self.clamp = clamp
        self.r = DB(input=12, output=12)
        self.y = DB(input=12, output=12)
        self.f = DB(input=12, output=12)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = x[0], x[1]
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = torch.exp(s1) * x2 + t1
            out = [y1, y2]
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)
            t2 = self.f(y2)
            y1 = x1 - t2
            out = [y1, y2]
        return out
