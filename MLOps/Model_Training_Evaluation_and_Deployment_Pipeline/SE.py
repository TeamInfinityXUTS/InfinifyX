import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, r=16):
        super().__init__()
        self.r = r
        self.built = False
        self.avg = nn.AdaptiveAvgPool2d(1)

    def _build(self, c):
        hidden = max(1, c // self.r)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c),
            nn.Sigmoid()
        )
        self.built = True

    def forward(self, x):
        b, c, _, _ = x.shape
        if not self.built:
            self._build(c)
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
