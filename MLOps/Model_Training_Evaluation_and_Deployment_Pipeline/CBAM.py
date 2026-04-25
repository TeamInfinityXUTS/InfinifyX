import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, r=16):
        super().__init__()
        self.r = r
        self.built = False

    def _build(self, c):
        hidden = max(1, c // self.r)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.built = True

    def forward(self, x):
        b, c, h, w = x.shape
        if not self.built:
            self._build(c)

        avg = x.mean((2, 3))
        mx = x.amax((2, 3))

        channel = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * channel

        avg_p = x.mean(1, keepdim=True)
        max_p = x.max(1, keepdim=True)[0]

        spatial = torch.cat([avg_p, max_p], dim=1)
        spatial = torch.sigmoid(self.spatial(spatial))

        return x * spatial
