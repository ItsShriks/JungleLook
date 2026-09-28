"""PointNet semantic segmentation (Qi et al., CVPR 2017).

Input  : (B, N, 3 + F) – local xyz followed by F per-point features
Output : (B, N, num_classes) logits, plus the 64x64 feature transform for the
         orthogonality regulariser.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


def _fc(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(cin, cout, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class TNet(nn.Module):
    """Predicts a k x k alignment matrix, initialised to the identity."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(_conv(k, 64), _conv(64, 128), _conv(128, 1024))
        self.head = nn.Sequential(_fc(1024, 512), _fc(512, 256), nn.Linear(256, k * k))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, k, N)
        g = self.encoder(x).max(dim=2).values
        m = self.head(g).view(-1, self.k, self.k)
        return m + torch.eye(self.k, device=x.device, dtype=x.dtype)


class PointNetSeg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, feature_transform: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.extra = in_channels - 3
        self.input_tnet = TNet(3)
        self.mlp1 = nn.Sequential(_conv(in_channels, 64), _conv(64, 64))
        self.feature_tnet = TNet(64) if feature_transform else None
        self.mlp2 = nn.Sequential(_conv(64, 64), _conv(64, 128), _conv(128, 1024))
        self.head = nn.Sequential(
            _conv(64 + 1024, 512), _conv(512, 256), _conv(256, 128),
            nn.Dropout(dropout), nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor):
        n = x.shape[1]
        x = x.transpose(1, 2)                                     # (B, C, N)
        xyz, extra = x[:, :3], x[:, 3:]
        xyz = torch.bmm(self.input_tnet(xyz).transpose(1, 2), xyz)
        x = self.mlp1(torch.cat([xyz, extra], dim=1))
        trans = None
        if self.feature_tnet is not None:
            trans = self.feature_tnet(x)
            x = torch.bmm(trans.transpose(1, 2), x)
        local = x
        g = self.mlp2(x).max(dim=2, keepdim=True).values.expand(-1, -1, n)
        logits = self.head(torch.cat([local, g], dim=1))
        return logits.transpose(1, 2), trans


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """|| I - A A^T ||_F, keeps the 64x64 transform close to orthogonal."""
    eye = torch.eye(trans.shape[1], device=trans.device, dtype=trans.dtype)
    return torch.linalg.matrix_norm(eye - torch.bmm(trans, trans.transpose(1, 2))).mean()
