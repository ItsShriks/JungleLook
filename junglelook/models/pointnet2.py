"""PointNet++ single-scale-grouping segmentation (Qi et al., NeurIPS 2017) in pure PyTorch.

No custom CUDA ops, so it runs on CPU, CUDA and Apple MPS. Radii are in metres
because block coordinates are metric (see ``data.py``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def square_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) x (B, M, 3) -> (B, N, M) squared distances."""
    return (a.pow(2).sum(-1, keepdim=True) - 2 * a @ b.transpose(1, 2)
            + b.pow(2).sum(-1).unsqueeze(1)).clamp_min(0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points (B, N, C), idx (B, ...) -> (B, ..., C)."""
    b = torch.arange(points.shape[0], device=points.device).view(-1, *([1] * (idx.dim() - 1)))
    return points[b, idx]


def farthest_point_sample(xyz: torch.Tensor, npoint: int, random_start: bool = False) -> torch.Tensor:
    b, n, _ = xyz.shape
    idx = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, n), float("inf"), device=xyz.device)
    far = torch.randint(0, n, (b,), device=xyz.device) if random_start \
        else torch.zeros(b, dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = far
        d = (xyz - xyz[batch, far].unsqueeze(1)).pow(2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Up to ``nsample`` neighbour indices within ``radius``; short groups repeat their nearest point."""
    n = xyz.shape[1]
    d = square_distance(centres, xyz)                                   # (B, S, N)
    d = d.masked_fill(d > radius ** 2, float("inf"))
    dist, idx = d.topk(min(nsample, n), dim=-1, largest=False)
    first = idx[..., :1].expand_as(idx)
    idx = torch.where(torch.isinf(dist), first, idx)
    if idx.shape[-1] < nsample:
        idx = torch.cat([idx, first[..., :1].expand(-1, -1, nsample - idx.shape[-1])], dim=-1)
    return idx


def _mlp2d(channels: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for cin, cout in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _mlp1d(channels: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for cin, cout in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channels: int, mlp: Sequence[int]):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.mlp = _mlp2d([in_channels + 3, *mlp])

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor]):
        # xyz (B, N, 3), feats (B, N, C) -> new_xyz (B, S, 3), new_feats (B, S, C')
        centres = index_points(xyz, farthest_point_sample(xyz, min(self.npoint, xyz.shape[1]), self.training))
        idx = ball_query(self.radius, self.nsample, xyz, centres)
        grouped = index_points(xyz, idx) - centres.unsqueeze(2)          # (B, S, K, 3)
        grouped = grouped / self.radius
        if feats is not None:
            grouped = torch.cat([grouped, index_points(feats, idx)], dim=-1)
        out = self.mlp(grouped.permute(0, 3, 2, 1)).max(dim=2).values     # (B, C', S)
        return centres, out.transpose(1, 2)


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels: int, mlp: Sequence[int]):
        super().__init__()
        self.mlp = _mlp1d([in_channels, *mlp])

    def forward(self, xyz_dense, xyz_sparse, feats_dense, feats_sparse):
        d = square_distance(xyz_dense, xyz_sparse)
        k = min(3, xyz_sparse.shape[1])
        dist, idx = d.topk(k, dim=-1, largest=False)
        w = 1.0 / (dist + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        interp = (index_points(feats_sparse, idx) * w.unsqueeze(-1)).sum(2)
        x = interp if feats_dense is None else torch.cat([feats_dense, interp], dim=-1)
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


class PointNet2Seg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 npoints: Sequence[int] = (1024, 256, 64, 16),
                 radii: Sequence[float] = (0.2, 0.4, 0.8, 1.6),
                 nsample: int = 32, dropout: float = 0.5):
        super().__init__()
        f = in_channels                     # local xyz is fed as a point feature too (S3DIS setup)
        widths = [(32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)]
        self.sa = nn.ModuleList()
        cin = f
        for np_, r, w in zip(npoints, radii, widths):
            self.sa.append(SetAbstraction(np_, r, nsample, cin, w))
            cin = w[-1]
        skip = [f] + [w[-1] for w in widths[:-1]]                        # channels at each level
        self.fp = nn.ModuleList()
        cin = widths[-1][-1]
        for level in reversed(range(len(widths))):
            out = (256, 256) if level > 1 else (256, 128) if level == 1 else (128, 128, 128)
            self.fp.append(FeaturePropagation(cin + skip[level], out))
            cin = out[-1]
        self.head = nn.Sequential(_mlp1d([128, 128]), nn.Dropout(dropout), nn.Conv1d(128, num_classes, 1))

    def forward(self, x: torch.Tensor):
        xyz = x[..., :3].contiguous()
        feats = x.contiguous()
        xyzs, featss = [xyz], [feats]
        for sa in self.sa:
            xyz, feats = sa(xyz, feats)
            xyzs.append(xyz)
            featss.append(feats)
        for i, fp in enumerate(self.fp):
            level = len(self.sa) - 1 - i
            feats = fp(xyzs[level], xyzs[level + 1], featss[level], feats)
        logits = self.head(feats.transpose(1, 2)).transpose(1, 2)
        return logits, None
