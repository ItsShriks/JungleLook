"""Scenes, spatial splits and block datasets for point-wise segmentation.

A *scene* is an ``.npz`` with ``xyz`` (float64), ``feats`` (N, F float32) and
``labels`` (N int64, -1 = ignore). Networks see square XY *blocks* of a scene:

* :class:`TrainBlocks` samples random blocks (optionally class-balanced centres,
  so rare classes such as stumps are seen often) with augmentation.
* :class:`EvalBlocks` tiles the whole scene with a sliding window and chunks every
  block so **every point** is predicted at least once; logits are averaged
  (voted) back onto the original points.

Block input channels: ``[x - cx, y - cy, z - z_min, *standardised features]`` in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

IGNORE = -1


@dataclass
class Scene:
    xyz: np.ndarray                  # (N, 3) float64
    feats: np.ndarray                # (N, F) float32, already standardised
    labels: Optional[np.ndarray]     # (N,) int64 or None

    @classmethod
    def load(cls, path: str, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> "Scene":
        z = np.load(path)
        feats = z["feats"].astype(np.float32)
        if mean is not None:
            feats = (feats - mean) / std
        return cls(z["xyz"], feats, z["labels"] if "labels" in z.files else None)

    def __len__(self) -> int:
        return len(self.xyz)


def spatial_split(xyz: np.ndarray, tile: float, ratios: Sequence[float], seed: int = 0) -> np.ndarray:
    """Assign whole ``tile`` x ``tile`` m squares to train/val/test (0/1/2).

    Splitting by area instead of by point avoids leakage: neighbouring points are
    nearly identical, so a random point split would overstate accuracy.
    """
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(axis=0)) / tile).astype(np.int64)
    _, tile_id, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    tile_id = tile_id.ravel()
    order = np.random.default_rng(seed).permutation(len(counts))
    frac = counts[order] / counts.sum()
    midpoints = np.cumsum(frac) - frac / 2          # tile's position in the shuffled point budget
    bounds = np.cumsum(ratios)[:-1] / np.sum(ratios)
    split_of_tile = np.empty(len(counts), dtype=np.int64)
    split_of_tile[order] = np.searchsorted(bounds, midpoints)
    return split_of_tile[tile_id]


def class_weights(labels: np.ndarray, num_classes: int, mode: str = "log") -> np.ndarray:
    counts = np.bincount(labels[labels >= 0], minlength=num_classes).astype(np.float64)
    freq = counts / max(counts.sum(), 1)
    if mode == "none":
        w = np.ones(num_classes)
    elif mode == "inverse_sqrt":
        w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    else:  # ENet-style 1 / ln(1.02 + f): strong for rare classes but bounded
        w = 1.0 / np.log(1.02 + freq)
    w[counts == 0] = 0.0
    return (w / w[counts > 0].mean()).astype(np.float32)


def _block_input(xyz: np.ndarray, feats: np.ndarray, centre: np.ndarray, z_min: float) -> np.ndarray:
    local = np.empty((len(xyz), 3), dtype=np.float32)
    local[:, :2] = xyz[:, :2] - centre[:2]
    local[:, 2] = xyz[:, 2] - z_min
    return np.concatenate([local, feats], axis=1)


class TrainBlocks(Dataset):
    def __init__(self, scene: Scene, block_size: float, num_points: int, samples_per_epoch: int,
                 class_balanced: bool = True, min_points: int = 128, augment: bool = True,
                 num_classes: Optional[int] = None):
        assert scene.labels is not None, "training needs labels"
        self.s, self.block, self.n = scene, block_size, num_points
        self.samples, self.min_points, self.augment = samples_per_epoch, min_points, augment
        self.tree = cKDTree(scene.xyz[:, :2])
        labels = scene.labels
        self.by_class = [np.flatnonzero(labels == c) for c in range(num_classes or labels.max() + 1)]
        self.by_class = [ix for ix in self.by_class if len(ix)] if class_balanced else \
            [np.flatnonzero(labels != IGNORE)]

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng()
        for _ in range(20):
            pool = self.by_class[rng.integers(len(self.by_class))]
            centre = self.s.xyz[pool[rng.integers(len(pool))]]
            idx = np.asarray(self.tree.query_ball_point(centre[:2], self.block / 2, p=np.inf))
            if len(idx) >= self.min_points:
                break
        z_min = self.s.xyz[idx, 2].min()
        idx = rng.choice(idx, self.n, replace=len(idx) < self.n)
        x = _block_input(self.s.xyz[idx], self.s.feats[idx], centre, z_min)
        if self.augment:
            x = augment_block(x, rng)
        return torch.from_numpy(x), torch.from_numpy(self.s.labels[idx].astype(np.int64))


def augment_block(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation about z, mirror, scale and jitter of the xyz channels only.

    The engineered features are z-rotation invariant, so they are left untouched.
    """
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    x[:, :2] = x[:, :2] @ np.array([[c, -s], [s, c]], dtype=np.float32).T
    if rng.random() < 0.5:
        x[:, 0] = -x[:, 0]
    x[:, :3] *= rng.uniform(0.9, 1.1)
    x[:, :3] += np.clip(rng.normal(0, 0.01, size=(len(x), 3)), -0.03, 0.03).astype(np.float32)
    return x


class EvalBlocks(Dataset):
    """Deterministic sliding-window chunks that cover every point of the scene."""

    def __init__(self, scene: Scene, block_size: float, num_points: int, stride: Optional[float] = None,
                 min_points: int = 1):
        self.s, self.block, self.n = scene, block_size, num_points
        stride = stride or block_size
        tree = cKDTree(scene.xyz[:, :2])
        lo, hi = scene.xyz[:, :2].min(axis=0), scene.xyz[:, :2].max(axis=0)
        xs = np.arange(lo[0] + block_size / 2, max(hi[0], lo[0] + block_size / 2) + stride, stride)
        ys = np.arange(lo[1] + block_size / 2, max(hi[1], lo[1] + block_size / 2) + stride, stride)
        rng = np.random.default_rng(0)
        self.chunks: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for cx in xs:
            for cy in ys:
                centre = np.array([cx, cy])
                idx = np.asarray(tree.query_ball_point(centre, block_size / 2 + 1e-6, p=np.inf), dtype=np.int64)
                if len(idx) < min_points:
                    continue
                z_min = float(scene.xyz[idx, 2].min())
                idx = rng.permutation(idx)
                for s in range(0, len(idx), num_points):
                    chunk = idx[s:s + num_points]
                    if len(chunk) < num_points:   # pad with other points of the same block
                        chunk = np.concatenate([chunk, rng.choice(idx, num_points - len(chunk))])
                    self.chunks.append((chunk, centre, z_min))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int):
        idx, centre, z_min = self.chunks[i]
        x = _block_input(self.s.xyz[idx], self.s.feats[idx], centre, z_min)
        return torch.from_numpy(x), torch.from_numpy(idx)


def feature_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    return mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32)


def save_scene(path: str, xyz: np.ndarray, feats: np.ndarray, labels: Optional[np.ndarray],
               extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    arrays = {"xyz": xyz, "feats": feats}
    if labels is not None:
        arrays["labels"] = labels
    arrays.update(extra or {})
    np.savez_compressed(path, **arrays)
