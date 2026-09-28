"""Label-free geometric feature extraction for forest point clouds.

* ``voxel_downsample``      – uniform density (UAV clouds are very uneven)
* ``DTM``                    – terrain model from a morphologically opened min-z grid
* ``eigen_features``         – covariance shape descriptors at one or more k-NN scales
* ``column_features``        – vertical context (max height above ground in an XY column)
* ``extract_features``       – everything above, driven by the ``features`` config section

All features except the raw coordinates are invariant to rotation about the z-axis,
so random z-rotation augmentation during training stays consistent with them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def voxel_downsample(xyz: np.ndarray, voxel: float, seed: int = 0) -> np.ndarray:
    """Indices of one random point per occupied voxel."""
    if not voxel or voxel <= 0:
        return np.arange(len(xyz))
    perm = np.random.default_rng(seed).permutation(len(xyz))
    keys = np.floor((xyz[perm] - xyz.min(axis=0)) / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(perm[first])


# --------------------------------------------------------------------------- terrain
@dataclass
class DTM:
    origin: np.ndarray   # (2,) xy of cell (0, 0) corner
    cell: float
    grid: np.ndarray     # (nx, ny) terrain elevation per cell centre

    @classmethod
    def fit(cls, xyz: np.ndarray, cell: float = 0.5, windows: Sequence[float] = (1.0, 2.0, 4.0),
            dh0: float = 0.15, slope: float = 0.3, smooth: float = 1.0) -> "DTM":
        """Progressive morphological filter (Zhang et al., 2003) on a min-z raster.

        1. lowest z per ``cell`` (empty cells filled from their nearest neighbour)
        2. for growing ``windows`` (m): grey-open the surface; a cell whose height drops
           by more than ``dh0 + slope * (w_k - w_{k-1})`` is marked non-ground.
           Small windows remove stumps, larger ones shrubs and logs, while the
           slope term keeps genuine terrain relief.
        3. non-ground cells are re-filled from the nearest ground cell, then smoothed.
        """
        origin = xyz[:, :2].min(axis=0)
        ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
        shape = tuple(ij.max(axis=0) + 1)
        zmin = np.full(shape, np.inf)
        np.minimum.at(zmin, (ij[:, 0], ij[:, 1]), xyz[:, 2])
        zmin = _fill_nearest(np.where(np.isinf(zmin), np.nan, zmin))
        closed = ndimage.grey_closing(zmin, size=(3, 3))   # single-cell pits = low noise
        zmin = np.where(closed - zmin > dh0, closed, zmin)

        surface, nonground, w_prev = zmin, np.zeros(shape, dtype=bool), 0.0
        for k, w in enumerate(sorted(windows)):
            size = max(3, int(round(w / cell)) | 1)
            opened = ndimage.grey_opening(surface, size=(size, size))
            dh = dh0 if k == 0 else dh0 + slope * (w - w_prev)
            nonground |= (surface - opened) > dh
            surface, w_prev = opened, w
        dtm = _fill_nearest(np.where(nonground, np.nan, zmin))
        if smooth > 0:
            dtm = ndimage.gaussian_filter(dtm, sigma=smooth)
        return cls(origin=origin, cell=cell, grid=dtm)

    def height(self, xyz: np.ndarray) -> np.ndarray:
        """Height above ground (bilinear interpolation of the DTM)."""
        return xyz[:, 2] - self.elevation(xyz[:, :2])

    def elevation(self, xy: np.ndarray) -> np.ndarray:
        coords = ((xy - self.origin) / self.cell - 0.5).T
        return ndimage.map_coordinates(self.grid, coords, order=1, mode="nearest")


def _fill_nearest(grid: np.ndarray) -> np.ndarray:
    invalid = np.isnan(grid)
    if not invalid.any():
        return grid
    if invalid.all():
        raise ValueError("DTM grid has no valid cells")
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


# --------------------------------------------------------------------------- shape
EIGEN_FEATURES = ("linearity", "planarity", "scattering", "curvature", "verticality",
                  "normal_z", "roughness", "density")


def eigen_features(xyz: np.ndarray, k: int = 20, tree: cKDTree | None = None,
                   chunk: int = 100_000, workers: int = -1) -> Dict[str, np.ndarray]:
    """Covariance (eigenvalue) descriptors of the k-nearest-neighbour patch of each point.

    With eigenvalues l1 >= l2 >= l3 and normal n (eigenvector of l3):
      linearity (l1-l2)/l1   – branches, stems, fallen logs
      planarity (l2-l3)/l1   – ground, cut stump tops
      scattering l3/l1       – foliage, brush
      curvature  l3/(l1+l2+l3)  (surface variation)
      verticality 1-|n_z|,  normal_z |n_z|
      roughness  distance of the point to its local plane
      density    log(k / volume of the k-NN sphere)
    """
    tree = tree or cKDTree(xyz)
    out = {name: np.empty(len(xyz), dtype=np.float32) for name in EIGEN_FEATURES}
    for s in range(0, len(xyz), chunk):
        pts = xyz[s:s + chunk]
        dist, nn = tree.query(pts, k=k, workers=workers)
        nbrs = xyz[nn]                                     # (m, k, 3)
        mean = nbrs.mean(axis=1, keepdims=True)
        centred = nbrs - mean
        cov = np.einsum("mki,mkj->mij", centred, centred) / k
        evals, evecs = np.linalg.eigh(cov)                 # ascending
        evals = np.clip(evals, 1e-12, None)
        l3, l2, l1 = evals[:, 0], evals[:, 1], evals[:, 2]
        normal = evecs[:, :, 0]
        sl = slice(s, s + len(pts))
        out["linearity"][sl] = (l1 - l2) / l1
        out["planarity"][sl] = (l2 - l3) / l1
        out["scattering"][sl] = l3 / l1
        out["curvature"][sl] = l3 / (l1 + l2 + l3)
        out["normal_z"][sl] = np.abs(normal[:, 2])
        out["verticality"][sl] = 1.0 - np.abs(normal[:, 2])
        out["roughness"][sl] = np.abs(np.einsum("mi,mi->m", pts - mean[:, 0], normal))
        radius = np.maximum(dist[:, -1], 1e-6)
        out["density"][sl] = np.log(k / (4.0 / 3.0 * np.pi * radius ** 3))
    return out


def column_features(xyz: np.ndarray, hag: np.ndarray, cell: float = 0.5,
                    gap: float = 0.4) -> Dict[str, np.ndarray]:
    """Vertical context per XY column of size ``cell``.

    * ``column_max_hag``     – tallest point in the column (canopy height)
    * ``column_stack_hag``   – top of the gap-free stack of points rising from the ground
                               (vertical gaps > ``gap`` m end it). A trunk continues into
                               the crown; a stump *under* a crown stops at ~0.5 m.
    * ``column_rel_height``  – hag / column_max_hag
    """
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(axis=0)) / cell).astype(np.int64)
    flat = np.ravel_multi_index((ij[:, 0], ij[:, 1]), tuple(ij.max(axis=0) + 1))
    uniq, inv = np.unique(flat, return_inverse=True)
    col_max = np.full(len(uniq), -np.inf)
    np.maximum.at(col_max, inv, hag)

    order = np.lexsort((hag, inv))
    h, c = hag[order], inv[order]
    new_col = np.r_[True, c[1:] != c[:-1]]
    breaks = new_col | np.r_[False, np.diff(h) > gap]
    segment = np.cumsum(breaks) - 1
    first_segment = segment[new_col][c]              # id of each column's lowest segment
    in_first = segment == first_segment
    stack = np.full(len(uniq), -np.inf)
    np.maximum.at(stack, c[in_first], h[in_first])

    col_max = col_max[inv].astype(np.float32)
    rel = np.where(col_max > 1e-3, hag / np.maximum(col_max, 1e-3), 0.0)
    return {"column_max_hag": col_max, "column_stack_hag": stack[inv].astype(np.float32),
            "column_rel_height": np.clip(rel, -1, 1).astype(np.float32)}


# --------------------------------------------------------------------------- driver
def extract_features(xyz: np.ndarray, cfg: dict) -> Tuple[Dict[str, np.ndarray], DTM]:
    """Compute every feature listed in the ``features`` config section.

    Returns a dict of (N,) float32 arrays and the fitted DTM. Multi-scale eigen
    features are suffixed with their k, e.g. ``planarity_k20``.
    """
    dcfg = cfg.get("dtm", {})
    dtm = DTM.fit(xyz, cell=dcfg.get("cell", 0.5), windows=dcfg.get("windows", (1.0, 2.0, 4.0)),
                  dh0=dcfg.get("dh0", 0.15), slope=dcfg.get("slope", 0.3),
                  smooth=dcfg.get("smooth", 1.0))
    feats: Dict[str, np.ndarray] = {"hag": dtm.height(xyz).astype(np.float32)}
    feats.update(column_features(xyz, feats["hag"], cfg.get("column_cell", 0.5), cfg.get("column_gap", 0.4)))

    tree = cKDTree(xyz)
    for k in cfg.get("scales", [20]):
        for name, v in eigen_features(xyz, k=int(k), tree=tree).items():
            feats[f"{name}_k{k}"] = v
    return feats, dtm


def feature_matrix(feats: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    missing = [n for n in names if n not in feats]
    if missing:
        raise KeyError(f"features not available: {missing}; have {sorted(feats)}")
    return np.stack([feats[n] for n in names], axis=1).astype(np.float32)


def default_feature_names(cfg: dict) -> List[str]:
    scales = cfg.get("scales", [20])
    names = ["hag", "column_max_hag", "column_stack_hag", "column_rel_height"]
    for k in scales:
        names += [f"{n}_k{k}" for n in EIGEN_FEATURES]
    return names
