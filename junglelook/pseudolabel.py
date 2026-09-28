"""Rule-based pseudo-labels from geometric features (no annotation needed).

Classes (default ``[ground, stump, other]``):

* ground – low height above the DTM
* stump  – compact, round, short clusters standing on the ground whose vertical
           stack of points ends low (a continuous stack means a tree base or shrub)
* other  – every other off-ground point (vegetation, logs, trees, debris)

Points in an uncertain height band just above the ground are marked ``IGNORE``
(-1) so the network is not trained on the least reliable rule decisions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

IGNORE = -1

DEFAULTS = {
    "classes": ["ground", "stump", "other"],
    "ground_max_hag": 0.12,
    "ignore_band": 0.08,           # [ground_max_hag, ground_max_hag + band) -> IGNORE
    "stump": {
        "min_hag": 0.05, "max_hag": 1.5,
        "max_stack_hag": 1.0,      # continuous vertical stack above => tree base / shrub, not a stump
        "eps": 0.08, "min_samples": 8, "min_points": 80,
        "height": [0.12, 0.8],     # m above ground
        "diameter": [0.3, 2.5],    # m (2 * 90th percentile XY radius, includes the root collar)
        "min_roundness": 0.3,      # sqrt(l2 / l1) of the XY covariance: 1 = circle, 0 = line
        "max_slenderness": 4.0,    # height / diameter (excludes saplings and poles)
        "max_scattering": 0.25,    # mean scattering: stumps are solid, brush is diffuse
    },
    # Replace rule-based stumps by reference stumps (needs a `reference` section).
    # Use it to measure how much the pseudo-labels cost, or when annotations exist.
    "stumps_from_reference": False,
}


def _merge(defaults: dict, cfg: Optional[dict]) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in defaults.items()}
    for k, v in (cfg or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cluster_stats(xyz: np.ndarray, hag: np.ndarray, cluster: np.ndarray,
                  scattering: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Per-cluster shape statistics (cluster ids >= 0; -1 is noise)."""
    rows = []
    for cid in np.unique(cluster[cluster >= 0]):
        m = cluster == cid
        p, h = xyz[m], hag[m]
        xy = p[:, :2] - p[:, :2].mean(axis=0)
        evals = np.sort(np.linalg.eigvalsh(np.cov(xy.T) + 1e-9 * np.eye(2)))
        radius = np.percentile(np.linalg.norm(xy, axis=1), 90)
        base = p[:, 2] - h                      # terrain elevation below each point
        rows.append({
            "id": int(cid), "n_points": int(m.sum()),
            "x": float(p[:, 0].mean()), "y": float(p[:, 1].mean()), "ground_z": float(np.median(base)),
            "height": float(np.percentile(h, 98)),
            "diameter": float(2 * radius),
            "roundness": float(np.sqrt(evals[0] / evals[1])),
            "scattering": float(scattering[m].mean()) if scattering is not None else np.nan,
        })
    df = pd.DataFrame(rows, columns=["id", "n_points", "x", "y", "ground_z", "height", "diameter",
                                     "roundness", "scattering"])
    df["slenderness"] = df["height"] / df["diameter"].clip(lower=1e-3)
    return df


def detect_stumps(xyz: np.ndarray, feats: Dict[str, np.ndarray], cfg: dict,
                  scattering_key: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Returns (instance id per point, -1 = not a stump; table of all candidate clusters)."""
    s = cfg
    hag, stack = feats["hag"], feats["column_stack_hag"]
    cand = np.flatnonzero((hag >= s["min_hag"]) & (hag <= s["max_hag"]) & (stack <= s["max_stack_hag"]))
    instance = np.full(len(xyz), -1, dtype=np.int64)
    if len(cand) < s["min_points"]:
        return instance, cluster_stats(xyz[:0], hag[:0], instance[:0])

    cl = DBSCAN(eps=s["eps"], min_samples=s["min_samples"], n_jobs=-1).fit_predict(xyz[cand])
    scat = feats.get(scattering_key) if scattering_key else None
    stats = cluster_stats(xyz[cand], hag[cand], cl, None if scat is None else scat[cand])
    ok = ((stats.n_points >= s["min_points"])
          & stats.height.between(*s["height"]) & stats.diameter.between(*s["diameter"])
          & (stats.roundness >= s["min_roundness"]) & (stats.slenderness <= s["max_slenderness"]))
    if scat is not None:
        ok &= stats.scattering <= s["max_scattering"]
    stats["accepted"] = ok.values

    keep = stats.loc[ok, "id"].to_numpy()
    remap = {cid: i for i, cid in enumerate(keep)}
    sel = np.isin(cl, keep)
    instance[cand[sel]] = [remap[c] for c in cl[sel]]
    stats["stump_id"] = stats["id"].map(remap).fillna(-1).astype(int)
    return instance, stats


def pseudo_label(xyz: np.ndarray, feats: Dict[str, np.ndarray], cfg: Optional[dict] = None,
                 scale: int = 20) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (labels with IGNORE, stump instance ids, stump candidate table)."""
    c = _merge(DEFAULTS, cfg)
    classes: List[str] = c["classes"]
    hag = feats["hag"]
    labels = np.full(len(xyz), classes.index("other"), dtype=np.int64)
    labels[hag < c["ground_max_hag"] + c["ignore_band"]] = IGNORE
    labels[hag < c["ground_max_hag"]] = classes.index("ground")

    instance = np.full(len(xyz), -1, dtype=np.int64)
    stats = pd.DataFrame()
    if "stump" in classes:
        instance, stats = detect_stumps(xyz, feats, c["stump"], f"scattering_k{scale}")
        labels[instance >= 0] = classes.index("stump")
    return labels, instance, stats


def compare_with_reference(xyz: np.ndarray, labels: np.ndarray, classes: List[str],
                           ref_xyz: np.ndarray, ref_labels: np.ndarray, ref_classes: List[str],
                           class_map: Dict[str, str], tolerance: float = 0.02) -> Dict:
    """Agreement of pseudo-labels with a reference labelling of (a subset of) the same points.

    ``class_map`` maps each pseudo class to a reference class. Points are matched by
    nearest neighbour within ``tolerance`` metres.
    """
    from .metrics import ConfusionMatrix

    dist, nn = cKDTree(ref_xyz).query(xyz, k=1, distance_upper_bound=tolerance)
    matched = np.isfinite(dist) & (labels != IGNORE)
    mapped = np.array([ref_classes.index(class_map[c]) for c in classes])
    cm = ConfusionMatrix(len(ref_classes))
    cm.update(mapped[labels[matched]], ref_labels[nn[matched]])
    report = cm.summary(ref_classes)
    report["matched_fraction"] = float(np.isfinite(dist).mean())
    report["ignored_fraction"] = float((labels == IGNORE).mean())
    return report
