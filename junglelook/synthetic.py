"""Synthetic forest plot with known labels: rolling ground, stumps, shrubs and trees."""
from __future__ import annotations

import numpy as np

GROUND, STUMP, OTHER = 0, 1, 2


def make_forest(size: float = 30.0, n_stumps: int = 25, n_shrubs: int = 12, n_trees: int = 4,
                density: float = 400.0, seed: int = 0):
    """Returns (xyz, labels, stump_centres). ``density`` is ground points per m^2."""
    rng = np.random.default_rng(seed)

    def terrain(x, y):
        return 0.04 * x + 0.3 * np.sin(x / 6.0) * np.cos(y / 7.0)

    n = int(size * size * density)
    gx, gy = rng.uniform(0, size, n), rng.uniform(0, size, n)
    parts = [(np.c_[gx, gy, terrain(gx, gy) + rng.normal(0, 0.015, n)], GROUND)]

    centres = []
    while len(centres) < n_stumps:
        c = rng.uniform(2, size - 2, 2)
        if all(np.linalg.norm(c - o) > 2.0 for o in centres):
            centres.append(c)
    for c in centres:
        r, h = rng.uniform(0.15, 0.35), rng.uniform(0.25, 0.6)
        m = int(2 * np.pi * r * h * 2500)
        th, z = rng.uniform(0, 2 * np.pi, m), rng.uniform(0, h, m)
        side = np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th), z]
        k = int(np.pi * r * r * 2500)
        rr, tt = r * np.sqrt(rng.uniform(0, 1, k)), rng.uniform(0, 2 * np.pi, k)
        top = np.c_[c[0] + rr * np.cos(tt), c[1] + rr * np.sin(tt), np.full(k, h)]
        pts = np.vstack([side, top])
        pts[:, 2] += terrain(c[0], c[1]) + rng.normal(0, 0.01, len(pts))
        parts.append((pts, STUMP))

    for _ in range(n_shrubs):   # diffuse blobs 0.5–1.5 m tall
        c = rng.uniform(1, size - 1, 2)
        m = 3000
        p = rng.normal(0, 1, (m, 3)) * [0.6, 0.6, 0.35] + [c[0], c[1], 0.7]
        p[:, 2] = np.abs(p[:, 2]) + terrain(p[:, 0], p[:, 1]) + 0.15
        parts.append((p, OTHER))

    for _ in range(n_trees):    # trunk + crown, much taller than a stump
        c = rng.uniform(2, size - 2, 2)
        m = 6000
        th, z = rng.uniform(0, 2 * np.pi, m), rng.uniform(0, 8, m)
        trunk = np.c_[c[0] + 0.2 * np.cos(th), c[1] + 0.2 * np.sin(th), z + terrain(c[0], c[1])]
        crown = rng.normal(0, 1, (m, 3)) * [1.5, 1.5, 1.2] + [c[0], c[1], 7 + terrain(c[0], c[1])]
        parts.append((np.vstack([trunk, crown]), OTHER))

    xyz = np.vstack([p for p, _ in parts])
    labels = np.concatenate([np.full(len(p), l) for p, l in parts])
    return xyz, labels, np.array(centres)
