"""Register annotated objects (e.g. stumps digitised on the RGB orthophoto, in UTM) onto a
LiDAR cloud in a local frame, without any known control points.

1. object centroids (one annotation file per object) vs. stump-like cluster centroids
   found in the target cloud by the geometric rules (loose thresholds)
2. 2D RANSAC over centroid pairs with equal separation -> rotation about z + xy shift
3. vertical offset = median z difference under the annotations
4. point-to-point ICP of all annotation points against the target cloud

    python -m junglelook.register --annotations "dataset/stumps_labelled_3/*.txt" \
        --target dataset/filtered_point_cloud.ply --output configs/transforms/stumps_utm_to_lidar.txt
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import io
from .features import extract_features, voxel_downsample
from .pseudolabel import detect_stumps

LOOSE_STUMP_RULES = {"min_hag": 0.05, "max_hag": 1.5, "max_stack_hag": 1.8, "eps": 0.1, "min_samples": 8,
                     "min_points": 30, "height": [0.1, 1.0], "diameter": [0.25, 2.5], "min_roundness": 0.3,
                     "max_slenderness": 10.0, "max_scattering": 1.0}


def ransac_2d(src: np.ndarray, dst: np.ndarray, inlier_dist: float = 0.6, min_pair_dist: float = 8.0,
              iterations: int = 3000, seed: int = 0) -> Tuple[np.ndarray, int]:
    """Rigid 2D transform (3x3 homogeneous) mapping ``src`` points onto ``dst`` points."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(dst)
    dd = np.linalg.norm(dst[:, None] - dst[None], axis=-1)
    best_T, best_n = np.eye(3), -1
    for _ in range(iterations):
        i, j = rng.choice(len(src), 2, replace=False)
        d = np.linalg.norm(src[i] - src[j])
        if d < min_pair_dist:
            continue
        pa, pb = np.nonzero(np.abs(dd - d) < inlier_dist / 2)
        if len(pa) == 0:
            continue
        pick = rng.choice(len(pa), min(len(pa), 200), replace=False)
        v1 = src[j] - src[i]
        for a, b in zip(pa[pick], pb[pick]):
            v2 = dst[b] - dst[a]
            th = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            t = dst[a] - R @ src[i]
            n = np.isfinite(tree.query(src @ R.T + t, distance_upper_bound=inlier_dist)[0]).sum()
            if n > best_n:
                best_n = n
                best_T = np.eye(3)
                best_T[:2, :2], best_T[:2, 2] = R, t
    return best_T, int(best_n)


def icp(src: np.ndarray, dst: np.ndarray, T: np.ndarray, max_dist: float = 0.3,
        iterations: int = 30) -> Tuple[np.ndarray, float, float]:
    """Point-to-point ICP refining the 4x4 ``T``. Returns (T, inlier fraction, inlier RMS)."""
    tree = cKDTree(dst)
    for _ in range(iterations):
        s = src @ T[:3, :3].T + T[:3, 3]
        d, nn = tree.query(s, distance_upper_bound=max_dist)
        ok = np.isfinite(d)
        a, b = s[ok], dst[nn[ok]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, mb - R @ ma
        T = step @ T
    s = src @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(s, distance_upper_bound=max_dist)
    ok = np.isfinite(d)
    return T, float(ok.mean()), float(np.sqrt((d[ok] ** 2).mean()))


def register(annotation_files: List[str], target_xyz: np.ndarray, log=print) -> np.ndarray:
    objects = [io.read_cloud(f)["xyz"] for f in annotation_files]
    centres = np.array([o.mean(axis=0) for o in objects])
    keep: List[int] = []                      # overlapping annotations -> one centroid
    for i, c in enumerate(centres):
        if all(np.linalg.norm(c[:2] - centres[k, :2]) > 0.5 for k in keep):
            keep.append(i)
    src_xy = centres[keep, :2]
    mu = src_xy.mean(axis=0)

    feats, _ = extract_features(target_xyz, {"scales": [10]})
    _, stats = detect_stumps(target_xyz, feats, LOOSE_STUMP_RULES, "scattering_k10")
    dst_xy = stats.loc[stats.accepted, ["x", "y"]].to_numpy()
    log(f"{len(src_xy)} annotated objects, {len(dst_xy)} candidate objects in target")

    T2, n = ransac_2d(src_xy - mu, dst_xy)
    log(f"RANSAC: {n}/{len(src_xy)} objects matched, rotation {np.degrees(np.arctan2(T2[1, 0], T2[0, 0])):.2f} deg")
    T = np.eye(4)
    T[:2, :2] = T2[:2, :2]
    T[:2, 3] = T2[:2, 2] - T2[:2, :2] @ mu

    pts = np.concatenate(objects)
    pts = pts[voxel_downsample(pts, 0.05)]
    xy_tree = cKDTree(target_xyz[:, :2])
    moved = pts[:, :2] @ T[:2, :2].T + T[:2, 3]
    d, nn = xy_tree.query(moved, distance_upper_bound=0.1)
    ok = np.isfinite(d)
    T[2, 3] = np.median(target_xyz[nn[ok], 2] - pts[ok, 2])
    T, inliers, rms = icp(pts, target_xyz, T)
    log(f"ICP: {inliers:.1%} inliers within 0.3 m, RMS {rms:.3f} m")
    return T


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True, help="one cloud file per object (globs ok)")
    ap.add_argument("--target", required=True, help="cloud to register onto")
    ap.add_argument("--output", required=True, help="4x4 transform (text) mapping annotations -> target")
    ap.add_argument("--voxel", type=float, default=0.03)
    args = ap.parse_args(argv)
    target = io.read_cloud(args.target)["xyz"]
    target = target[voxel_downsample(target, args.voxel)]
    T = register(io.expand_paths(args.annotations), target)
    np.savetxt(args.output, T, fmt="%.10f", header="4x4 rigid transform: annotations -> target frame")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
