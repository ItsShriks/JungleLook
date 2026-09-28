"""Pipeline stages that turn raw, unlabelled clouds into training scenes.

    extract      raw cloud(s) -> voxelised xyz + geometric features       work/<name>/features.npz
    pseudolabel  features -> rule-based labels + stump instances           work/<name>/labels.npz
    split        spatial train / val / test scenes + normalisation stats   work/<name>/scenes/*.npz

Every stage also writes a PLY you can open in CloudCompare to inspect the result.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import io
from .data import class_weights, feature_stats, save_scene, spatial_split
from .features import extract_features, voxel_downsample
from .metrics import format_summary
from .pseudolabel import DEFAULTS as PL_DEFAULTS
from .pseudolabel import IGNORE, _merge, pseudo_label


def _p(cfg: Dict, *parts: str) -> str:
    return os.path.join(cfg["work_dir"], *parts)


def classes_of(cfg: Dict) -> List[str]:
    return _merge(PL_DEFAULTS, cfg["pseudolabel"])["classes"]


# --------------------------------------------------------------------------- extract
def load_and_featurize(inputs, cfg: Dict, log=print) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Read + crop + voxelise + compute features. Returns (cloud columns, features)."""
    d = cfg["data"]
    cloud = io.read_clouds(inputs, d.get("bounds"))
    log(f"read {len(cloud['xyz']):,} points from {len(io.expand_paths(inputs))} file(s)")
    keep = voxel_downsample(cloud["xyz"], d.get("voxel_size") or 0)
    cloud = io.subset(cloud, keep)
    log(f"voxel {d.get('voxel_size')} m -> {len(keep):,} points")
    feats, _ = extract_features(cloud["xyz"], cfg["features"])
    log(f"computed {len(feats)} features")
    return cloud, feats


def run_extract(cfg: Dict, log=print) -> None:
    os.makedirs(cfg["work_dir"], exist_ok=True)
    cloud, feats = load_and_featurize(cfg["data"]["inputs"], cfg, log)
    names = sorted(feats)
    arrays = {"xyz": cloud["xyz"], "feature_names": np.array(names),
              "feats": np.stack([feats[n] for n in names], axis=1)}
    label_field = cfg["data"].get("label_field")
    if label_field:
        if label_field not in cloud:
            raise KeyError(f"label_field {label_field!r} not in input; columns: {sorted(cloud)}")
        arrays["input_labels"] = cloud[label_field]
    if "rgb" in cloud:
        arrays["rgb"] = cloud["rgb"]
    if cfg.get("reference"):
        ref, ref_inst = paint_reference(cloud["xyz"], feats, cfg["reference"], log)
        arrays["ref_labels"], arrays["ref_instance"] = ref, ref_inst
    np.savez_compressed(_p(cfg, "features.npz"), **arrays)
    io.write_ply(_p(cfg, "features.ply"), cloud["xyz"],
                 {k: feats[k] for k in ("hag", "column_max_hag", "column_stack_hag", *[n for n in names if n.endswith(
                     f"_k{cfg['features']['scales'][0]}")])})
    log(f"wrote {_p(cfg, 'features.npz')}")


def load_features(cfg: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    z = np.load(_p(cfg, "features.npz"))
    names = [str(n) for n in z["feature_names"]]
    feats = {n: z["feats"][:, i] for i, n in enumerate(names)}
    extra = {k: z[k] for k in z.files if k not in ("xyz", "feats", "feature_names")}
    return z["xyz"], feats, extra


# --------------------------------------------------------------------------- reference
def paint_reference(xyz: np.ndarray, feats: Dict[str, np.ndarray], ref: Dict,
                    log=print) -> Tuple[np.ndarray, np.ndarray]:
    """Build reference labels for evaluation by painting annotated clouds onto the scene.

    ``ref.sources``: list of ``{path, class, transform?, radius?, min_hag?, instances?}``.
    Each file's points (optionally moved by a 4x4 ``transform``) label every scene point
    within ``radius``; with ``instances: true`` each file is one object instance.
    """
    classes = ref["classes"]
    labels = np.full(len(xyz), classes.index(ref["default_class"]) if ref.get("default_class") else IGNORE)
    instance = np.full(len(xyz), -1, dtype=np.int64)
    tree = cKDTree(xyz)
    n_inst = 0
    for src in ref["sources"]:
        T = np.loadtxt(src["transform"]) if src.get("transform") else None
        cls = classes.index(src["class"])
        for path in io.expand_paths(src["path"]):
            p = io.read_cloud(path)["xyz"]
            if T is not None:
                p = p @ T[:3, :3].T + T[:3, 3]
            radius = src.get("radius", 0.02)
            p = p[voxel_downsample(p, radius / 2)]        # annotations are often denser than the scene
            hits = np.unique(np.concatenate(
                [np.asarray(h, dtype=np.int64) for h in tree.query_ball_point(p, radius)] or [[]]
            )).astype(np.int64)
            if src.get("min_hag") is not None:
                hits = hits[feats["hag"][hits] >= src["min_hag"]]
            labels[hits] = cls
            if src.get("instances"):
                instance[hits] = n_inst
                n_inst += 1
    counts = {c: int((labels == i).sum()) for i, c in enumerate(classes)}
    log(f"reference labels: {counts}" + (f", {n_inst} instances" if n_inst else ""))
    return labels, instance


def match_centres(pred_xy: np.ndarray, ref_xy: np.ndarray, max_dist: float = 0.75) -> Dict:
    """Object-level precision / recall with one-to-one greedy matching of XY centres."""
    pred_xy, ref_xy = np.asarray(pred_xy).reshape(-1, 2), np.asarray(ref_xy).reshape(-1, 2)
    if len(ref_xy) == 0 or len(pred_xy) == 0:
        return {"reference": len(ref_xy), "predicted": len(pred_xy), "matched": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0}
    d = np.linalg.norm(ref_xy[:, None] - pred_xy[None], axis=-1)
    matched, used_r, used_p = 0, set(), set()
    for flat in np.argsort(d, axis=None):
        r, p = np.unravel_index(flat, d.shape)
        if d[r, p] > max_dist:
            break
        if r in used_r or p in used_p:
            continue
        used_r.add(r)
        used_p.add(p)
        matched += 1
    prec, rec = matched / len(pred_xy), matched / len(ref_xy)
    return {"reference": len(ref_xy), "predicted": len(pred_xy), "matched": matched,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4)}


def instance_detection(pred_xy: np.ndarray, ref_xyz: np.ndarray, ref_instance: np.ndarray,
                       max_dist: float = 0.75) -> Dict:
    """A reference instance is found when a predicted instance centre lies within ``max_dist`` m."""
    ids = np.unique(ref_instance[ref_instance >= 0])
    ref_xy = np.array([ref_xyz[ref_instance == i, :2].mean(axis=0) for i in ids]).reshape(-1, 2)
    return match_centres(pred_xy, ref_xy, max_dist)


def reference_report(xyz, labels, instance, classes, extra, ref_cfg) -> Optional[Dict]:
    """Point-level (mapped through ``class_map``) and instance-level agreement with the reference."""
    if "ref_labels" not in extra or not ref_cfg:
        return None
    from .metrics import ConfusionMatrix

    ref_classes = ref_cfg["classes"]
    mapped = np.array([ref_classes.index(ref_cfg["class_map"][c]) for c in classes])
    valid = labels != IGNORE
    cm = ConfusionMatrix(len(ref_classes))
    cm.update(mapped[labels[valid]], extra["ref_labels"][valid])
    report = {"points": cm.summary(ref_classes), "ignored_fraction": float((~valid).mean())}
    if "ref_instance" in extra and (extra["ref_instance"] >= 0).any() and (instance >= 0).any():
        centres = np.array([xyz[instance == i, :2].mean(axis=0) for i in np.unique(instance[instance >= 0])])
        report["instances"] = instance_detection(centres, xyz, extra["ref_instance"])
    return report


# --------------------------------------------------------------------------- pseudolabel
def run_pseudolabel(cfg: Dict, log=print) -> None:
    xyz, feats, extra = load_features(cfg)
    classes = classes_of(cfg)
    if "input_labels" in extra:
        lmap = cfg["data"].get("label_map") or {}
        raw = extra["input_labels"]
        labels = np.full(len(raw), IGNORE, dtype=np.int64)
        for value, name in lmap.items():
            labels[raw == value] = classes.index(name)
        instance = np.full(len(raw), -1, dtype=np.int64)
        log("using labels from the input file (label_field)")
    else:
        labels, instance, stats = pseudo_label(xyz, feats, cfg["pseudolabel"], cfg["features"]["scales"][0])
        stats.to_csv(_p(cfg, "stump_candidates.csv"), index=False)
        log(f"stump candidates: {len(stats)}, accepted: {int(stats['accepted'].sum()) if len(stats) else 0}")
        if _merge(PL_DEFAULTS, cfg["pseudolabel"])["stumps_from_reference"]:
            ref_classes = cfg["reference"]["classes"]
            ref_stump = extra["ref_labels"] == ref_classes.index("stump")
            labels[labels == classes.index("stump")] = classes.index("other")
            labels[ref_stump] = classes.index("stump")
            instance = extra.get("ref_instance", np.full(len(labels), -1)).copy()
            log(f"stumps taken from the reference: {int(ref_stump.sum()):,} points")
    counts = {c: int((labels == i).sum()) for i, c in enumerate(classes)}
    counts["ignore"] = int((labels == IGNORE).sum())
    log(f"labels: {counts}")
    np.savez_compressed(_p(cfg, "labels.npz"), labels=labels, instance=instance)
    io.write_ply(_p(cfg, "pseudolabels.ply"), xyz, {"label": labels, "instance": instance, "hag": feats["hag"]})

    report = {"counts": counts}
    ref = reference_report(xyz, labels, instance, classes, extra, cfg.get("reference"))
    if ref:
        report["vs_reference"] = ref
        log("pseudo-labels vs reference:\n" + format_summary(ref["points"]))
        if "instances" in ref:
            log(f"stump instances vs reference: {ref['instances']}")
    with open(_p(cfg, "pseudolabel_report.json"), "w") as f:
        json.dump(report, f, indent=2)


# --------------------------------------------------------------------------- split
def run_split(cfg: Dict, log=print) -> None:
    xyz, feats, extra = load_features(cfg)
    lab = np.load(_p(cfg, "labels.npz"))
    labels, instance = lab["labels"], lab["instance"]
    classes = classes_of(cfg)
    names = cfg["model"].get("input_features") or sorted(feats)
    fmat = np.stack([feats[n] for n in names], axis=1).astype(np.float32)

    s = cfg["split"]
    split = spatial_split(xyz, s["tile_size"], s["ratios"], s["seed"])
    os.makedirs(_p(cfg, "scenes"), exist_ok=True)
    meta = {"classes": classes, "feature_names": names, "splits": {}}
    for i, name in enumerate(("train", "val", "test")):
        m = split == i
        more = {"instance": instance[m]}
        for k in ("ref_labels", "ref_instance"):
            if k in extra:
                more[k] = extra[k][m]
        save_scene(_p(cfg, "scenes", f"{name}.npz"), xyz[m], fmat[m], labels[m], more)
        meta["splits"][name] = {"points": int(m.sum()),
                                **{c: int((labels[m] == j).sum()) for j, c in enumerate(classes)}}
        if name == "train":
            mean, std = feature_stats(fmat[m])
            meta["mean"], meta["std"] = mean.tolist(), std.tolist()
            meta["class_weights"] = class_weights(labels[m], len(classes), cfg["train"]["class_weights"]).tolist()
    for name, st in meta["splits"].items():
        missing = [c for c in classes if st[c] == 0]
        log(f"{name:>5}: {st['points']:>9,} pts  " + "  ".join(f"{c}={st[c]:,}" for c in classes)
            + (f"   WARNING: no {missing}" if missing else ""))
    with open(_p(cfg, "scenes", "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    io.write_ply(_p(cfg, "split.ply"), xyz, {"split": split, "label": labels})
