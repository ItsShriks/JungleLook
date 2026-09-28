"""Whole-scene inference, post-processing, evaluation and prediction on new clouds."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader

from . import io
from .data import EvalBlocks, Scene
from .metrics import ConfusionMatrix, format_summary, plot_confusion
from .pseudolabel import IGNORE, cluster_stats


@torch.no_grad()
def predict_scene(model, scene: Scene, block_size: float, num_points: int, stride: Optional[float],
                  batch_size: int, device, num_classes: int) -> np.ndarray:
    """Average softmax over all sliding-window chunks that contain each point -> (N, C)."""
    model.eval()
    ds = EvalBlocks(scene, block_size, num_points, stride or block_size / 2)
    probs = np.zeros((len(scene), num_classes), dtype=np.float32)
    votes = np.zeros(len(scene), dtype=np.float32)
    for x, idx in DataLoader(ds, batch_size=batch_size, shuffle=False):
        logits, _ = model(x.to(device))
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy().reshape(-1, num_classes)
        idx = idx.numpy().ravel()
        np.add.at(probs, idx, p)
        np.add.at(votes, idx, 1.0)
    return probs / np.maximum(votes, 1)[:, None]


def knn_smooth(xyz: np.ndarray, probs: np.ndarray, k: int) -> np.ndarray:
    """Average class probabilities over the k nearest neighbours (removes salt-and-pepper noise)."""
    if k <= 1:
        return probs
    _, nn = cKDTree(xyz).query(xyz, k=k, workers=-1)
    return probs[nn].mean(axis=1)


def extract_instances(xyz: np.ndarray, hag: np.ndarray, pred: np.ndarray, cls: int,
                      eps: float, min_points: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """DBSCAN the points predicted as ``cls`` into objects and measure each one."""
    instance = np.full(len(xyz), -1, dtype=np.int64)
    sel = np.flatnonzero(pred == cls)
    if len(sel) < min_points:
        return instance, cluster_stats(xyz[:0], hag[:0], instance[:0])
    cl = DBSCAN(eps=eps, min_samples=min(10, min_points), n_jobs=-1).fit_predict(xyz[sel])
    sizes = np.bincount(cl[cl >= 0])
    good = np.flatnonzero(sizes >= min_points)
    remap = np.full(len(sizes), -1)
    remap[good] = np.arange(len(good))
    instance[sel[cl >= 0]] = remap[cl[cl >= 0]]
    stats = cluster_stats(xyz, hag, instance)
    return instance, stats


def load_checkpoint(path: str, device):
    from .train import model_from_config

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = model_from_config(ckpt["cfg"], ckpt["meta"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def _postprocess(cfg, meta, xyz, probs, hag):
    classes = meta["classes"]
    probs = knn_smooth(xyz, probs, cfg["infer"].get("smooth_k", 0))
    pred = probs.argmax(1)
    inst_cfg = cfg["infer"].get("instances") or {}
    instance, stats = np.full(len(xyz), -1), pd.DataFrame()
    if inst_cfg.get("class") in classes:
        instance, stats = extract_instances(xyz, hag, pred, classes.index(inst_cfg["class"]),
                                            inst_cfg["eps"], inst_cfg["min_points"])
    return probs, pred, instance, stats


# --------------------------------------------------------------------------- evaluate
def run_evaluate(cfg: Dict, checkpoint: str, split: str = "test", log=print) -> Dict:
    from .pipeline import instance_detection
    from .train import load_split, pick_device

    device = pick_device(cfg["train"]["device"])
    model, ckpt = load_checkpoint(checkpoint, device)
    meta, tcfg = ckpt["meta"], ckpt["cfg"]
    classes = meta["classes"]
    scene = load_split(tcfg, meta, split)
    raw = np.load(os.path.join(tcfg["work_dir"], "scenes", f"{split}.npz"))
    hag = raw["feats"][:, meta["feature_names"].index("hag")] if "hag" in meta["feature_names"] else scene.xyz[:, 2]

    probs = predict_scene(model, scene, tcfg["train"]["block_size"], tcfg["train"]["num_points"],
                          cfg["infer"].get("stride"), cfg["infer"]["batch_size"], device, len(classes))
    probs, pred, instance, stats = _postprocess(cfg, meta, scene.xyz, probs, hag)

    out_dir = os.path.join(os.path.dirname(checkpoint), f"eval_{split}")
    os.makedirs(out_dir, exist_ok=True)
    cm = ConfusionMatrix(len(classes))
    cm.update(pred, scene.labels)
    report = {"checkpoint": checkpoint, "split": split, "vs_pseudolabels": cm.summary(classes)}
    log(f"[{split}] model vs pseudo-labels:\n" + format_summary(report["vs_pseudolabels"]))
    plot_confusion(cm.mat, classes, os.path.join(out_dir, "confusion_vs_pseudolabels.png"))

    ref_cfg = tcfg.get("reference")
    if ref_cfg and "ref_labels" in raw.files:
        ref_classes = ref_cfg["classes"]
        mapped = np.array([ref_classes.index(ref_cfg["class_map"][c]) for c in classes])
        rcm = ConfusionMatrix(len(ref_classes))
        rcm.update(mapped[pred], raw["ref_labels"])
        report["vs_reference"] = rcm.summary(ref_classes)
        log(f"[{split}] model vs reference annotations:\n" + format_summary(report["vs_reference"]))
        plot_confusion(rcm.mat, ref_classes, os.path.join(out_dir, "confusion_vs_reference.png"))
        # the rule-based labels on the same points, for a direct comparison
        valid = scene.labels != IGNORE
        pcm = ConfusionMatrix(len(ref_classes))
        pcm.update(mapped[scene.labels[valid]], raw["ref_labels"][valid])
        report["pseudolabels_vs_reference"] = pcm.summary(ref_classes)
        log(f"[{split}] (for comparison) pseudo-labels vs reference:\n"
            + format_summary(report["pseudolabels_vs_reference"]))
        if "ref_instance" in raw.files and len(stats):
            report["instances_vs_reference"] = instance_detection(
                stats[["x", "y"]].to_numpy(), scene.xyz, raw["ref_instance"])
            log(f"[{split}] instances vs reference: {report['instances_vs_reference']}")

    fields = {"pred": pred, "label": scene.labels, "instance": instance,
              **{f"prob_{c}": probs[:, i] for i, c in enumerate(classes)}}
    if "ref_labels" in raw.files:
        fields["reference"] = raw["ref_labels"]
    io.write_ply(os.path.join(out_dir, f"{split}_pred.ply"), scene.xyz, fields)
    stats.to_csv(os.path.join(out_dir, "instances.csv"), index=False)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {out_dir}")
    return report


# --------------------------------------------------------------------------- predict
def run_predict(checkpoint: str, inputs, out_path: str, overrides: Optional[Dict] = None,
                full_resolution: bool = False, log=print) -> None:
    """Segment a new (unlabelled) cloud with a trained model."""
    from .config import deep_merge
    from .pipeline import load_and_featurize
    from .train import pick_device

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = deep_merge(ckpt["cfg"], overrides or {})
    meta = ckpt["meta"]
    device = pick_device(cfg["train"]["device"])
    model, _ = load_checkpoint(checkpoint, device)

    cloud, feats = load_and_featurize(inputs, cfg, log)
    fmat = np.stack([feats[n] for n in meta["feature_names"]], axis=1).astype(np.float32)
    fmat = (fmat - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    scene = Scene(cloud["xyz"], fmat, None)
    probs = predict_scene(model, scene, cfg["train"]["block_size"], cfg["train"]["num_points"],
                          cfg["infer"].get("stride"), cfg["infer"]["batch_size"], device, len(meta["classes"]))
    probs, pred, instance, stats = _postprocess(cfg, meta, scene.xyz, probs, feats["hag"])
    counts = {c: int((pred == i).sum()) for i, c in enumerate(meta["classes"])}
    log(f"predicted: {counts}; instances: {len(stats)}")

    xyz, fields = scene.xyz, {"pred": pred, "instance": instance, "hag": feats["hag"],
                              **{f"prob_{c}": probs[:, i] for i, c in enumerate(meta["classes"])}}
    if "rgb" in cloud:
        fields["rgb"] = cloud["rgb"]
    if full_resolution:   # carry labels back to every original point
        full = io.read_clouds(inputs, cfg["data"].get("bounds"))
        _, nn = cKDTree(xyz).query(full["xyz"], k=1, workers=-1)
        xyz, fields = full["xyz"], {k: v[nn] for k, v in fields.items() if k != "rgb"}
        if "rgb" in full:
            fields["rgb"] = full["rgb"]
    io.write_cloud(out_path, xyz, fields, label_key="pred")
    stats_path = os.path.splitext(out_path)[0] + "_instances.csv"
    stats.to_csv(stats_path, index=False)
    log(f"wrote {out_path} and {stats_path}")
