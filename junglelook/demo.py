"""End-to-end demonstration on synthetic forest plots with known ground truth.

    junglelook demo                     # ~5-10 min on a laptop CPU, faster on GPU / Apple MPS
    junglelook demo --epochs 20 --set model.name=pointnet2

1. generates two plots (A for training, B never seen) with ground, stumps, shrubs and trees
2. runs extract -> pseudolabel -> split -> train -> evaluate on plot A (no labels used for training)
3. segments plot B with the trained model and scores it against the true labels
4. writes figures and a summary to ``work/demo/``
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from . import io
from .config import apply_overrides, load_config
from .metrics import ConfusionMatrix, format_summary
from .synthetic import STUMP, make_forest

CONFIG = os.path.join(os.path.dirname(__file__), "resources", "demo.yaml")


def _write_plot(out_dir: str, name: str, seed: int) -> None:
    xyz, labels, centres = make_forest(size=40.0, n_stumps=40, n_shrubs=20, n_trees=6, seed=seed)
    io.write_ply(os.path.join(out_dir, f"{name}.ply"), xyz)                       # what the pipeline sees
    io.write_ply(os.path.join(out_dir, f"{name}_truth.ply"), xyz, {"label": labels})
    stumps_dir = os.path.join(out_dir, f"{name}_stumps")
    os.makedirs(stumps_dir, exist_ok=True)
    stump = xyz[labels == STUMP]
    owner = np.linalg.norm(stump[:, None, :2] - centres[None], axis=-1).argmin(1)
    for i in range(len(centres)):
        io.write_ply(os.path.join(stumps_dir, f"stump_{i:03d}.ply"), stump[owner == i])
    np.savetxt(os.path.join(out_dir, f"{name}_stump_centres.csv"), centres, delimiter=",",
               header="x,y", comments="")


def run_demo(out_dir: str = "work/demo", overrides: Optional[List[str]] = None, log=print) -> dict:
    from .infer import run_evaluate, run_predict
    from .pipeline import match_centres, run_extract, run_pseudolabel, run_split
    from .render import render_file
    from .train import run_train

    os.makedirs(out_dir, exist_ok=True)
    log("1/4 generating synthetic plots A (train) and B (unseen)")
    _write_plot(out_dir, "plot_a", seed=0)
    _write_plot(out_dir, "plot_b", seed=1)

    cfg = load_config(CONFIG)
    cfg["work_dir"] = out_dir
    cfg["data"]["inputs"] = [os.path.join(out_dir, "plot_a.ply")]
    cfg["reference"]["sources"][0]["path"] = os.path.join(out_dir, "plot_a_stumps", "*.ply")
    cfg = apply_overrides(cfg, overrides or [])

    log("2/4 features, pseudo-labels and model training on plot A")
    run_extract(cfg, log)
    run_pseudolabel(cfg, log)
    run_split(cfg, log)
    checkpoint = os.path.join(run_train(cfg, log), "best.pt")
    run_evaluate(cfg, checkpoint, "test", log)

    log("3/4 segmenting unseen plot B")
    seg = os.path.join(out_dir, "plot_b_seg.ply")
    run_predict(checkpoint, [os.path.join(out_dir, "plot_b.ply")], seg, full_resolution=True, log=log)
    pred = io.read_cloud(seg)["pred"].astype(int)
    truth = io.read_cloud(os.path.join(out_dir, "plot_b_truth.ply"))["label"].astype(int)
    classes = cfg["pseudolabel"]["classes"]
    cm = ConfusionMatrix(len(classes))
    cm.update(pred, truth)
    points = cm.summary(classes)
    inst = np.loadtxt(os.path.join(out_dir, "plot_b_stump_centres.csv"), delimiter=",", skiprows=1)
    import pandas as pd

    found = pd.read_csv(os.path.join(out_dir, "plot_b_seg_instances.csv"))
    stumps = match_centres(found[["x", "y"]].to_numpy() if len(found) else np.zeros((0, 2)), inst)
    log("plot B (unseen) vs ground truth:\n" + format_summary(points))
    log(f"plot B stump detection: {stumps}")

    log("4/4 rendering figures")
    render_file(os.path.join(out_dir, "plot_b_truth.ply"), "label", os.path.join(out_dir, "plot_b_truth.png"),
                classes, title="Plot B - ground truth")
    render_file(seg, "pred", os.path.join(out_dir, "plot_b_prediction.png"), classes,
                instances_csv=os.path.join(out_dir, "plot_b_seg_instances.csv"),
                title=f"Plot B - {cfg['model']['name']} prediction (mIoU {points['miou']:.2f}, "
                      f"stump F1 {stumps['f1']:.2f})")
    render_file(os.path.join(out_dir, "pseudolabels.ply"), "label", os.path.join(out_dir, "plot_a_pseudolabels.png"),
                classes, title="Plot A - pseudo-labels from geometric rules (grey = ignored)")

    summary = {"checkpoint": checkpoint, "plot_b_points": points, "plot_b_stumps": stumps}
    with open(os.path.join(out_dir, "demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"done - figures and summary in {out_dir}/")
    return summary
