"""YAML configuration with defaults and ``--set a.b=value`` overrides."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "name": "experiment",
    "work_dir": None,                     # default: work/<name>
    "data": {
        "inputs": [],                     # files / globs, any format supported by junglelook.io
        "bounds": None,                   # [xmin, ymin, xmax, ymax] crop
        "voxel_size": 0.03,               # m, 0 disables
        "label_field": None,              # use an existing per-point label column instead of pseudo-labels
        "label_map": None,                # {value_in_file: class_name} for label_field
    },
    "features": {
        "scales": [10, 30],               # k-NN sizes for the eigen features
        "column_cell": 0.5,
        "dtm": {"cell": 0.5, "windows": [1.0, 2.0, 4.0], "dh0": 0.15, "slope": 0.3, "smooth": 1.0},
    },
    "pseudolabel": {},                    # see junglelook.pseudolabel.DEFAULTS
    "reference": None,                    # optional labelled data used only for evaluation
    "split": {"tile_size": 10.0, "ratios": [0.7, 0.15, 0.15], "seed": 0},
    "model": {
        "name": "pointnet2",
        "input_features": None,           # None = all extracted features
        "pointnet": {"feature_transform": True, "dropout": 0.3},
        "pointnet2": {"npoints": [1024, 256, 64, 16], "radii": [0.2, 0.4, 0.8, 1.6], "nsample": 32,
                      "dropout": 0.5},
    },
    "train": {
        "block_size": 6.0, "num_points": 4096, "batch_size": 16, "epochs": 40,
        "samples_per_epoch": 2000, "lr": 1e-3, "weight_decay": 1e-4, "warmup_epochs": 2,
        "class_weights": "log", "class_balanced": True, "ft_reg_weight": 1e-3,
        "val_every": 1, "patience": 10, "num_workers": 0, "device": "auto", "seed": 0,
        "grad_clip": 5.0,
    },
    "infer": {
        "stride": None,                   # None = block_size / 2 (every point voted ~4x)
        "batch_size": 16,
        "smooth_k": 8,                    # k-NN majority smoothing of predictions, 0 = off
        "instances": {"class": "stump", "eps": 0.10, "min_points": 30},
    },
}


def deep_merge(base: Dict, override: Optional[Dict]) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: Dict, overrides: List[str]) -> Dict:
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(path: Optional[str], overrides: Optional[List[str]] = None) -> Dict:
    user = {}
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    cfg = apply_overrides(deep_merge(DEFAULTS, user), overrides or [])
    cfg["work_dir"] = cfg["work_dir"] or os.path.join("work", cfg["name"])
    return cfg


def save_config(cfg: Dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
