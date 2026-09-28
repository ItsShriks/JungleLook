"""Quick-look images of a labelled cloud (top view + side view), no CloudCompare needed.

    junglelook render --input work/demo/plot_b_seg.ply --field pred --output plot_b.png
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import io

CLASS_COLOURS = {"ground": "#b89f7a", "stump": "#d62728", "other": "#2ca02c", "ignore": "#bbbbbb"}
DEFAULT_CLASSES = ["ground", "stump", "other"]


def render(cloud: Dict[str, np.ndarray], field: str, path: str, classes: Optional[List[str]] = None,
           title: Optional[str] = None, max_points: int = 400_000,
           instances: Optional[np.ndarray] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    xyz, values = cloud["xyz"], np.asarray(cloud[field])
    if len(xyz) > max_points:
        keep = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, values = xyz[keep], values[keep]
    xyz = xyz - xyz.min(axis=0)
    categorical = np.issubdtype(values.dtype, np.integer) or np.allclose(values, np.round(values))
    classes = classes or DEFAULT_CLASSES

    fig, (top, side) = plt.subplots(2, 1, figsize=(9, 9.5), gridspec_kw={"height_ratios": [3, 1]})
    order = np.argsort(xyz[:, 2])                 # draw high points last in the top view
    if categorical:
        values = values.astype(int)
        colours = [CLASS_COLOURS.get(c, f"C{i}") for i, c in enumerate(classes)]
        cmap = ListedColormap(colours + [CLASS_COLOURS["ignore"]])
        codes = np.where((values >= 0) & (values < len(classes)), values, len(classes))
        kw = dict(c=codes, cmap=cmap, vmin=-0.5, vmax=len(classes) + 0.5)
        handles = [Patch(color=colours[i], label=c) for i, c in enumerate(classes) if (codes == i).any()]
    else:
        kw = dict(c=values, cmap="viridis")
        handles = []
    top.scatter(xyz[order, 0], xyz[order, 1], s=0.3, linewidths=0, **{**kw, "c": kw["c"][order]})
    top.set_aspect("equal")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    if instances is not None and len(instances):
        top.scatter(instances[:, 0], instances[:, 1], s=60, facecolors="none", edgecolors="black",
                    linewidths=0.8, label="detected stump")
        handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec="black", label="detected stump"))
    if handles:
        top.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    top.set_title(title or field)

    band = np.abs(xyz[:, 1] - np.median(xyz[:, 1])) < 2.0        # 4 m wide transect
    side.scatter(xyz[band, 0], xyz[band, 2], s=0.5, linewidths=0, **{**kw, "c": kw["c"][band]})
    side.set_aspect("equal")
    side.set_xlabel("x (m)")
    side.set_ylabel("z (m)")
    side.set_title("side view of a 4 m transect", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_file(input_path: str, field: str, output: str, classes: Optional[List[str]] = None,
                instances_csv: Optional[str] = None, title: Optional[str] = None) -> None:
    cloud = io.read_cloud(input_path)
    if field not in cloud:
        raise KeyError(f"field {field!r} not in {input_path}; available: {sorted(k for k in cloud if k != 'xyz')}")
    inst = None
    if instances_csv:
        import pandas as pd

        df = pd.read_csv(instances_csv)
        if len(df):
            inst = df[["x", "y"]].to_numpy() - cloud["xyz"][:, :2].min(axis=0)
    render(cloud, field, output, classes, title, instances=inst)
