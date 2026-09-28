"""Segmentation metrics from a running confusion matrix (rows = target, cols = prediction)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.n = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
        m = target != self.ignore_index
        self.mat += np.bincount(self.n * target[m] + pred[m], minlength=self.n ** 2).reshape(self.n, self.n)

    @property
    def iou(self) -> np.ndarray:
        tp = np.diag(self.mat)
        denom = self.mat.sum(0) + self.mat.sum(1) - tp
        return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)

    def summary(self, class_names: Optional[List[str]] = None) -> Dict:
        names = class_names or [str(i) for i in range(self.n)]
        tp = np.diag(self.mat).astype(float)
        support, predicted = self.mat.sum(1), self.mat.sum(0)
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), np.nan)
        recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), np.nan)
        iou = self.iou
        present = support > 0
        return {
            "overall_accuracy": float(tp.sum() / max(self.mat.sum(), 1)),
            "mean_accuracy": float(np.nanmean(recall[present])) if present.any() else float("nan"),
            "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "per_class": {n: {"iou": _f(iou[i]), "precision": _f(precision[i]), "recall": _f(recall[i]),
                              "f1": _f(f1[i]), "support": int(support[i])} for i, n in enumerate(names)},
            "confusion_matrix": self.mat.tolist(),
        }


def _f(x: float) -> Optional[float]:
    return None if np.isnan(x) else round(float(x), 4)


def format_summary(s: Dict) -> str:
    lines = [f"OA {s['overall_accuracy']:.4f} | mAcc {s['mean_accuracy']:.4f} | mIoU {s['miou']:.4f}",
             f"{'class':<14}{'IoU':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'support':>12}"]
    for name, m in s["per_class"].items():
        cells = [f"{m[k]:.4f}" if m[k] is not None else "   -  " for k in ("iou", "precision", "recall", "f1")]
        lines.append(f"{name:<14}" + "".join(f"{c:>8}" for c in cells) + f"{m['support']:>12}")
    return "\n".join(lines)


def plot_confusion(mat: np.ndarray, class_names: List[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.asarray(mat, dtype=float)
    norm = mat / np.maximum(mat.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 2, 1.2 * len(class_names) + 1.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(mat[i, j])})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
