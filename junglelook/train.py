"""Training loop for PointNet / PointNet++ segmentation on prepared scenes."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import save_config
from .data import Scene, TrainBlocks
from .infer import predict_scene
from .metrics import ConfusionMatrix
from .models import build_model, feature_transform_regularizer


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_from_config(cfg: Dict, meta: Dict) -> nn.Module:
    m = cfg["model"]
    return build_model(m["name"], in_channels=3 + len(meta["feature_names"]),
                       num_classes=len(meta["classes"]), **m.get(m["name"], {}))


def load_meta(cfg: Dict) -> Dict:
    with open(os.path.join(cfg["work_dir"], "scenes", "meta.json")) as f:
        return json.load(f)


def load_split(cfg: Dict, meta: Dict, name: str) -> Scene:
    return Scene.load(os.path.join(cfg["work_dir"], "scenes", f"{name}.npz"),
                      np.array(meta["mean"], dtype=np.float32), np.array(meta["std"], dtype=np.float32))


def run_train(cfg: Dict, log=print) -> str:
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    device = pick_device(t["device"])
    meta = load_meta(cfg)
    classes = meta["classes"]

    run_dir = os.path.join(cfg["work_dir"], "runs", f"{cfg['model']['name']}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    train_scene, val_scene = load_split(cfg, meta, "train"), load_split(cfg, meta, "val")
    ds = TrainBlocks(train_scene, t["block_size"], t["num_points"], t["samples_per_epoch"],
                     class_balanced=t["class_balanced"], num_classes=len(classes))
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=False, drop_last=True,
                        num_workers=t["num_workers"], persistent_workers=t["num_workers"] > 0)

    model = model_from_config(cfg, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{cfg['model']['name']}: {n_params / 1e6:.2f}M params on {device}; "
        f"train {len(train_scene):,} pts, val {len(val_scene):,} pts; classes {classes}")
    weights = torch.tensor(meta["class_weights"], device=device)
    log(f"class weights: {dict(zip(classes, np.round(meta['class_weights'], 3)))}")
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = len(loader)
    total, warm = t["epochs"] * steps_per_epoch, t["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, bad_epochs, history = -1.0, 0, []
    metrics_path = os.path.join(run_dir, "metrics.csv")
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, loss_sum, cm = time.time(), 0.0, ConfusionMatrix(len(classes))
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, trans = model(x)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                if trans is not None and t["ft_reg_weight"] > 0:
                    loss = loss + t["ft_reg_weight"] * feature_transform_regularizer(trans)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()
            cm.update(logits.argmax(-1).cpu().numpy(), y.cpu().numpy())
        tr = cm.summary(classes)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / steps_per_epoch,
               "train_miou": tr["miou"], "train_oa": tr["overall_accuracy"], "time_s": 0.0}

        if epoch % t["val_every"] == 0 or epoch == t["epochs"]:
            probs = predict_scene(model, val_scene, t["block_size"], t["num_points"],
                                  cfg["infer"].get("stride"), cfg["infer"]["batch_size"], device, len(classes))
            vcm = ConfusionMatrix(len(classes))
            vcm.update(probs.argmax(1), val_scene.labels)
            val = vcm.summary(classes)
            row.update(val_miou=val["miou"], val_oa=val["overall_accuracy"],
                       **{f"val_iou_{c}": val["per_class"][c]["iou"] for c in classes})
            ckpt = {"model": model.state_dict(), "cfg": cfg, "meta": meta, "epoch": epoch, "val": val}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if val["miou"] > best:
                best, bad_epochs = val["miou"], 0
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                with open(os.path.join(run_dir, "best_val.json"), "w") as f:
                    json.dump({"epoch": epoch, **val}, f, indent=2)
            else:
                bad_epochs += t["val_every"]
        row["time_s"] = round(time.time() - t0, 1)
        history.append(row)
        msg = f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | train mIoU {row['train_miou']:.3f}"
        if "val_miou" in row:
            ious = " ".join(f"{c}={_fmt(row[f'val_iou_{c}'])}" for c in classes)
            msg += f" | val mIoU {row['val_miou']:.3f} ({ious})"
        log(f"{msg} | {row['time_s']}s")
        _write_csv(metrics_path, history)
        if bad_epochs >= t["patience"]:
            log(f"early stop: no val mIoU improvement for {t['patience']} epochs")
            break

    log(f"best val mIoU {best:.4f} -> {os.path.join(run_dir, 'best.pt')}")
    _plot_history(history, os.path.join(run_dir, "curves.png"))
    return run_dir


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _write_csv(path: str, rows) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _plot_history(rows, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.plot(ep, [r["train_loss"] for r in rows])
    a.set_title("train loss")
    a.set_xlabel("epoch")
    b.plot(ep, [r["train_miou"] for r in rows], label="train")
    v = [(r["epoch"], r["val_miou"]) for r in rows if "val_miou" in r]
    if v:
        b.plot(*zip(*v), label="val")
    b.set_title("mIoU")
    b.set_xlabel("epoch")
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
