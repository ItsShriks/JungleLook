import os

import numpy as np
import pytest
import torch

from junglelook import io
from junglelook.config import load_config
from junglelook.data import EvalBlocks, Scene, TrainBlocks, spatial_split
from junglelook.features import DTM, eigen_features, extract_features, voxel_downsample
from junglelook.metrics import ConfusionMatrix
from junglelook.models import PointNet2Seg, PointNetSeg
from junglelook.pseudolabel import IGNORE, pseudo_label
from junglelook.synthetic import GROUND, OTHER, STUMP, make_forest


@pytest.fixture(scope="module")
def forest():
    return make_forest()


# --------------------------------------------------------------------------- io
def test_ply_roundtrip(tmp_path):
    xyz = np.random.rand(100, 3) * 1000 + [445000, 5696000, 470]
    lab = np.arange(100)
    io.write_ply(str(tmp_path / "a.ply"), xyz, {"label": lab, "hag": np.ones(100)})
    c = io.read_cloud(str(tmp_path / "a.ply"))
    np.testing.assert_allclose(c["xyz"], xyz)          # doubles keep UTM precision
    np.testing.assert_array_equal(c["label"], lab)


def test_cloudcompare_ascii(tmp_path):
    p = tmp_path / "cc.txt"
    p.write_text("//X;Y;Z;R;G;B;Stump1\n1;2;3;255;0;0;1\n4;5;6;0;255;0;1\n")
    c = io.read_cloud(str(p))
    assert c["xyz"].shape == (2, 3) and c["rgb"].max() == 1.0 and "stump1" in c


# --------------------------------------------------------------------------- features
def test_voxel_downsample_one_point_per_voxel():
    xyz = np.random.rand(10000, 3)
    idx = voxel_downsample(xyz, 0.25)
    assert len(idx) <= 64 and len(np.unique(np.floor(xyz[idx] / 0.25), axis=0)) == len(idx)


def test_dtm_recovers_ground_under_objects(forest):
    xyz, labels, _ = forest
    hag = DTM.fit(xyz).height(xyz)
    assert np.median(np.abs(hag[labels == GROUND])) < 0.05
    assert np.median(hag[labels == STUMP]) > 0.1        # stumps are not absorbed into the terrain


def test_eigen_features_plane_vs_line():
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(0, 1, 2000), rng.uniform(0, 1, 2000), np.zeros(2000)]
    line = np.c_[rng.uniform(0, 1, 2000), np.zeros(2000), np.zeros(2000)] + rng.normal(0, 1e-3, (2000, 3))
    fp, fl = eigen_features(plane, k=20), eigen_features(line, k=20)
    assert fp["scattering"].mean() < 0.01 and fp["normal_z"].mean() > 0.99
    assert fp["planarity"].mean() > fl["planarity"].mean()
    assert fl["linearity"].mean() > 0.8


# --------------------------------------------------------------------------- pseudo-labels
def test_pseudolabels_find_stumps(forest):
    xyz, labels, centres = forest
    feats, _ = extract_features(xyz, {"scales": [20]})
    pl, instance, stats = pseudo_label(xyz, feats, scale=20)
    valid = pl != IGNORE
    assert (pl[valid & (labels == GROUND)] == GROUND).mean() > 0.95
    found = stats[stats.accepted][["x", "y"]].to_numpy()
    d = np.linalg.norm(centres[:, None] - found[None], axis=-1).min(axis=1)
    assert (d < 0.5).mean() > 0.8, f"stump recall {(d < 0.5).mean():.2f}"
    assert (pl[labels == OTHER] == STUMP).mean() < 0.05   # trees and shrubs are rejected


# --------------------------------------------------------------------------- data
def test_spatial_split_is_by_tile():
    xyz = np.random.rand(20000, 3) * [100, 100, 1]
    s = spatial_split(xyz, 10.0, [0.7, 0.15, 0.15], seed=0)
    tiles = np.floor((xyz[:, :2] - xyz[:, :2].min(axis=0)) / 10).astype(int)
    for t in np.unique(tiles, axis=0):
        assert len(np.unique(s[(tiles == t).all(1)])) == 1
    frac = np.bincount(s, minlength=3) / len(s)
    assert abs(frac[0] - 0.7) < 0.15 and frac[1] > 0 and frac[2] > 0


def test_eval_blocks_cover_every_point():
    xyz = np.random.rand(5000, 3) * [20, 20, 2]
    scene = Scene(xyz, np.zeros((5000, 2), np.float32), np.zeros(5000, int))
    ds = EvalBlocks(scene, block_size=6.0, num_points=512, stride=3.0)
    seen = np.zeros(5000, bool)
    for i in range(len(ds)):
        x, idx = ds[i]
        assert x.shape == (512, 5)
        seen[idx.numpy()] = True
    assert seen.all()


def test_train_blocks_shapes():
    xyz = np.random.rand(5000, 3) * [20, 20, 2]
    lab = (xyz[:, 2] > 1).astype(int)
    ds = TrainBlocks(Scene(xyz, np.zeros((5000, 4), np.float32), lab), 5.0, 256, 8, num_classes=2)
    x, y = ds[0]
    assert x.shape == (256, 7) and y.shape == (256,)


# --------------------------------------------------------------------------- models
@pytest.mark.parametrize("cls,kw", [(PointNetSeg, {}),
                                    (PointNet2Seg, {"npoints": (256, 64, 16, 8)})])
def test_models_forward_backward(cls, kw):
    model = cls(in_channels=3 + 5, num_classes=3, **kw)
    x = torch.randn(2, 1024, 8)
    logits, _ = model(x)
    assert logits.shape == (2, 1024, 3)
    logits.sum().backward()


def test_models_learn_toy_task():
    """Both nets must fit 'is z above 0' – catches broken gradients / wiring."""
    torch.manual_seed(0)
    for model in (PointNetSeg(3, 2), PointNet2Seg(3, 2, npoints=(128, 32, 8, 4), radii=(0.3, 0.6, 1.2, 2.4))):
        opt = torch.optim.Adam(model.parameters(), 1e-2)
        for _ in range(60):
            x = torch.randn(8, 512, 3)
            y = (x[..., 2] > 0).long()
            loss = torch.nn.functional.cross_entropy(model(x)[0].reshape(-1, 2), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        x = torch.randn(4, 512, 3)
        acc = (model(x)[0].argmax(-1) == (x[..., 2] > 0)).float().mean()
        assert acc > 0.9, f"{type(model).__name__} acc {acc:.2f}"


def test_confusion_metrics():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 2, -1]))
    s = cm.summary()
    assert s["per_class"]["0"]["iou"] == 0.5 and s["per_class"]["2"]["iou"] == 1.0
    assert abs(s["overall_accuracy"] - 0.75) < 1e-9


# --------------------------------------------------------------------------- end to end
def test_pipeline_end_to_end(tmp_path, forest):
    from junglelook import pipeline
    from junglelook.infer import run_evaluate, run_predict
    from junglelook.train import run_train

    xyz, labels, _ = forest
    io.write_ply(str(tmp_path / "plot.ply"), xyz)
    io.write_ply(str(tmp_path / "ref_stumps.ply"), xyz[labels == STUMP])
    cfg = load_config(None, [
        f"work_dir={tmp_path / 'work'}", f"data.inputs=[{tmp_path / 'plot.ply'}]", "data.voxel_size=0.05",
        "features.scales=[16]", "split.tile_size=10", "model.name=pointnet",
        "train.epochs=2", "train.samples_per_epoch=32", "train.batch_size=8", "train.num_points=1024",
        "train.block_size=5", "train.device=cpu", "infer.batch_size=8",
    ])
    cfg["reference"] = {"classes": ["not_stump", "stump"], "default_class": "not_stump",
                        "class_map": {"ground": "not_stump", "other": "not_stump", "stump": "stump"},
                        "sources": [{"path": str(tmp_path / "ref_stumps.ply"), "class": "stump", "radius": 0.03}]}
    def log(*a):
        pass

    pipeline.run_extract(cfg, log)
    pipeline.run_pseudolabel(cfg, log)
    pipeline.run_split(cfg, log)
    run_dir = run_train(cfg, log)
    report = run_evaluate(cfg, os.path.join(run_dir, "best.pt"), "test", log)
    assert "vs_reference" in report and 0 <= report["vs_pseudolabels"]["miou"] <= 1
    out = tmp_path / "pred.ply"
    run_predict(os.path.join(run_dir, "best.pt"), [str(tmp_path / "plot.ply")], str(out), log=log)
    pred = io.read_cloud(str(out))
    assert "pred" in pred and len(pred["xyz"]) > 1000


# --------------------------------------------------------------------------- registration
def test_register_recovers_known_transform(tmp_path, forest):
    from junglelook.register import register

    xyz, labels, centres = forest
    th = np.radians(-79.0)                          # target -> "UTM" frame of the annotations
    T_true = np.eye(4)
    T_true[:3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    T_true[:3, 3] = [445600.0, 5696580.0, 472.8]
    stump = xyz[labels == STUMP]
    owner = np.linalg.norm(stump[:, None, :2] - centres[None], axis=-1).argmin(1)
    files = []
    for i in range(len(centres)):
        p = stump[owner == i] @ T_true[:3, :3].T + T_true[:3, 3]
        files.append(str(tmp_path / f"stump_{i}.ply"))
        io.write_ply(files[-1], p)
    T = register(files, xyz, log=lambda *a: None)
    back = (stump @ T_true[:3, :3].T + T_true[:3, 3]) @ T[:3, :3].T + T[:3, 3]
    assert np.abs(back - stump).max() < 0.05          # annotation -> target maps onto the original points


# --------------------------------------------------------------------------- cli / render
def test_cli_parses_every_command():
    from junglelook.__main__ import build_parser

    ap = build_parser()
    assert ap.parse_args(["run", "-c", "x.yaml", "--set", "train.epochs=1"]).set == ["train.epochs=1"]
    predict = ap.parse_args(["predict", "--checkpoint", "a.pt", "--input", "b.laz", "--output", "c.ply"])
    assert predict.input == ["b.laz"]
    assert ap.parse_args(["demo"]).output_dir == "work/demo"


def test_render_writes_png(tmp_path):
    from junglelook.render import render_file

    xyz, labels, _ = make_forest(size=10, n_stumps=3, n_shrubs=1, n_trees=1, density=50)
    io.write_ply(str(tmp_path / "c.ply"), xyz, {"pred": labels})
    render_file(str(tmp_path / "c.ply"), "pred", str(tmp_path / "c.png"))
    assert (tmp_path / "c.png").stat().st_size > 10_000


def test_demo_config_is_packaged():
    from junglelook.config import load_config
    from junglelook.demo import CONFIG

    cfg = load_config(CONFIG)
    assert cfg["pseudolabel"]["classes"] == ["ground", "stump", "other"] and cfg["reference"]["sources"]
