"""junglelook command line interface.

  junglelook demo                                   synthetic end-to-end demo (no data needed)
  junglelook run         -c CONFIG                  extract -> pseudolabel -> split -> train -> evaluate
  junglelook extract     -c CONFIG                  features for unlabelled clouds
  junglelook pseudolabel -c CONFIG                  rule-based labels + stump candidates
  junglelook split       -c CONFIG                  spatial train / val / test scenes
  junglelook train       -c CONFIG                  PointNet / PointNet++ training
  junglelook evaluate    -c CONFIG --checkpoint CK  metrics vs pseudo-labels and reference
  junglelook predict     --checkpoint CK --input CLOUD --output OUT.ply
  junglelook render      --input OUT.ply --field pred --output OUT.png
  junglelook register    --annotations "stumps/*.txt" --target CLOUD --output T.txt

Any config value can be overridden:  --set train.epochs=60 model.name=pointnet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

PIPELINE = ("run", "extract", "pseudolabel", "split", "train", "evaluate")


def _logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(log_dir, "log.txt"))], force=True)
    return logging.getLogger("junglelook").info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="junglelook", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    for name in PIPELINE:
        p = sub.add_parser(name, help=f"pipeline stage: {name}")
        p.add_argument("-c", "--config", required=True, help="YAML config")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
        if name == "evaluate":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--split", default="test", choices=["train", "val", "test"])

    p = sub.add_parser("predict", help="segment new, unlabelled clouds with a trained model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", nargs="+", required=True, help="cloud file(s) or globs")
    p.add_argument("--output", required=True, help=".ply, .las or .laz")
    p.add_argument("--full-resolution", action="store_true", help="label every input point, not only voxelised ones")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. infer.smooth_k=0")

    p = sub.add_parser("demo", help="end-to-end demo on synthetic forest plots")
    p.add_argument("--output-dir", default="work/demo")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="demo config overrides")

    p = sub.add_parser("render", help="top + side view PNG of a labelled cloud")
    p.add_argument("--input", required=True)
    p.add_argument("--field", default="pred", help="scalar field to colour by (pred, label, hag, prob_stump, ...)")
    p.add_argument("--output", required=True)
    p.add_argument("--classes", nargs="*", default=None, help="class names for categorical fields")
    p.add_argument("--instances", help="instances CSV to overlay (from predict)")

    p = sub.add_parser("register", help="align annotation clouds onto a target cloud (RANSAC + ICP)")
    p.add_argument("--annotations", nargs="+", required=True, help="one cloud per object, globs ok")
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True, help="4x4 transform text file")
    p.add_argument("--voxel", type=float, default=0.03)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    if args.command in PIPELINE:
        from . import pipeline
        from .config import load_config

        cfg = load_config(args.config, args.set)
        log = _logger(cfg["work_dir"])
        if args.command in ("extract", "run"):
            pipeline.run_extract(cfg, log)
        if args.command in ("pseudolabel", "run"):
            pipeline.run_pseudolabel(cfg, log)
        if args.command in ("split", "run"):
            pipeline.run_split(cfg, log)
        checkpoint = getattr(args, "checkpoint", None)
        if args.command in ("train", "run"):
            from .train import run_train

            checkpoint = os.path.join(run_train(cfg, log), "best.pt")
        if args.command in ("evaluate", "run"):
            from .infer import run_evaluate

            run_evaluate(cfg, checkpoint, getattr(args, "split", "test"), log)

    elif args.command == "predict":
        from .config import apply_overrides
        from .infer import run_predict

        log = _logger(os.path.dirname(os.path.abspath(args.output)))
        run_predict(args.checkpoint, args.input, args.output, apply_overrides({}, args.set),
                    args.full_resolution, log)

    elif args.command == "demo":
        from .demo import run_demo

        run_demo(args.output_dir, args.set, _logger(args.output_dir))

    elif args.command == "render":
        from .render import render_file

        render_file(args.input, args.field, args.output, args.classes, args.instances)
        print(f"wrote {args.output}")

    elif args.command == "register":
        from .register import main as register_main

        register_main(["--annotations", *args.annotations, "--target", args.target,
                       "--output", args.output, "--voxel", str(args.voxel)])

    if args.command not in ("render", "register"):
        logging.getLogger("junglelook").info(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
