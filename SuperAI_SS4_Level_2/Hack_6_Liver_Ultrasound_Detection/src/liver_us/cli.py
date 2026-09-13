"""Command-line front ends. Every command takes `--config` and repeatable `--set a.b=value`."""
from __future__ import annotations

import argparse
import subprocess
import sys

from .config import load_config
from .utils import setup_logging


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set model.weights=yolo26l.pt")
    return p


def download_main(argv: list[str] | None = None) -> int:
    p = _parser("Download the competition zip from Kaggle and stream-extract it into datasets/raw")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    log = setup_logging()
    cfg = load_config(args.config, args.overrides)
    raw = cfg.resolve(cfg.data.root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "image_sizes.csv").exists():
        log.info("data already extracted in %s", raw); return 0
    zip_path = raw / f"{cfg.data.kaggle_slug}.zip"
    if not zip_path.exists():
        subprocess.run(["kaggle", "competitions", "download", "-c", cfg.data.kaggle_slug, "-p", str(raw)], check=True)
    from .extract import extract
    extract(zip_path, raw)
    zip_path.unlink()
    log.info("extracted into %s and removed the zip", raw)
    return 0


def prepare_main(argv: list[str] | None = None) -> int:
    p = _parser("Build datasets/index.csv and the YOLO symlink layout")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()
    from .data import build_index
    from .prepare import build_yolo_dataset
    build_index()
    build_yolo_dataset(load_config(args.config, args.overrides))
    return 0


def train_main(argv: list[str] | None = None) -> int:
    p = _parser("Train one detector")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()
    from . import train
    train.run(load_config(args.config, args.overrides))
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    p = _parser("Predict val+test, tune the threshold on val, write submissions/submission.csv")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()
    from . import predict
    predict.run(load_config(args.config, args.overrides))
    return 0
