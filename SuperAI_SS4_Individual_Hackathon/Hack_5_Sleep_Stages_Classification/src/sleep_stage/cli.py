"""Command-line front ends. Every command takes `--config` and repeatable `--set a.b=value`."""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile

from .config import load_config
from .utils import setup_logging


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set model.name=lgbm")
    return p


def download_main(argv: list[str] | None = None) -> int:
    p = _parser("Download the competition data from Kaggle into data.root")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    log = setup_logging()
    cfg = load_config(args.config, args.overrides)
    root = cfg.resolve(cfg.data.root); root.mkdir(parents=True, exist_ok=True)
    if (root / "train" / "train").exists():
        log.info("data already present in %s", root); return 0
    subprocess.run(["kaggle", "competitions", "download", "-c", cfg.data.kaggle_slug, "-p", str(root)], check=True)
    with zipfile.ZipFile(root / f"{cfg.data.kaggle_slug}.zip") as z:
        z.extractall(root)
    log.info("extracted into %s", root)
    return 0


def cache_main(argv: list[str] | None = None) -> int:
    p = _parser("Build the per-night feature cache")
    p.add_argument("--force", action="store_true", help="rebuild nights that are already cached")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()
    from .data import build_cache
    build_cache(load_config(args.config, args.overrides), force=args.force)
    return 0


def train_main(argv: list[str] | None = None) -> int:
    p = _parser("Cross-validate a model and save test probabilities")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()
    from . import train
    train.run(load_config(args.config, args.overrides))
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    p = _parser("Write submissions/submission.csv from saved run(s)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()
    from . import predict
    predict.run(load_config(args.config, args.overrides))
    return 0
