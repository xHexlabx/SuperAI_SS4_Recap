"""Command-line front ends. Every command takes `--config` and repeatable `--set a.b=value`."""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .utils import setup_logging


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a config value, e.g. --set train.epochs=5")
    return p


def download_main(argv: list[str] | None = None) -> int:
    p = _parser("Download the UWB dataset from Kaggle into data.root")
    p.add_argument("--force", action="store_true", help="re-download even if the data is present")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from .data import download

    cfg = load_config(args.config, args.overrides)
    download(cfg, force=args.force)
    return 0


def train_main(argv: list[str] | None = None) -> int:
    p = _parser("Train the signal-to-image pose classifier")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import train

    cfg = load_config(args.config, args.overrides)
    train.run(cfg)
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    p = _parser("Run inference and write a submission CSV")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import predict

    cfg = load_config(args.config, args.overrides)
    predict.run(cfg)
    return 0
