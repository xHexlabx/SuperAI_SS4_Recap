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
                   metavar="KEY=VALUE", help="override a config value, e.g. --set llm.model=Qwen/Qwen3-8B")
    return p


def download_main(argv: list[str] | None = None) -> int:
    p = _parser("Download the legal-act-classification data from Kaggle into data.root")
    p.add_argument("--force", action="store_true", help="re-download even if the data is present")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from .data import download

    download(load_config(args.config, args.overrides), force=args.force)
    return 0


def compile_main(argv: list[str] | None = None) -> int:
    p = _parser("Compile every signing-authority clause into a machine-checkable rule")
    p.add_argument("--baseline", action="store_true",
                   help="use the regex compiler instead of the LLM (no server needed)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    cfg = load_config(args.config, args.overrides)
    if args.baseline:
        from . import baseline

        baseline.run(cfg)
    else:
        from . import compile as compile_mod

        compile_mod.run(cfg)
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    p = _parser("Score the compiled rules on train and list the clauses still failing")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import evaluate

    evaluate.run(load_config(args.config, args.overrides))
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    p = _parser("Apply the compiled rules to test and write a submission CSV")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import predict

    predict.run(load_config(args.config, args.overrides))
    return 0
