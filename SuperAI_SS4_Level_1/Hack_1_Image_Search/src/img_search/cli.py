"""Command-line front ends. Every command takes `--config` and repeatable `--set a.b=value`."""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .utils import setup_logging


def _parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set match.threshold=0.6")
    return p


def download_main(argv: list[str] | None = None) -> int:
    p = _parser("Download the image-search competition data from Kaggle")
    p.add_argument("--force", action="store_true", help="re-download even if the data is present")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import data

    data.download(load_config(args.config, args.overrides), force=args.force)
    return 0


def embed_main(argv: list[str] | None = None) -> int:
    p = _parser("Embed every query/train/test image and cache the vectors")
    p.add_argument("--models", nargs="*", default=None, help="encoder keys (default: embed.models)")
    p.add_argument("--all", action="store_true", help="every encoder in the registry")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from .embed import embed_all
    from .encoders import ENCODERS

    cfg = load_config(args.config, args.overrides)
    models = list(ENCODERS) if args.all else args.models
    embed_all(cfg, models, force=args.force)
    return 0


def map_main(argv: list[str] | None = None) -> int:
    p = _parser("Match the 22 query logos to their brand folders in train/")
    p.add_argument("--models", nargs="*", default=None,
                   help="encoders to average over (default: a CLIP + a SigLIP + a DINO)")
    p.add_argument("--min-score", type=float, default=0.0, help="refuse matches weaker than this")
    p.add_argument("--dry-run", action="store_true", help="print the table without writing the YAML")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import mapping

    cfg = load_config(args.config, args.overrides)
    table = mapping.build_map(cfg, args.models, min_score=args.min_score)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not args.dry_run:
        mapping.write_map(cfg, table)
        print("\nตรวจด้วยตาต่อ:  uv run python scripts/visualize.py map")
    return 0


def benchmark_main(argv: list[str] | None = None) -> int:
    p = _parser("Score every encoder on the validation split built from train/")
    p.add_argument("--models", nargs="*", default=None,
                   help="encoder keys; join with + for an ensemble (default: benchmark.models)")
    p.add_argument("--all", action="store_true", help="every encoder in the registry")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import benchmark
    from .encoders import ENCODERS

    cfg = load_config(args.config, args.overrides)
    models = list(ENCODERS) if args.all else args.models
    benchmark.run(cfg, models)
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    p = _parser("Write submissions/submission.csv")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    from . import predict

    predict.run(load_config(args.config, args.overrides))
    return 0
