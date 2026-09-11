"""Sweep one config knob at a time on the proxy folds.

เลือก encoder ด้วย `img-benchmark` แล้วมาจูนส่วนที่เหลือด้วยตัวนี้ — ทุกอย่างอ่าน embedding จาก
cache ยกเว้นชุด `views` ที่ต้องเข้า GPU ใหม่ (เพราะเปลี่ยนวิธี preprocess)

    uv run python scripts/ablate.py reject aggregate score_norm
    uv run python scripts/ablate.py views          # ต้อง embed ใหม่ ใช้เวลานานกว่า
    uv run python scripts/ablate.py ensemble --models siglip2-so400m dinov2-l clip-l14
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_search import benchmark as benchmark_mod  # noqa: E402
from img_search import proxy as proxy_mod  # noqa: E402
from img_search.config import load_config  # noqa: E402
from img_search.utils import setup_logging  # noqa: E402

# ชื่อการทดลอง -> (จุดที่จะเปลี่ยนใน config, ค่าที่จะลอง)
SWEEPS = {
    "reject": ("match.reject", ["gallery", "threshold", "ratio", "gallery+threshold",
                                "threshold+ratio", "gallery+threshold+ratio"]),
    "aggregate": ("gallery.aggregate", ["max", "topk", "proto"]),
    "score_norm": ("match.score_norm", ["none", "zscore"]),
    "views": ("embed.views", [["crop"], ["squash"], ["pad"], ["pad", "crop"],
                              ["pad", "squash", "crop"]]),
    "pad_color": ("embed.pad_color", ["edge", "white", "black"]),
    "dino_feature": ("embed.dino_feature", ["cls", "cls+avg"]),
    "neg_margin": ("match.neg_margin", [-0.05, -0.02, 0.0, 0.02, 0.05]),
    "expand_k": ("gallery.expand_k", [0, 1, 2, 3, 5, 10, 20, 30]),
    "expand_frac": ("gallery.expand_frac", [0.0, 0.1, 0.2, 0.33, 0.5, 0.75, 1.0]),
    "expand_negatives": ("gallery.expand_negatives", [False, True]),
}


def _set(cfg, dotted: str, value) -> None:
    target = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _get(cfg, dotted: str):
    target = cfg
    for part in dotted.split("."):
        target = getattr(target, part)
    return target


def run_sweep(cfg, name: str, models: list[str]) -> pd.DataFrame:
    dotted, values = SWEEPS[name]
    original = _get(cfg, dotted)
    rows = []
    for value in values:
        _set(cfg, dotted, value)
        folds = proxy_mod.build_folds(cfg, n_folds=cfg.benchmark.folds, seed=cfg.benchmark.seed,
                                      min_images=cfg.benchmark.min_images,
                                      seen_ratio=cfg.benchmark.seen_ratio,
                                  neg_gallery_frac=cfg.benchmark.neg_gallery_frac)
        try:
            m = benchmark_mod.evaluate_model(cfg, models, folds)
        except Exception as exc:
            print(f"  {name}={value}: skipped ({exc})")
            continue
        rows.append({name: str(value), "accuracy": m["accuracy"],
                     "balanced": 0.5 * (m["known_acc"] + m["unknown_recall"]),
                     "known_acc": m["known_acc"], "unknown_recall": m["unknown_recall"],
                     "threshold": m["threshold_production"]})
    _set(cfg, dotted, original)
    return pd.DataFrame(rows)


def run_ensemble(cfg, models: list[str], max_size: int = 3) -> pd.DataFrame:
    """Every combination of up to `max_size` encoders, concatenated."""
    folds = proxy_mod.build_folds(cfg, n_folds=cfg.benchmark.folds, seed=cfg.benchmark.seed,
                                  min_images=cfg.benchmark.min_images,
                                  seen_ratio=cfg.benchmark.seen_ratio,
                                  neg_gallery_frac=cfg.benchmark.neg_gallery_frac)
    rows = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(models, size):
            try:
                m = benchmark_mod.evaluate_model(cfg, list(combo), folds)
            except Exception as exc:
                print(f"  {'+'.join(combo)}: skipped ({exc})")
                continue
            rows.append({"models": "+".join(combo), "accuracy": m["accuracy"],
                         "balanced": 0.5 * (m["known_acc"] + m["unknown_recall"]),
                         "known_acc": m["known_acc"], "unknown_recall": m["unknown_recall"],
                         "threshold": m["threshold_production"]})
    return pd.DataFrame(rows).sort_values("balanced", ascending=False, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", choices=sorted(SWEEPS) + ["ensemble"])
    p.add_argument("--models", nargs="*", default=None, help="encoder(s) to run the sweep with")
    p.add_argument("--max-size", type=int, default=3, help="ensemble: largest combination to try")
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()
    setup_logging()

    cfg = load_config(args.config, args.overrides)
    models = args.models or cfg.embed.models

    for name in args.sweeps:
        print(f"\n=== {name} ({'+'.join(models) if name != 'ensemble' else 'combinations'}) ===")
        table = run_ensemble(cfg, models, args.max_size) if name == "ensemble" \
            else run_sweep(cfg, name, models)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
