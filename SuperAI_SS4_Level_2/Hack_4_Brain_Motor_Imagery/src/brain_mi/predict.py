"""Turn saved test probabilities (one or more runs) into a Kaggle submission.

Only EEG-derived probabilities are used here: no counter, no cue order, no oracle labels.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import CLASSES
from .preprocess import load_dataset

log = logging.getLogger("brain_mi.predict")


def _latest_run(models_dir: Path) -> Path:
    runs = sorted([p for p in models_dir.iterdir() if (p / "results.json").exists()], key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no runs in {models_dir}")
    return runs[-1]


def load_probs(run: Path, source: str) -> np.ndarray:
    files = {"full": ["test_probs_full.npy"], "folds": ["test_probs_folds.npy"],
             "both": ["test_probs_full.npy", "test_probs_folds.npy"]}[source]
    arrs = [np.load(run / f) for f in files if (run / f).exists()]
    if not arrs:
        raise FileNotFoundError(f"{run}: no test probabilities for source={source}")
    return np.mean(arrs, axis=0)


def run(cfg: Config) -> pd.DataFrame:
    models_dir = cfg.resolve(cfg.train.out_dir)
    runs = [cfg.resolve(r) if "/" in r else models_dir / r for r in cfg.predict.runs] or [_latest_run(models_dir)]
    probs = np.mean([load_probs(r, cfg.predict.source) for r in runs], axis=0)
    ds = load_dataset(cfg)
    ids = ds.test.meta.id.values
    pred = np.array(CLASSES)[probs.argmax(1)]
    sub = pd.DataFrame({"id": ids, "predict": pred})
    ss = pd.read_csv(cfg.resolve(cfg.data.root) / "sample_submission.csv")
    sub = ss[["id"]].merge(sub, on="id", how="left")
    assert sub.predict.notna().all(), "missing test ids"
    out = cfg.resolve(cfg.predict.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    log.info("ensemble of %d run(s) %s -> %s | class counts %s", len(runs), [r.name for r in runs], out,
             sub.predict.value_counts().to_dict())
    return sub
