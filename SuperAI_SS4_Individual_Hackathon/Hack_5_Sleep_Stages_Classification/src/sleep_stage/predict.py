"""Turn saved test probabilities into submissions/submission.csv.

Probabilities are averaged over runs, then decoded per night with the Viterbi pass — the
night boundaries matter, so the flat probability array is cut back into nights first.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES
from .config import Config
from .data import load_dataset
from .calibrate import apply_bias
from .smooth import decode

log = logging.getLogger("sleep_stage.predict")


def _latest_run(models_dir: Path) -> Path:
    runs = sorted([p for p in models_dir.iterdir() if (p / "results.json").exists()],
                  key=lambda p: p.stat().st_mtime)
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
    lengths = [len(n.X) for n in ds.test]
    if sum(lengths) != len(probs):
        raise ValueError(f"probabilities cover {len(probs)} epochs, test has {sum(lengths)}")
    hmm = np.load(runs[0] / "hmm.npz")
    per_night = list(np.split(probs, np.cumsum(lengths)[:-1]))
    biases = [np.load(r / "class_bias.npy") for r in runs if (r / "class_bias.npy").exists()]
    if biases and cfg.smooth.calibrate:
        bias = np.mean(biases, axis=0)
        log.info("class bias %s", {c: round(float(v), 2) for c, v in zip(CLASSES, bias)})
        per_night = apply_bias(per_night, bias)
    pred = np.concatenate(decode(per_night, hmm["A"], hmm["prior"], cfg.smooth))

    ids = np.concatenate([n.ids for n in ds.test])
    sub = pd.DataFrame({"id": ids, "labels": np.array(CLASSES)[pred]})
    ss = pd.read_csv(cfg.resolve(cfg.data.root) / "sample_submission.csv", encoding="utf-8-sig")
    sub = ss[["id"]].merge(sub, on="id", how="left")
    if sub.labels.isna().any():
        raise ValueError(f"{int(sub.labels.isna().sum())} test ids have no prediction")
    out = cfg.resolve(cfg.predict.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    log.info("ensemble of %d run(s) %s -> %s | class counts %s", len(runs), [r.name for r in runs], out,
             sub.labels.value_counts().to_dict())
    return sub
