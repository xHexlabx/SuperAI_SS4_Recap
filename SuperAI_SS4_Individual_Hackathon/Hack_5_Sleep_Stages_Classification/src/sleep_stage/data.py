"""Reading the competition CSVs and caching one feature matrix per night.

train/train/trainNNN.csv is a whole recording with a `Sleep_Stage` column; the label is
constant inside each 1920-row epoch. test/test_segment/testNNN/ holds the same recording
already cut into epochs — and the EDA shows the file numbering is *contiguous in time*
(TEMP, EDA and HR join across segment boundaries to within sensor resolution), so a test
subject is reassembled into one night before features are computed.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES
from .config import Config
from .features import COLUMNS, night_features

log = logging.getLogger("sleep_stage.data")
LABEL_COLUMN = "Sleep_Stage"
CACHE_VERSION = "v2"


@dataclass
class Night:
    subject: str
    X: np.ndarray                 # (n_epochs, n_features) float32
    y: np.ndarray | None          # (n_epochs,) int8 index into CLASSES, None for test
    ids: np.ndarray | None        # (n_epochs,) submission ids, test only


@dataclass
class Dataset:
    train: list[Night]
    test: list[Night]
    names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.names)


def train_dir(cfg: Config) -> Path:
    return cfg.resolve(cfg.data.root) / "train" / "train"


def test_dir(cfg: Config) -> Path:
    return cfg.resolve(cfg.data.root) / "test" / "test_segment"


def subject_ids(cfg: Config) -> tuple[list[str], list[str]]:
    tr = sorted(p.stem for p in train_dir(cfg).glob("*.csv"))
    te = sorted(p.name for p in test_dir(cfg).iterdir() if p.is_dir())
    return tr, te


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", usecols=columns)


def read_train_night(path: Path, epoch_samples: int) -> tuple[np.ndarray, np.ndarray]:
    df = _read_csv(path, COLUMNS + [LABEL_COLUMN])
    n_epochs = len(df) // epoch_samples
    raw = df[COLUMNS].to_numpy(dtype=np.float32)[: n_epochs * epoch_samples]
    labels = df[LABEL_COLUMN].to_numpy()[: n_epochs * epoch_samples].reshape(n_epochs, epoch_samples)
    first = labels[:, 0]
    mixed = int((labels != first[:, None]).any(1).sum())
    if mixed:
        log.warning("%s: %d epoch(s) with a mixed label — taking the first row", path.name, mixed)
    y = np.array([CLASSES.index(s) for s in first], dtype=np.int8)
    return raw, y


def read_test_night(folder: Path, epoch_samples: int) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(folder.glob("*.csv"), key=lambda p: int(re.search(r"_(\d+)$", p.stem).group(1)))
    parts = [_read_csv(f, COLUMNS).to_numpy(dtype=np.float32) for f in files]
    for f, part in zip(files, parts):
        if part.shape[0] != epoch_samples:
            raise ValueError(f"{f}: expected {epoch_samples} rows, got {part.shape[0]}")
    return np.concatenate(parts, axis=0), np.array([f.stem for f in files])


def _cache_path(cfg: Config, subject: str) -> Path:
    return cfg.resolve(cfg.data.cache_dir) / CACHE_VERSION / f"{subject}.npz"


def _build_one(args) -> str:
    cfg, subject = args
    out = _cache_path(cfg, subject)
    if out.exists():
        return subject
    out.parent.mkdir(parents=True, exist_ok=True)
    es = cfg.features.epoch_samples
    if subject.startswith("train"):
        raw, y = read_train_night(train_dir(cfg) / f"{subject}.csv", es)
        ids = None
    else:
        raw, ids = read_test_night(test_dir(cfg) / subject, es)
        y = None
    X, names = night_features(raw, cfg.features)
    if y is not None and len(y) != len(X):
        raise ValueError(f"{subject}: {len(X)} feature rows vs {len(y)} labels")
    np.savez_compressed(out, X=X, names=np.array(names),
                        **({"y": y} if y is not None else {}),
                        **({"ids": ids} if ids is not None else {}))
    return subject


def build_cache(cfg: Config, force: bool = False) -> None:
    tr, te = subject_ids(cfg)
    subjects = tr + te
    if force:
        for s in subjects:
            _cache_path(cfg, s).unlink(missing_ok=True)
    todo = [s for s in subjects if not _cache_path(cfg, s).exists()]
    log.info("feature cache %s: %d/%d night(s) to build", CACHE_VERSION, len(todo), len(subjects))
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=cfg.data.n_jobs) as pool:
        for i, subject in enumerate(pool.map(_build_one, [(cfg, s) for s in todo]), 1):
            log.info("  [%3d/%d] %s", i, len(todo), subject)


def load_dataset(cfg: Config) -> Dataset:
    build_cache(cfg)
    tr_ids, te_ids = subject_ids(cfg)
    names: list[str] = []
    train, test = [], []
    for subject in tr_ids + te_ids:
        d = np.load(_cache_path(cfg, subject), allow_pickle=False)
        names = names or [str(n) for n in d["names"]]
        night = Night(subject=subject, X=d["X"],
                      y=d["y"] if "y" in d else None,
                      ids=d["ids"] if "ids" in d else None)
        (train if subject.startswith("train") else test).append(night)
    log.info("loaded %d train night(s) (%d epochs) and %d test night(s) (%d epochs), %d raw features",
             len(train), sum(len(n.X) for n in train), len(test), sum(len(n.X) for n in test), len(names))
    return Dataset(train=train, test=test, names=names)
