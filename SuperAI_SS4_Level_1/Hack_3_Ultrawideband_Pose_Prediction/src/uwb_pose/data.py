"""Kaggle download + raw CSV -> (signals, labels) arrays.

Everything this module writes lands under `data.root` (gitignored), so a fresh clone rebuilds
its dataset with one command instead of carrying it in git.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("uwb_pose.data")

_NON_SIGNAL_HINTS = ("id", "label", "class", "target", "pose", "subject", "file", "name", "index")


def _kaggle_bin() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise RuntimeError(
            "kaggle CLI not found. Install it with `uv tool install kaggle`, "
            "then authenticate with `kaggle auth login`."
        )
    return exe


def download(cfg, force: bool = False) -> Path:
    """Download and unzip the configured Kaggle dataset/competition into `data.root`."""
    if not cfg.data.kaggle_slug:
        raise ValueError(
            "data.kaggle_slug is empty — set it in configs/default.yaml or pass "
            "`--set data.kaggle_slug=<owner/dataset>`."
        )
    dest = cfg.resolve(cfg.data.root)
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".downloaded"
    if marker.exists() and not force:
        log.info("dataset already present at %s (use --force to re-download)", dest)
        return dest

    kind = cfg.data.kaggle_kind
    if kind == "competition":
        cmd = [_kaggle_bin(), "competitions", "download", "-c", cfg.data.kaggle_slug, "-p", str(dest)]
    elif kind == "dataset":
        cmd = [_kaggle_bin(), "datasets", "download", "-d", cfg.data.kaggle_slug, "-p", str(dest), "--unzip"]
    else:
        raise ValueError(f"data.kaggle_kind must be 'dataset' or 'competition', got {kind!r}")

    log.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    for zf in dest.glob("*.zip"):
        log.info("unzipping %s", zf.name)
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()

    marker.write_text(f"{kind}:{cfg.data.kaggle_slug}\n", encoding="utf-8")
    log.info("dataset ready at %s", dest)
    return dest


def read_split(cfg, split: str) -> pd.DataFrame:
    """Read the train or test CSV named in the config."""
    name = cfg.data.train_csv if split == "train" else cfg.data.test_csv
    path = cfg.resolve(cfg.data.root) / name
    if not path.exists():
        matches = sorted(cfg.resolve(cfg.data.root).rglob(name))
        if not matches:
            raise FileNotFoundError(
                f"{path} not found. Run the download step first, or point data.{split}_csv at "
                f"the right file (files present: {_list_files(cfg)})."
            )
        path = matches[0]
    log.info("reading %s (%s)", path.name, split)
    return pd.read_csv(path)


def _list_files(cfg) -> list[str]:
    root = cfg.resolve(cfg.data.root)
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.csv"))[:20]


def resolve_columns(cfg, df: pd.DataFrame, split: str = "train") -> tuple[str | None, str | None, list[str]]:
    """Work out which column is the label, which is the id, and which hold the signal.

    The test split legitimately has no label column, so a missing label is only an error
    on the training split.
    """
    label_col = cfg.data.label_col
    if label_col is not None and label_col not in df.columns:
        guess = next((c for c in df.columns if c.lower() in ("label", "class", "target", "pose", "y")), None)
        if guess is not None:
            log.warning("label column %r not found — using %r instead", label_col, guess)
            label_col = guess
        elif split == "train":
            raise KeyError(
                f"label column {label_col!r} not in CSV. Columns start with: {list(df.columns[:10])}"
            )
        else:
            label_col = None

    id_col = cfg.data.id_col
    if id_col is not None and id_col not in df.columns:
        raise KeyError(f"id column {id_col!r} not in CSV. Columns start with: {list(df.columns[:10])}")
    if id_col is None:
        id_col = next((c for c in df.columns if c.lower() in ("id", "sample_id", "row_id")), None)

    if cfg.data.signal_cols:
        signal_cols = list(cfg.data.signal_cols)
        missing = [c for c in signal_cols if c not in df.columns]
        if missing:
            raise KeyError(f"signal columns missing from CSV: {missing[:10]}")
    else:
        drop = {c for c in (label_col, id_col) if c}
        signal_cols = [
            c for c in df.columns
            if c not in drop
            and pd.api.types.is_numeric_dtype(df[c])
            and not any(h == c.lower() for h in _NON_SIGNAL_HINTS)
        ]
    if not signal_cols:
        raise ValueError("no numeric signal columns found — set data.signal_cols explicitly")
    return label_col, id_col, signal_cols


def to_arrays(cfg, df: pd.DataFrame, split: str, classes: list[str] | None = None):
    """Turn a raw dataframe into (X, y, ids, classes).

    `y` is None for the test split when it carries no label column.
    """
    label_col, id_col, signal_cols = resolve_columns(cfg, df, split)
    X = df[signal_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ids = df[id_col].to_numpy() if id_col else np.arange(len(df))

    y = None
    if label_col and label_col in df.columns and split == "train":
        raw = df[label_col]
        if classes is None:
            classes = [str(c) for c in sorted(raw.astype(str).unique())]
        mapping = {c: i for i, c in enumerate(classes)}
        unknown = set(raw.astype(str)) - set(mapping)
        if unknown:
            raise ValueError(f"labels not seen at training time: {sorted(unknown)[:10]}")
        y = raw.astype(str).map(mapping).to_numpy(dtype=np.int64)

    log.info(
        "%s split: %d samples, %d signal columns%s",
        split, X.shape[0], X.shape[1],
        f", {len(classes)} classes" if classes else "",
    )
    return X, y, ids, classes


def train_val_split(X: np.ndarray, y: np.ndarray, val_size: float, seed: int):
    """Stratified split, so every pose class is represented in validation."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(X))
    tr, va = train_test_split(idx, test_size=val_size, random_state=seed, stratify=y)
    return X[tr], y[tr], X[va], y[va]


# --------------------------------------------------------------------------------------
# npy_dir layout: one .npy per sample plus an annotations CSV (the shape this Kaggle
# competition ships). Samples stay on disk and are memory-mapped at encode time, so a
# dataset far bigger than RAM still trains.
# --------------------------------------------------------------------------------------

def detect_layout(cfg) -> str:
    """Pick the layout from what is actually on disk, unless the config forces one."""
    if cfg.data.layout != "auto":
        return cfg.data.layout
    root = cfg.resolve(cfg.data.root)
    if (root / cfg.data.annotations_csv).exists():
        return "npy_dir"
    # Look only in the configured sample directories — the encode cache also holds .npy
    # files and would otherwise make every dataset look like this layout.
    for sub in (cfg.data.train_dir, cfg.data.test_dir):
        d = root / sub
        if d.is_dir() and next(d.glob(f"*{cfg.data.npy_suffix}"), None) is not None:
            return "npy_dir"
    return "wide_csv"


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], exclude: set[str] = frozenset()) -> str | None:
    for name in candidates:
        for col in df.columns:
            if col in exclude:
                continue
            if col.lower() == name:
                return col
    for col in df.columns:
        if col in exclude:
            continue
        if any(name in col.lower() for name in candidates):
            return col
    return None


def load_class_names(cfg) -> list[str] | None:
    """Read classes.csv if the competition ships one, so label ids keep their real names."""
    path = cfg.resolve(cfg.data.root) / cfg.data.classes_csv
    if not path.exists():
        return None
    df = pd.read_csv(path)
    name_col = _pick_column(df, ("name", "class", "label", "pose"))
    if name_col is None:
        return None
    id_col = _pick_column(df, ("id", "index", "idx"), exclude={name_col})
    if id_col is not None:
        df = df.sort_values(id_col)
    names = [str(v) for v in df[name_col].tolist()]
    log.info("classes.csv: %d classes -> %s", len(names), names)
    return names


def load_npy_split(cfg, split: str, classes: list[str] | None = None):
    """Return (paths, y, ids, classes) for a directory-of-.npy split."""
    root = cfg.resolve(cfg.data.root)
    sample_dir = root / (cfg.data.train_dir if split == "train" else cfg.data.test_dir)
    if not sample_dir.exists():
        found = sorted({p.parent.relative_to(root).as_posix() for p in root.rglob(f"*{cfg.data.npy_suffix}")})
        raise FileNotFoundError(
            f"{sample_dir} not found. Directories holding .npy files: {found[:10]}"
        )

    if split == "train":
        ann_path = root / cfg.data.annotations_csv
        if not ann_path.exists():
            raise FileNotFoundError(f"{ann_path} not found — needed to label the training split")
        ann = pd.read_csv(ann_path)

        file_col = cfg.data.file_col or _pick_column(ann, ("file", "name", "image", "id", "path"))
        if file_col is None:
            raise KeyError(f"cannot tell which column of {ann_path.name} holds the filename: {list(ann.columns)}")
        label_col = cfg.data.label_col if cfg.data.label_col in ann.columns else None
        label_col = label_col or _pick_column(ann, ("label", "class", "target", "pose"), exclude={file_col})
        if label_col is None:
            raise KeyError(f"cannot tell which column of {ann_path.name} holds the label: {list(ann.columns)}")
        log.info("annotations.csv: file column %r, label column %r", file_col, label_col)

        stems = ann[file_col].astype(str).map(lambda v: Path(v).stem)
        raw_labels = ann[label_col]

        # Keep the label space exactly as the CSV writes it, so the submission matches
        # sample_submission.csv. classes.csv is used for readable logging only.
        labels = raw_labels.astype(str)
        if classes is None:
            uniq = labels.unique()
            numeric = all(v.lstrip("-").isdigit() for v in uniq)
            classes = sorted(uniq, key=int) if numeric else sorted(uniq)
        names = load_class_names(cfg)
        if names:
            pretty = [f"{c}={names[int(c)]}" if c.lstrip("-").isdigit() and int(c) < len(names) else c
                      for c in classes]
            log.info("label space: %s", ", ".join(pretty))

        paths, y, ids = [], [], []
        missing = 0
        for stem, label in zip(stems, labels):
            fp = sample_dir / f"{stem}{cfg.data.npy_suffix}"
            if not fp.exists():
                missing += 1
                continue
            paths.append(fp)
            y.append(classes.index(label))
            ids.append(stem)
        if missing:
            log.warning("%d rows in annotations.csv have no matching .npy file", missing)
        if not paths:
            raise FileNotFoundError(f"no .npy file in {sample_dir} matched annotations.csv")
        log.info("train split: %d samples, %d classes", len(paths), len(classes))
        return paths, np.asarray(y, dtype=np.int64), np.asarray(ids), classes

    paths = sorted(sample_dir.glob(f"*{cfg.data.npy_suffix}"))
    if not paths:
        raise FileNotFoundError(f"no {cfg.data.npy_suffix} files in {sample_dir}")
    ids = np.asarray([p.stem for p in paths])
    log.info("test split: %d samples", len(paths))
    return paths, None, ids, classes


def load_split(cfg, split: str, classes: list[str] | None = None):
    """Layout-agnostic loader: returns (samples, y, ids, classes).

    `samples` is either an (n, length) array (wide_csv) or a list of .npy paths (npy_dir);
    the Dataset handles both.
    """
    layout = detect_layout(cfg)
    if layout == "npy_dir":
        return load_npy_split(cfg, split, classes)
    df = read_split(cfg, split)
    return to_arrays(cfg, df, split, classes)


def split_indices(n: int, y: np.ndarray, val_size: float, seed: int):
    """Stratified index split that works for both array- and path-backed sample lists."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    return train_test_split(idx, test_size=val_size, random_state=seed, stratify=y)


def take(samples, idx: np.ndarray):
    if isinstance(samples, np.ndarray):
        return samples[idx]
    return [samples[i] for i in idx]
