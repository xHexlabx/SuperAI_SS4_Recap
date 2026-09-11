"""Kaggle download + file indexing.

Everything this module touches lives under `data.root` (gitignored), so a fresh clone
rebuilds its dataset with one command instead of carrying 35 MB of JPEGs in git.

The competition ships three folders:
  queries/queries/{0..21}.jpg   โลโก้ต้นฉบับ 1 รูปต่อคลาส  -> คลาส 0..21
  train/train/<brand>/*.jpg     รูปโลโก้ในสภาพจริง แยกโฟลเดอร์ตามแบรนด์ (173 โฟลเดอร์)
  test/images/<uuid>.jpg        1120 รูปที่ต้องทำนาย
`train/` ไม่ได้บอกว่าโฟลเดอร์ไหนตรงกับ query ไหน — `img_search.mapping` เป็นตัวจับคู่ให้
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("img_search.data")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def _kaggle_bin() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise RuntimeError(
            "kaggle CLI not found. Install it with `uv tool install kaggle`, then put your "
            "API token in ~/.kaggle/kaggle.json (Kaggle → Settings → Create New Token)."
        )
    return exe


def download(cfg, force: bool = False) -> Path:
    """Download and unzip the competition data into `data.root`."""
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
        cmd = [_kaggle_bin(), "datasets", "download", "-d", cfg.data.kaggle_slug, "-p", str(dest)]
    else:
        raise ValueError(f"data.kaggle_kind must be 'dataset' or 'competition', got {kind!r}")

    log.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    for zf in sorted(dest.glob("*.zip")):
        log.info("unzipping %s", zf.name)
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
        zf.unlink()

    marker.write_text(f"{kind}:{cfg.data.kaggle_slug}\n", encoding="utf-8")
    log.info("dataset ready at %s", dest)
    return dest


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}. Run `make data` first.")
    return path


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def queries_index(cfg) -> pd.DataFrame:
    """The 22 reference logos. The filename stem *is* the class id."""
    folder = _require(cfg.resolve(cfg.data.root) / cfg.data.queries_dir, "queries folder")
    rows = []
    for p in _list_images(folder):
        try:
            cls = int(p.stem)
        except ValueError as exc:  # a query filename that is not an integer breaks the label map
            raise ValueError(f"query filename {p.name!r} is not <class-id>.jpg") from exc
        rows.append({"path": str(p), "cls": cls, "source": "query", "folder": f"__query_{cls}"})
    df = pd.DataFrame(rows).sort_values("cls", ignore_index=True)

    expected = set(range(cfg.data.n_classes))
    got = set(df["cls"])
    if got != expected:
        raise ValueError(f"expected query classes {sorted(expected)}, found {sorted(got)}")
    return df


def train_index(cfg) -> pd.DataFrame:
    """Every train image with the brand folder it came from (no class ids yet)."""
    root = _require(cfg.resolve(cfg.data.root) / cfg.data.train_dir, "train folder")
    rows = [
        {"path": str(p), "folder": sub.name, "source": "train"}
        for sub in sorted(root.iterdir())
        if sub.is_dir()
        for p in _list_images(sub)
    ]
    if not rows:
        raise FileNotFoundError(f"no images under {root}")
    return pd.DataFrame(rows)


def test_index(cfg) -> pd.DataFrame:
    """Test images, ordered to match sample_submission so we can write it back directly."""
    folder = _require(cfg.resolve(cfg.data.root) / cfg.data.test_dir, "test folder")
    sub = sample_submission(cfg)
    id_col = cfg.data.id_col

    on_disk = {p.name: p for p in _list_images(folder)}
    missing = [n for n in sub[id_col] if n not in on_disk]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} file(s) listed in sample_submission are missing from {folder}, "
            f"e.g. {missing[:3]}"
        )
    return pd.DataFrame(
        {"path": [str(on_disk[n]) for n in sub[id_col]], id_col: sub[id_col], "source": "test"}
    )


def sample_submission(cfg) -> pd.DataFrame:
    path = _require(cfg.resolve(cfg.data.root) / cfg.data.sample_submission, "sample_submission.csv")
    return pd.read_csv(path)


def all_images(cfg) -> pd.DataFrame:
    """queries + train + test in one frame — one embedding pass covers every command."""
    q, t, s = queries_index(cfg), train_index(cfg), test_index(cfg)
    return pd.concat(
        [q[["path", "source"]], t[["path", "source"]], s[["path", "source"]]], ignore_index=True
    ).drop_duplicates("path", ignore_index=True)
