"""Build the symlink layout ultralytics expects from the extracted competition tree.

datasets/yolo/<name>/{images,labels}/<id>[__osK].{jpg,txt}
Mobile oversampling and negative-subsampling are expressed purely as which symlinks
exist, so the raw tree is never copied.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Config
from .data import CLASSES, INDEX, RAW, build_index, load_index, find_split_dirs

log = logging.getLogger("liver_us.prepare")


def ensure_index() -> pd.DataFrame:
    if not INDEX.exists():
        log.info("building %s", INDEX)
        build_index()
    return load_index()


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    dst.symlink_to(src)


def make_split(name: str, rows: pd.DataFrame, yolo_dir: Path, oversample: dict[str, int] | None = None) -> Path:
    """Create yolo_dir/<name>/{images,labels} with one symlink per (image, copy)."""
    out = yolo_dir / name
    if out.exists():
        import shutil
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    n = 0
    for r in rows.itertuples(index=False):
        reps = (oversample or {}).get(r.image_id, 1)
        for k in range(reps):
            suffix = "" if k == 0 else f"__os{k}"
            _link(Path(r.abs_path), out / "images" / f"{r.image_id}{suffix}.jpg")
            if isinstance(r.label_path, str) and r.label_path:
                _link(Path(r.label_path), out / "labels" / f"{r.image_id}{suffix}.txt")
            n += 1
    log.info("%s: %d image links (%d unique) -> %s", name, n, len(rows), out)
    return out


def build_yolo_dataset(cfg: Config) -> Path:
    """Return the path of the data yaml for this config's data settings."""
    df = ensure_index()
    yolo_dir = cfg.resolve(cfg.data.yolo_dir)
    rng = np.random.default_rng(cfg.train.seed)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    if cfg.data.train_on_val:
        train = pd.concat([train, val])
    if cfg.data.neg_fraction < 1.0:
        neg = train[train.n_boxes == 0]
        keep = rng.random(len(neg)) < cfg.data.neg_fraction
        train = pd.concat([train[train.n_boxes > 0], neg[keep]])
    oversample = {}
    if cfg.data.mobile_oversample > 1:
        oversample = {i: cfg.data.mobile_oversample for i in train.image_id[train.source == "mobile"]}
    tag = f"os{cfg.data.mobile_oversample}_neg{cfg.data.neg_fraction:g}" + ("_tv" if cfg.data.train_on_val else "")
    if cfg.data.synth_dir:
        tag += "_" + Path(cfg.data.synth_dir).name
    train_dir = make_split(f"train_{tag}", train, yolo_dir, oversample)
    if cfg.data.synth_dir:
        sd = cfg.resolve(cfg.data.synth_dir)
        n = 0
        for img in sorted((sd / "images").glob("*.jpg")):
            _link(img, train_dir / "images" / img.name)
            lbl = sd / "labels" / (img.stem + ".txt")
            if lbl.exists():
                _link(lbl, train_dir / "labels" / lbl.name)
            n += 1
        log.info("added %d synthetic images from %s", n, sd)
    val_dir = make_split("val", val, yolo_dir)
    test_dir = make_split("test", df[df.split == "test"], yolo_dir)
    spec = dict(path=str(yolo_dir), train=str(train_dir / "images"), val=str(val_dir / "images"),
                test=str(test_dir / "images"), names={i: c for i, c in enumerate(CLASSES)})
    yml = yolo_dir / f"data_{tag}.yaml"
    yml.write_text(yaml.safe_dump(spec, sort_keys=False))
    log.info("data yaml -> %s", yml)
    return yml
