"""Build the reference set that test images get compared against.

  positives  โลโก้ต้นฉบับ 22 รูปจาก `queries/` (คลาส 0..21)
             + รูปจากโฟลเดอร์ใน train/ ที่ `query_map.yaml` บอกว่าเป็นแบรนด์เดียวกัน (ถ้ามี)
  negatives  รูปจากโฟลเดอร์ที่เหลือใน train/ (คลาส 22)

negatives คือหัวใจของรอบนี้: แทนที่จะเดา threshold ว่า "คล้ายพอหรือยัง" เราให้คลาส 22 มีตัวแทน
จริง ~2,300 รูป แล้วถามคำถามที่ตอบง่ายกว่ามากแทน — ภาพนี้ใกล้โลโก้ที่ถาม หรือใกล้โลโก้แบรนด์อื่น
มากกว่ากัน
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as data_mod
from .mapping import load_map

log = logging.getLogger("img_search.gallery")


@dataclass
class Gallery:
    frame: pd.DataFrame          # columns: path, cls, source

    @property
    def paths(self) -> list[str]:
        return self.frame["path"].tolist()

    @property
    def cls(self) -> np.ndarray:
        return self.frame["cls"].to_numpy()


def labelled_train(cfg) -> pd.DataFrame:
    """Train images with a class: 0..21 if `query_map.yaml` matched that folder, else 22."""
    try:
        mapping = load_map(cfg)
    except FileNotFoundError:
        mapping = {}
        log.warning("no query_map.yaml — treating every train folder as class %d",
                    cfg.data.unknown_class)
    folder_to_cls = {v: k for k, v in mapping.items() if v is not None}

    train = data_mod.train_index(cfg).copy()
    missing = sorted(set(folder_to_cls) - set(train["folder"]))
    if missing:
        raise ValueError(f"query_map.yaml points at folders that do not exist: {missing}")

    train["cls"] = train["folder"].map(folder_to_cls).fillna(cfg.data.unknown_class).astype(int)
    return train


def build(cfg) -> Gallery:
    """The gallery used for the real submission."""
    sources = list(cfg.gallery.sources)
    unknown = cfg.data.unknown_class
    parts = []

    if "queries" in sources:
        parts.append(data_mod.queries_index(cfg)[["path", "cls", "source"]])

    if {"train", "negatives"} & set(sources):
        train = labelled_train(cfg)
        if "train" in sources:
            known = train[train["cls"] != unknown]
            if len(known):
                parts.append(known.assign(source="train_positive")[["path", "cls", "source"]])
        if "negatives" in sources:
            neg = train[train["cls"] == unknown]
            parts.append(neg.assign(source="negative")[["path", "cls", "source"]])

    if not parts:
        raise ValueError(f"gallery.sources {sources} produced an empty gallery")

    frame = pd.concat(parts, ignore_index=True).drop_duplicates("path", ignore_index=True)
    log.info("gallery: %d items %s", len(frame), frame["source"].value_counts().to_dict())
    return Gallery(frame)
