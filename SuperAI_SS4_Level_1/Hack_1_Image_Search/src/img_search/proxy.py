"""A stand-in for the hidden test set, built entirely out of `train/`.

**ปัญหา**: Kaggle ไม่ให้ label ของ test มา จะเลือกโมเดล/จูน threshold ก็ต้องเดาแล้วส่งวันละ 5 ครั้ง

**ทางออก**: `train/train/` คือคลังโลโก้ ~173 แบรนด์ แบรนด์ละหนึ่งโฟลเดอร์ ซึ่งมีโครงสร้าง
เหมือนโจทย์เป๊ะ ๆ แค่เปลี่ยนชื่อแบรนด์ เราจึง "จำลองโจทย์" ขึ้นมาเองได้:

    สุ่ม 22 โฟลเดอร์มาเป็นคลาสปลอม  ->  หยิบโฟลเดอร์ละ 1 รูปเป็น "query"
                                        รูปที่เหลือของโฟลเดอร์นั้นเป็นคำตอบที่ถูก
    โฟลเดอร์ที่เหลืออีก ~151 อัน      ->  คลาส 22 ("ไม่ตรงสักอัน")

ได้ validation ที่มี label จริง ๆ หลายพันรูป โดยไม่ต้องแตะ test เลย และเพราะคลาสปลอมถูกสุ่มใหม่
ทุก fold ผลจึงไม่ผูกกับแบรนด์ชุดใดชุดหนึ่ง

โฟลเดอร์ฝั่งลบถูกหั่นสองแบบเพื่อให้เหมือนของจริง:
  seen   — รูปครึ่งหนึ่งอยู่ใน gallery อีกครึ่งอยู่ใน validation (แบรนด์ที่เราเคยเห็น)
  unseen — รูปทั้งหมดอยู่ใน validation (แบรนด์ที่ไม่เคยเห็นเลย)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as data_mod

log = logging.getLogger("img_search.proxy")


@dataclass
class Fold:
    idx: int
    gallery: pd.DataFrame       # path, cls  (0..n-1 = โลโก้อ้างอิงคลาสละ 1 รูป, 22 = negative)
    val: pd.DataFrame           # path, cls
    brands: dict[int, str]      # cls -> ชื่อโฟลเดอร์ที่ถูกสุ่มมาเป็นคลาสนั้น

    def summary(self) -> str:
        u = (self.val["cls"] == self.val["cls"].max()).sum()
        return (f"fold {self.idx}: gallery {len(self.gallery)} "
                f"(known {(self.gallery['cls'] != 22).sum()} / neg {(self.gallery['cls'] == 22).sum()}), "
                f"val {len(self.val)} (known {len(self.val) - u} / unknown {u})")


def build_folds(cfg, n_folds: int = 3, seed: int = 42, min_images: int = 4,
                seen_ratio: float = 0.5, neg_gallery_frac: float = 0.5) -> list[Fold]:
    """Draw `n_folds` independent pseudo-tasks out of the train brand folders."""
    train = data_mod.train_index(cfg)
    n_classes, unknown = cfg.data.n_classes, cfg.data.unknown_class

    by_folder = {f: b["path"].to_numpy() for f, b in train.groupby("folder", sort=True)}
    eligible = sorted(f for f, p in by_folder.items() if len(p) >= min_images)
    if len(eligible) < n_classes:
        raise ValueError(
            f"only {len(eligible)} folder(s) have >= {min_images} images, need {n_classes}"
        )

    folds = []
    for i in range(n_folds):
        rng = np.random.default_rng(seed + 1000 * i)
        chosen = list(rng.choice(eligible, size=n_classes, replace=False))
        rest = [f for f in by_folder if f not in set(chosen)]
        rng.shuffle(rest)
        n_seen = int(round(len(rest) * seen_ratio))
        seen, unseen = rest[:n_seen], rest[n_seen:]

        gal_rows, val_rows = [], []
        for cls, folder in enumerate(chosen):
            paths = by_folder[folder].copy()
            rng.shuffle(paths)
            gal_rows.append({"path": paths[0], "cls": cls, "source": "query"})
            val_rows += [{"path": p, "cls": cls} for p in paths[1:]]

        for folder in seen:
            paths = by_folder[folder].copy()
            rng.shuffle(paths)
            # ขนาดของ negative gallery มีผลโดยตรงกับคะแนน max ที่ใช้ตัดสิน — ต้องคุมได้
            cut = min(len(paths) - 1, max(1, round(len(paths) * neg_gallery_frac)))
            gal_rows += [{"path": p, "cls": unknown, "source": "negative"} for p in paths[:cut]]
            val_rows += [{"path": p, "cls": unknown} for p in paths[cut:]]
        for folder in unseen:
            val_rows += [{"path": p, "cls": unknown} for p in by_folder[folder]]

        fold = Fold(i, pd.DataFrame(gal_rows), pd.DataFrame(val_rows),
                    {c: f for c, f in enumerate(chosen)})
        log.info("%s", fold.summary())
        folds.append(fold)
    return folds
