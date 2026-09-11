"""Find which brand folders in `train/` are actually one of the 22 query brands.

โฟลเดอร์ใน `train/train/` เกือบทั้งหมดเป็นแบรนด์ *อื่น* (Gourmet Market, UNIQLO, KING POWER,
AIS Fibre, ...) ซึ่งดีมาก เพราะนั่นแปลว่าเราเอามันไปเป็น **negative gallery** ของคลาส 22 ได้เลย

แต่ถ้าบังเอิญมีโฟลเดอร์ไหนเป็นแบรนด์เดียวกับ query จริง ๆ (เช่น Taco Bell) การปล่อยให้มันอยู่
ฝั่ง negative จะทำให้ภาพ test ของแบรนด์นั้นถูกตัดเป็นคลาส 22 ทั้งหมด — พังทั้งคลาส
โมดูลนี้จึงมีหน้าที่เดียว: หาโฟลเดอร์พวกนั้นให้เจอแล้วย้ายไปอยู่ฝั่งบวก

    uv run img-map --min-score 0.85          # เขียน configs/query_map.yaml
    uv run python scripts/visualize.py map   # ตรวจด้วยตาว่าจับคู่ถูกจริงไหม

เกณฑ์สูงเข้าไว้ และคะแนนเฉลี่ยจากหลายโมเดล — คู่ที่ผิดเสียหายมากกว่าคู่ที่พลาดไป เพราะถ้าพลาด
เราก็แค่เสียโอกาสได้ gallery เพิ่ม แต่ถ้าผิดคือเสียทั้งคลาส แก้มือใน YAML ได้ตลอด
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import data as data_mod
from .embed import get_embeddings

log = logging.getLogger("img_search.mapping")

MAP_MODELS = ["clip-l14", "siglip2-so400m", "dinov2-l"]


def folder_scores(cfg, models: list[str], topk: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Similarity of every query class (rows) to every train folder (columns), averaged over models."""
    queries = data_mod.queries_index(cfg)
    train = data_mod.train_index(cfg)
    folders = sorted(train["folder"].unique())
    idx_by_folder = {f: np.flatnonzero((train["folder"] == f).to_numpy()) for f in folders}

    total = np.zeros((len(queries), len(folders)), dtype=np.float64)
    used = 0
    for key in models:
        try:
            q = get_embeddings(cfg, key, queries["path"].tolist())
            t = get_embeddings(cfg, key, train["path"].tolist())
        except Exception as exc:
            log.error("mapping: skipping %s (%s)", key, exc)
            continue
        sim = q @ t.T                                   # (22, n_train)
        for j, f in enumerate(folders):
            block = sim[:, idx_by_folder[f]]
            k = min(topk, block.shape[1])
            total[:, j] += np.sort(block, axis=1)[:, -k:].mean(axis=1)
        used += 1

    if used == 0:
        raise RuntimeError("no encoder produced embeddings for the query/train mapping")
    return pd.DataFrame(total / used, index=queries["cls"].to_numpy(), columns=folders), folders


def build_map(cfg, models: list[str] | None = None, min_score: float = 0.0,
              topk: int = 5) -> pd.DataFrame:
    """Greedy one-to-one assignment of query classes to train folders."""
    scores, folders = folder_scores(cfg, models or MAP_MODELS, topk=topk)
    mat = scores.to_numpy()

    order = np.dstack(np.unravel_index(np.argsort(mat, axis=None)[::-1], mat.shape))[0]
    taken_cls: set[int] = set()
    taken_folder: set[int] = set()
    assign: dict[int, tuple[str, float]] = {}
    for i, j in order:
        if i in taken_cls or j in taken_folder or mat[i, j] < min_score:
            continue
        assign[int(scores.index[i])] = (folders[j], float(mat[i, j]))
        taken_cls.add(i)
        taken_folder.add(j)

    rows = []
    for i, cls in enumerate(scores.index):
        folder, score = assign.get(int(cls), (None, float("nan")))
        ranked = np.argsort(mat[i])[::-1]
        rows.append({
            "cls": int(cls),
            "folder": folder,
            "score": score,
            "runner_up": folders[ranked[1]],
            "runner_up_score": float(mat[i, ranked[1]]),
        })
    return pd.DataFrame(rows).sort_values("cls", ignore_index=True)


def write_map(cfg, table: pd.DataFrame, path: str | Path | None = None) -> Path:
    out = Path(path) if path else cfg.resolve(cfg.gallery.query_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# class id -> ชื่อโฟลเดอร์ใน train/train",
        "# สร้างอัตโนมัติด้วย `make map` แล้วตรวจด้วยตาผ่าน `scripts/visualize.py map`",
        "# แก้มือได้เลย ใส่ null ถ้าคลาสนั้นไม่มีโฟลเดอร์ที่ตรงกันจริง ๆ",
        "",
    ]
    for r in table.itertuples():
        # json.dumps ให้ double-quoted scalar ที่ YAML อ่านได้ตรง ๆ — ไม่มี document marker ต่อท้าย
        quoted = "null" if r.folder is None else json.dumps(r.folder, ensure_ascii=False)
        lines.append(f"{r.cls}: {quoted}   # score={r.score:.3f} "
                     f"(รองลงมา {r.runner_up} {r.runner_up_score:.3f})")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out)
    return out


def load_map(cfg) -> dict[int, str | None]:
    path = cfg.resolve(cfg.gallery.query_map)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `make map` first (it matches the 22 query logos to "
            f"the brand folders in train/)."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {int(k): (str(v) if v is not None else None) for k, v in raw.items()}
