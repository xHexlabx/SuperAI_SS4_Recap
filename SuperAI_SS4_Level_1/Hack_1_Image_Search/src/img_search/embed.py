"""Embedding cache.

หนึ่งไฟล์ .npz ต่อ (โมเดล × วิธี preprocess) เก็บ path -> vector ไว้ทั้งชุด
รันซ้ำหรือเปลี่ยนแค่วิธี match ก็ไม่ต้องผ่าน GPU อีก — benchmark 10 โมเดลจึงทำได้ในรอบเดียว
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from . import data as data_mod
from .encoders import NonFiniteEmbedding, embed_paths, load_encoder, spec_for
from .utils import ensure_dir, get_device, l2norm

log = logging.getLogger("img_search.embed")


def cache_key(cfg, model_key: str) -> str:
    spec = spec_for(model_key)
    parts = [spec.key, "-".join(cfg.embed.views), cfg.embed.pad_color]
    if spec.kind == "hf-dino":
        parts.append(cfg.embed.dino_feature)
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", "__".join(parts))


def cache_path(cfg, model_key: str) -> Path:
    return ensure_dir(cfg.resolve(cfg.embed.cache_dir)) / f"{cache_key(cfg, model_key)}.npz"


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return dict(zip(z["paths"].tolist(), z["emb"]))


def _save_cache(path: Path, store: dict[str, np.ndarray]) -> None:
    keys = sorted(store)
    np.savez(path, paths=np.array(keys), emb=np.stack([store[k] for k in keys]).astype(np.float32))


def get_embeddings(cfg, model_key: str, paths: list[str], *, force: bool = False) -> np.ndarray:
    """Return (len(paths), dim) L2-normalised embeddings, computing only what is missing."""
    path = cache_path(cfg, model_key)
    store = {} if force else _load_cache(path)

    todo = [p for p in dict.fromkeys(paths) if p not in store]
    if todo:
        vecs = _run_encoder(cfg, model_key, todo)
        store.update(zip(todo, vecs))
        _save_cache(path, store)
        log.info("cache written: %s (%d vectors)", path.name, len(store))

    return np.stack([store[p] for p in paths]).astype(np.float32)


def _run_encoder(cfg, model_key: str, todo: list[str]) -> np.ndarray:
    """Embed `todo`, halving the batch and finally falling back to CPU on CUDA OOM.

    เครื่องเดียวมักรันอย่างอื่นค้างไว้บน GPU อยู่แล้ว (เช่น vLLM) — OOM จึงไม่ควรทำให้ทั้ง
    benchmark ล้ม แค่ช้าลง
    """
    import gc

    import torch

    device = get_device()
    batch = cfg.embed.batch_size
    half = cfg.embed.amp
    while True:
        try:
            enc = load_encoder(model_key, device, cfg.embed.dino_feature,
                               half=half and device.type == "cuda")
            log.info("%s: dim=%d, input=%dpx, views=%s, %d image(s) on %s (batch %d, %s)",
                     enc.spec.key, enc.dim, enc.image_size, cfg.embed.views, len(todo),
                     device.type, batch, "fp32" if not half else "half")
            try:
                return embed_paths(enc, todo, views=cfg.embed.views, pad_color=cfg.embed.pad_color,
                                   batch_size=batch, num_workers=cfg.embed.num_workers,
                                   device=device)
            finally:
                del enc
                gc.collect()
                torch.cuda.empty_cache()
        except NonFiniteEmbedding as exc:
            if not half:
                raise
            log.warning("%s: %s — retrying in float32", model_key, exc)
            half = False
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if device.type == "cuda" and batch > 4:
                batch //= 2
                log.warning("%s: CUDA OOM, retrying with batch %d", model_key, batch)
            elif device.type == "cuda":
                free, total = (v / 1024**3 for v in torch.cuda.mem_get_info())
                log.warning("%s: CUDA OOM with only %.1f/%.1f GB free — falling back to CPU "
                            "(slow). Free the GPU and rerun for a faster pass.",
                            model_key, free, total)
                device, batch = torch.device("cpu"), cfg.embed.batch_size
            else:
                raise


def get_matrix(cfg, paths: list[str], *, models: list[str] | None = None,
               weights: list[float] | None = None, force: bool = False) -> np.ndarray:
    """Embeddings for one or more encoders, concatenated into a single space.

    การต่อเวกเตอร์ที่ normalise แล้วเข้าด้วยกัน ให้ผลเท่ากับการเฉลี่ย cosine ของแต่ละโมเดล
    (ถ่วงด้วย weight) — เป็นวิธี ensemble ที่ถูกที่สุดและไม่ต้องเทรนอะไรเพิ่ม
    """
    models = models or cfg.embed.models
    if not models:
        raise ValueError("embed.models is empty")
    weights = weights if weights is not None else list(cfg.embed.weights)
    if not weights:
        weights = [1.0] * len(models)
    if len(weights) != len(models):
        raise ValueError(f"embed.weights has {len(weights)} entries for {len(models)} models")

    blocks = []
    for key, w in zip(models, weights):
        e = get_embeddings(cfg, key, paths, force=force)
        blocks.append(e * float(np.sqrt(max(w, 0.0))))
    if len(blocks) == 1:
        return blocks[0]
    return l2norm(np.concatenate(blocks, axis=1))


def embed_all(cfg, models: list[str] | None = None, force: bool = False) -> None:
    """Warm the cache for every image in the competition, for each requested encoder."""
    paths = data_mod.all_images(cfg)["path"].tolist()
    for key in models or cfg.embed.models:
        spec = spec_for(key)
        try:
            get_embeddings(cfg, key, paths, force=force)
        except Exception as exc:  # a gated or oversized model should not kill the whole sweep
            hint = " (gated model — accept the licence on its HF page first)" if spec.gated else ""
            log.error("skipping %s: %s%s", key, exc, hint)
