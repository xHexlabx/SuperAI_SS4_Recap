"""Encoding store + Torch Dataset.

Encoding a sample costs far more than a forward pass, so re-encoding every epoch would leave
the GPU idle. `EncodedStore` encodes a whole split once into a float16 memmap; every fold and
every later run with the same encode settings reads straight from it.
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .signal2image import encode_multichannel

log = logging.getLogger("uwb_pose.dataset")

# ImageNet statistics — the pretrained timm backbones expect inputs normalised this way.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _raw_sample(samples, idx: int) -> np.ndarray:
    """The sample as stored — a CSV row, or the contents of its .npy file."""
    if isinstance(samples, np.ndarray):
        return samples[idx]
    return np.asarray(np.load(samples[idx], mmap_mode="r"))


class EncodedStore:
    """Images for one whole split, encoded once and memory-mapped.

    Keyed by the encode settings *and* the sample count, so changing an encoder or the split
    produces a new file rather than silently reusing stale images. Folds index into one store,
    which is why the key deliberately describes the full split and not a subset.
    """

    def __init__(self, samples, cfg, tag: str):
        self.samples = samples
        self.cfg = cfg
        self.size = cfg.encode.img_size
        self.n_channels = cfg.data.n_channels
        self.array: np.ndarray | None = None
        if cfg.encode.cache:
            self.array = self._build(tag)

    def _key(self, tag: str) -> str:
        payload = {
            "tag": tag,
            "n": len(self.samples),
            "shape": list(np.shape(_raw_sample(self.samples, 0))),
            "n_channels": self.n_channels,
            "encode": self.cfg.to_dict()["encode"],
        }
        payload["encode"].pop("cache", None)
        payload["encode"].pop("cache_dir", None)
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _build(self, tag: str) -> np.ndarray:
        cache_dir = self.cfg.resolve(self.cfg.encode.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{tag}_{self._key(tag)}.npy"
        shape = (len(self.samples), 3, self.size, self.size)

        if path.exists():
            log.info("using cached encodings %s", path.name)
            return np.load(path, mmap_mode="r")

        log.info("encoding %d samples -> %s (%.2f GB)", len(self.samples), path.name,
                 float(np.prod(shape)) * 2 / 1024**3)
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=shape)
        for i in tqdm(range(len(self.samples)), desc=f"encode:{tag}", unit="sample"):
            mm[i] = encode_multichannel(
                _raw_sample(self.samples, i), self.n_channels, self.cfg.encode
            ).astype(np.float16)
        mm.flush()
        del mm
        tmp.rename(path)
        return np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.samples)

    def image(self, idx: int) -> np.ndarray:
        if self.array is not None:
            return np.asarray(self.array[idx], dtype=np.float32)
        return encode_multichannel(_raw_sample(self.samples, idx), self.n_channels, self.cfg.encode)


class UWBImageDataset(Dataset):
    """A view over part of an `EncodedStore`: yields (image, label) as normalised 3xHxW."""

    def __init__(self, store: EncodedStore, indices: np.ndarray, y: np.ndarray | None,
                 augment: bool = False):
        self.store = store
        self.indices = np.asarray(indices)
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = self.store.image(idx)
        if self.augment:
            img = _augment(img)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if self.y is None:
            return tensor, 0
        return tensor, int(self.y[i])


def _augment(img: np.ndarray) -> np.ndarray:
    """Light augmentation only.

    A range-Doppler or spectrogram image is not a photo: flipping it horizontally reverses time
    and vertically reverses velocity, and this dataset contains class pairs that differ by
    exactly that (sitting down vs standing up). So we stick to SpecAugment-style masking and
    mild gain jitter, which model occlusion and amplitude drift without inventing a new label.
    """
    img = img.copy()
    _, h, w = img.shape
    rng = np.random

    if rng.rand() < 0.5:  # time mask
        span = rng.randint(1, max(2, w // 10))
        start = rng.randint(0, w - span)
        img[:, :, start:start + span] = 0.0
    if rng.rand() < 0.5:  # frequency / range mask
        span = rng.randint(1, max(2, h // 10))
        start = rng.randint(0, h - span)
        img[:, start:start + span, :] = 0.0
    if rng.rand() < 0.3:  # gain jitter
        img = np.clip(img * rng.uniform(0.9, 1.1), 0.0, 1.0)
    return img


def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own numpy seed.

    Workers are forked, so they inherit one numpy global RNG state and would otherwise draw the
    identical mask positions every epoch — augmentation that repeats is barely augmentation.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def build_loaders(cfg, store: EncodedStore, tr_idx, va_idx, y) -> tuple[DataLoader, DataLoader]:
    train_ds = UWBImageDataset(store, tr_idx, y[tr_idx], augment=True)
    val_ds = UWBImageDataset(store, va_idx, y[va_idx], augment=False)
    common = dict(
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.train.num_workers > 0,
        worker_init_fn=_seed_worker,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                          drop_last=len(tr_idx) > cfg.train.batch_size, **common)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **common)
    return train_dl, val_dl
