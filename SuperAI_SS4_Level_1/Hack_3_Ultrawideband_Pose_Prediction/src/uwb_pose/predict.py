"""Inference entry point: encode the test samples and write a Kaggle submission CSV.

`predict.checkpoint` may name a single .pt file or a run directory. Pointing it at a
cross-validation run directory averages the folds' probabilities, which costs nothing at this
scale and is steadier than betting on whichever fold happened to score best.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import load_split
from .dataset import EncodedStore, UWBImageDataset
from .model import load_checkpoint
from .utils import describe_device, ensure_dir, get_device

log = logging.getLogger("uwb_pose.predict")


def _checkpoints(path: Path) -> list[Path]:
    if path.is_dir():
        found = sorted(path.glob("*.pt"))
        if not found:
            raise FileNotFoundError(f"no .pt checkpoints in {path}")
        return found
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} not found — train first, or set predict.checkpoint")
    return [path]


@torch.no_grad()
def run(cfg: Config) -> pd.DataFrame:
    device = get_device()
    log.info("device: %s", describe_device(device))

    paths = _checkpoints(cfg.resolve(cfg.predict.checkpoint))
    log.info("using %d checkpoint(s) from %s", len(paths), cfg.predict.checkpoint)

    # Encode the test split exactly as training did, but with the paths from the live config.
    _, train_cfg, classes = load_checkpoint(paths[0], device)
    train_cfg.data.root = cfg.data.root
    train_cfg.data.layout = cfg.data.layout
    train_cfg.data.test_csv = cfg.data.test_csv
    train_cfg.data.test_dir = cfg.data.test_dir
    train_cfg.encode.cache_dir = cfg.encode.cache_dir
    train_cfg.train.num_workers = cfg.train.num_workers

    samples, _, ids, _ = load_split(train_cfg, "test", classes=classes)
    store = EncodedStore(samples, train_cfg, tag="test")
    ds = UWBImageDataset(store, np.arange(len(samples)), None, augment=False)
    dl = DataLoader(ds, batch_size=cfg.predict.batch_size, shuffle=False,
                    num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda")

    total = np.zeros((len(samples), len(classes)), dtype=np.float64)
    for ckpt in paths:
        model, _, ckpt_classes = load_checkpoint(ckpt, device)
        if ckpt_classes != classes:
            raise ValueError(f"{ckpt.name} was trained on different classes: {ckpt_classes}")
        probs = []
        for images, _ in tqdm(dl, desc=f"predict:{ckpt.stem}", unit="batch", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        total += np.concatenate(probs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preds = [classes[i] for i in (total / len(paths)).argmax(1)]

    id_name = train_cfg.data.id_col or "id"
    label_name = train_cfg.data.label_col or "class"
    sub = pd.DataFrame({id_name: ids, label_name: preds})

    out_path = cfg.resolve(cfg.predict.out_csv)
    ensure_dir(out_path.parent)
    sub.to_csv(out_path, index=False)
    log.info("wrote %s (%d rows)", out_path, len(sub))
    log.info("prediction distribution:\n%s", sub[label_name].value_counts().sort_index().to_string())
    return sub
