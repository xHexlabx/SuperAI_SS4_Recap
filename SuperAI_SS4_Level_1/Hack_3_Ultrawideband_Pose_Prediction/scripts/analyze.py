#!/usr/bin/env python3
"""Out-of-fold confusion matrix and per-class report for a cross-validation run.

    uv run python scripts/analyze.py models/<run_name>

Every training sample is scored by the one fold that did not train on it, so the matrix covers
all 647 samples without a model ever grading its own training data. That is the view that says
*which* classes a run confuses, which an accuracy number cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uwb_pose.config import load_config  # noqa: E402
from uwb_pose.data import load_class_names, load_split  # noqa: E402
from uwb_pose.dataset import EncodedStore, UWBImageDataset  # noqa: E402
from uwb_pose.model import load_checkpoint  # noqa: E402
from uwb_pose.train import _folds  # noqa: E402
from uwb_pose.utils import get_device, seed_everything, setup_logging  # noqa: E402

EN = ["fall", "jump", "lie down", "run", "stand→sit", "sit→stand", "walk"]


@torch.no_grad()
def main() -> int:
    setup_logging()
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    ckpts = sorted(run_dir.glob("fold*.pt"))
    if not ckpts:
        print(f"no fold*.pt in {run_dir}")
        return 2

    device = get_device()
    _, cfg, classes = load_checkpoint(ckpts[0], device)
    cfg.data.root = load_config().data.root  # paths come from the live config, not the checkpoint
    cfg.encode.cache_dir = load_config().encode.cache_dir

    seed_everything(cfg.data.seed)
    samples, y, _, _ = load_split(cfg, "train")
    store = EncodedStore(samples, cfg, tag="train")

    cfg.train.folds = len(ckpts)
    oof = np.full(len(y), -1, dtype=np.int64)

    for k, (_, va_idx) in enumerate(_folds(cfg, y)):
        model, _, _ = load_checkpoint(ckpts[k], device)
        ds = UWBImageDataset(store, va_idx, None, augment=False)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        preds = []
        for images, _ in dl:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                preds.append(model(images.to(device)).argmax(1).cpu().numpy())
        oof[va_idx] = np.concatenate(preds)
        del model
        torch.cuda.empty_cache()

    assert (oof >= 0).all(), "some samples were never in a validation fold"

    names = load_class_names(cfg) or classes
    labels = [f"{c} {EN[int(c)]}" if int(c) < len(EN) else c for c in classes]

    print(f"\nout-of-fold accuracy: {(oof == y).mean():.4f}  ({len(y)} samples, {len(ckpts)} folds)\n")
    cm = confusion_matrix(y, oof, labels=range(len(classes)))
    print("confusion matrix (row = true, col = predicted)")
    print(pd.DataFrame(cm, index=labels, columns=[c for c in classes]).to_string())
    print("\n" + classification_report(y, oof, target_names=labels, digits=3, zero_division=0))

    # The pairs worth reading first: highest off-diagonal counts, both directions summed.
    pairs = [(cm[i, j] + cm[j, i], labels[i], labels[j])
             for i in range(len(classes)) for j in range(i + 1, len(classes))]
    print("most-confused pairs:")
    for n, a, b in sorted(pairs, reverse=True)[:5]:
        if n:
            print(f"  {n:3d}  {a}  <->  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
