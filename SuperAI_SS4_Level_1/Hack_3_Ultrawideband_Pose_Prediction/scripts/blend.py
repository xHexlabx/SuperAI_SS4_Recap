#!/usr/bin/env python3
"""Average the test predictions of several runs and write a submission.

    uv run python scripts/blend.py models/a models/b --out submissions/blend.csv

Unlike `ensemble.py` this reports no score, because it is built for models that cannot have
one: a run trained with `train.full_data=true` has seen every labelled sample, so there is no
held-out set left to measure it on. That is the trade — the model learns from 647 samples
instead of 517, and the only way to find out whether that helped is the leaderboard.

Each run gets equal weight regardless of how many checkpoints it holds, so a 5-fold run does
not outvote a single full-data model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uwb_pose.config import load_config  # noqa: E402
from uwb_pose.data import load_split  # noqa: E402
from uwb_pose.dataset import EncodedStore, UWBImageDataset  # noqa: E402
from uwb_pose.model import load_checkpoint  # noqa: E402
from uwb_pose.utils import describe_device, ensure_dir, get_device, setup_logging  # noqa: E402


@torch.no_grad()
def run_test_probs(run_dir: Path, live, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ckpts = sorted(run_dir.glob("fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no fold*.pt in {run_dir}")

    _, cfg, classes = load_checkpoint(ckpts[0], device)
    cfg.data.root = live.data.root
    cfg.data.layout = live.data.layout
    cfg.data.test_dir = live.data.test_dir
    cfg.encode.cache_dir = live.encode.cache_dir
    cfg.train.num_workers = live.train.num_workers

    samples, _, ids, _ = load_split(cfg, "test", classes=classes)
    store = EncodedStore(samples, cfg, tag="test")
    ds = UWBImageDataset(store, np.arange(len(samples)), None, augment=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg.train.num_workers)

    total = np.zeros((len(samples), len(classes)))
    for ckpt in ckpts:
        model, _, _ = load_checkpoint(ckpt, device)
        probs = []
        for images, _ in dl:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        total += np.concatenate(probs) / len(ckpts)
        del model
        torch.cuda.empty_cache()
    return total, ids, classes


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description="Average test predictions across runs")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="submissions/blend.csv")
    ap.add_argument("--weights", help="comma-separated weight per run (default: equal)")
    args = ap.parse_args()

    live = load_config()
    device = get_device()
    print(f"device: {describe_device(device)}")

    weights = ([float(w) for w in args.weights.split(",")] if args.weights
               else [1.0] * len(args.runs))
    if len(weights) != len(args.runs):
        raise SystemExit(f"{len(weights)} weights for {len(args.runs)} runs")

    total, ids, classes = None, None, None
    for path, w in zip(map(Path, args.runs), weights):
        probs, run_ids, cls = run_test_probs(path, live, device)
        n = len(sorted(path.glob("fold*.pt")))
        print(f"  {path.name:26s} {n:2d} checkpoint(s)  weight {w:g}")
        if total is None:
            total, ids, classes = np.zeros_like(probs), run_ids, cls
        elif cls != classes:
            raise SystemExit(f"{path} was trained on different classes")
        total += w * probs

    sub = pd.DataFrame({"id": ids, "class": [classes[i] for i in total.argmax(1)]})
    out = Path(args.out)
    ensure_dir(out.parent)
    sub.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(sub)} rows) from {len(args.runs)} run(s) — no score by construction")
    print(sub["class"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
