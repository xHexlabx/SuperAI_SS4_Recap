#!/usr/bin/env python3
"""Average several cross-validation runs and measure whether it actually helps.

    uv run python scripts/ensemble.py models/final models/ablate_sg2d_fine ...
    uv run python scripts/ensemble.py models/a models/b --submit submissions/ensemble.csv

Stacking encoders into one image's RGB channels made things worse in every test — but that is
a different mechanism from averaging *separately trained* models, which only needs the runs to
make different mistakes. This script measures the difference instead of assuming it: every
combination is scored out-of-fold, on the same fold split each run used, so a combination that
does not beat its best member is visible before anything reaches a submission.
"""

from __future__ import annotations

import argparse
import itertools
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
from uwb_pose.train import _folds  # noqa: E402
from uwb_pose.utils import ensure_dir, get_device, seed_everything, setup_logging  # noqa: E402

# Above this many runs, an exhaustive 2^n search stops being worth its time.
EXHAUSTIVE_LIMIT = 12


@torch.no_grad()
def _probs(model, store, indices, device, batch_size=64, workers=4) -> np.ndarray:
    ds = UWBImageDataset(store, indices, None, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    out = []
    for images, _ in dl:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def run_oof_and_test(run_dir: Path, live, y, train_samples, device, want_test: bool):
    """Out-of-fold probabilities for the train split, and optionally test probabilities."""
    ckpts = sorted(run_dir.glob("fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no fold*.pt in {run_dir}")
    if len(ckpts) < 2:
        raise ValueError(
            f"{run_dir} has only {len(ckpts)} fold checkpoint — an interrupted run cannot "
            f"produce out-of-fold predictions for every sample. Re-run it or leave it out."
        )

    _, cfg, classes = load_checkpoint(ckpts[0], device)
    # Paths always come from the live config; only the encoding comes from the checkpoint.
    cfg.data.root = live.data.root
    cfg.data.layout = live.data.layout
    cfg.data.test_dir = live.data.test_dir
    cfg.encode.cache_dir = live.encode.cache_dir
    cfg.train.folds = len(ckpts)

    store = EncodedStore(train_samples, cfg, tag="train")
    oof = np.zeros((len(y), len(classes)))
    test_probs = None

    if want_test:
        test_samples, _, test_ids, _ = load_split(cfg, "test", classes=classes)
        test_store = EncodedStore(test_samples, cfg, tag="test")
        test_probs = np.zeros((len(test_samples), len(classes)))
    else:
        test_ids = None

    seed_everything(cfg.data.seed)
    for k, (_, va_idx) in enumerate(_folds(cfg, y)):
        model, _, _ = load_checkpoint(ckpts[k], device)
        oof[va_idx] = _probs(model, store, va_idx, device)
        if want_test:
            test_probs += _probs(model, test_store, np.arange(len(test_samples)), device) / len(ckpts)
        del model
        torch.cuda.empty_cache()

    # Every sample must have been scored by exactly one fold; a partially filled matrix would
    # otherwise sail through as rows of zeros that argmax silently reads as class 0.
    if not np.all(oof.sum(axis=1) > 0):
        missing = int((oof.sum(axis=1) == 0).sum())
        raise RuntimeError(f"{run_dir}: {missing} samples got no out-of-fold prediction")

    return oof, classes, test_probs, test_ids


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Average cross-validation runs and score every combination")
    parser.add_argument("runs", nargs="+", help="run directories, each holding fold*.pt")
    parser.add_argument("--submit", metavar="CSV", help="write the best combination's test predictions here")
    opts = parser.parse_args()
    args, submit = opts.runs, opts.submit

    live = load_config()
    device = get_device()
    seed_everything(live.data.seed)
    train_samples, y, _, _ = load_split(live, "train")

    runs, oofs, tests, n_folds, ids, classes = [], [], [], [], None, None
    for d in args:
        path = Path(d)
        oof, cls, tp, tid = run_oof_and_test(path, live, y, train_samples, device, submit is not None)
        classes = classes or cls
        if cls != classes:
            raise ValueError(f"{path} was trained on different classes")
        runs.append(path.name); oofs.append(oof); tests.append(tp); ids = tid if tid is not None else ids
        n_folds.append(len(sorted(path.glob("fold*.pt"))))
        print(f"  {path.name:28s} {n_folds[-1]:2d} folds   OOF acc {(oof.argmax(1) == y).mean():.4f}", flush=True)

    def score(combo) -> float:
        return float((np.mean([oofs[i] for i in combo], axis=0).argmax(1) == y).mean())

    solo = [score((i,)) for i in range(len(runs))]

    if len(runs) <= EXHAUSTIVE_LIMIT:
        scored = []
        for r in range(1, len(runs) + 1):
            for combo in itertools.combinations(range(len(runs)), r):
                scored.append((score(combo), combo, score(combo) - max(solo[i] for i in combo)))
        header = f"every combination of {len(runs)} runs"
    else:
        # Exhaustive search is 2^n. Plain forward selection is cheap but gets trapped: it
        # commits to the best single run, and the best *combination* often does not contain it
        # (measured — a 4-run set beat the best single by 0.6 points while excluding it).
        # Restarting the search from every run in turn costs n times as much and finds those.
        scored = []
        for start in range(len(runs)):
            chosen, current = [start], solo[start]
            remaining = set(range(len(runs))) - {start}
            scored.append((current, (start,), 0.0))
            while remaining:
                acc, pick = max((score(tuple(chosen + [i])), i) for i in remaining)
                if acc <= current:
                    break
                chosen.append(pick); remaining.discard(pick); current = acc
                combo = tuple(sorted(chosen))
                scored.append((acc, combo, acc - max(solo[i] for i in combo)))
        header = f"greedy forward selection from {len(runs)} starting points"

    print(f"\n{header}")
    print(f"{'combination':52s} {'OOF acc':>9s}  {'vs best member':>14s}")
    print("-" * 80)
    seen = set()
    for acc, combo, delta in sorted(scored, reverse=True)[:14]:
        if combo in seen:
            continue
        seen.add(combo)
        name = " + ".join(runs[i] for i in combo)
        mark = "" if len(combo) == 1 else f"{delta:+.4f}"
        print(f"{name[:52]:52s} {acc:9.4f}  {mark:>14s}")

    best_acc, best_combo, _ = max(scored)
    print(f"\nbest: {' + '.join(runs[i] for i in best_combo)}  ->  OOF acc {best_acc:.4f}")
    total = sum(n_folds[i] for i in best_combo)
    print(f"      ({len(best_combo)} run(s), {total} models in total)")

    if submit:
        # Equal weight per run, not per model: a 10-fold run should not outvote a 5-fold one
        # just for having been split more finely.
        probs = np.mean([tests[i] for i in best_combo], axis=0)
        sub = pd.DataFrame({"id": ids, "class": [classes[i] for i in probs.argmax(1)]})
        out = Path(submit)
        ensure_dir(out.parent)
        sub.to_csv(out, index=False)
        print(f"wrote {out} ({len(sub)} rows) from {len(best_combo)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
