"""Training entry point: raw radar samples -> encoded images -> fine-tuned vision backbone.

`train.folds > 1` runs stratified k-fold cross-validation. With 647 labelled samples a single
held-out split of 130 is noisy enough that a 3-point difference means little, so the ablations
and the final model selection both go through CV; the per-fold checkpoints then double as an
ensemble at prediction time.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Config
from .data import load_split
from .dataset import EncodedStore, build_loaders
from .engine import fit
from .model import build_model
from .utils import count_params, describe_device, ensure_dir, get_device, make_run_name, seed_everything

log = logging.getLogger("uwb_pose.train")


def _folds(cfg, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(y))
    if cfg.train.folds <= 1:
        tr, va = train_test_split(idx, test_size=cfg.data.val_size,
                                  random_state=cfg.data.seed, stratify=y)
        return [(tr, va)]
    skf = StratifiedKFold(n_splits=cfg.train.folds, shuffle=True, random_state=cfg.data.seed)
    return list(skf.split(idx, y))



def _run_full_data(cfg, store, y, classes, device, run_dir, run_name) -> dict:
    """Train one model on every labelled sample, for a fixed number of epochs.

    Cross-validation spends 20% of a 647-sample set on measurement, so every model reported
    here learned from 517 samples. Once the settings are fixed there is nothing left to
    measure, and that fifth of the data is better spent training. The cost is real: with no
    validation split there is no early stopping and no score for this model — pick `epochs`
    from the median best epoch the CV runs reported, and judge the result on the leaderboard.
    """
    from torch.utils.data import DataLoader

    from .dataset import UWBImageDataset
    from .engine import build_optimizer, mix_batch
    from .model import save_checkpoint

    log.warning("full_data: training on all %d samples — no validation, no early stopping, "
                "no score. Running exactly %d epochs.", len(y), cfg.train.epochs)

    ds = UWBImageDataset(store, np.arange(len(y)), y, augment=True)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
                    num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda",
                    persistent_workers=cfg.train.num_workers > 0)

    model = build_model(cfg, num_classes=len(classes)).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    log.info("model %s | %.1fM trainable params", cfg.model.name, count_params(model) / 1e6)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    steps_per_epoch = max(1, len(dl) // cfg.train.grad_accum)
    step = 0

    from tqdm import tqdm

    from .engine import lr_at

    model.train()
    for epoch in range(cfg.train.epochs):
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, (images, targets) in enumerate(tqdm(dl, desc=f"full {epoch + 1}/{cfg.train.epochs}",
                                                   leave=False)):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images, ta, tb, lam = mix_batch(images, targets, cfg)
            with torch.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
                logits = model(images)
                loss = (lam * criterion(logits, ta) + (1.0 - lam) * criterion(logits, tb))
                loss = loss / cfg.train.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % cfg.train.grad_accum == 0:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, steps_per_epoch, cfg)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            running += loss.item() * cfg.train.grad_accum * images.size(0)
            seen += images.size(0)
        if (epoch + 1) % 10 == 0 or epoch == cfg.train.epochs - 1:
            log.info("epoch %3d/%d | train_loss %.4f", epoch + 1, cfg.train.epochs,
                     running / max(1, seen))

    fold_dir = ensure_dir(run_dir / run_name)
    metrics = {"acc": float("nan"), "macro_f1": float("nan"), "full_data": True,
               "n_train": int(len(y)), "epochs": cfg.train.epochs}
    save_checkpoint(fold_dir / "fold0.pt", model, cfg, classes, metrics)
    (fold_dir / "summary.json").write_text(
        json.dumps({"config": cfg.to_dict(), "classes": classes, "summary": metrics}, indent=2),
        encoding="utf-8")
    log.info("full-data run complete: %s", fold_dir)
    return metrics


def run(cfg: Config) -> dict:
    seed_everything(cfg.data.seed)
    device = get_device()
    log.info("device: %s", describe_device(device))

    samples, y, _, classes = load_split(cfg, "train")
    if y is None:
        raise ValueError("training split has no labels — check data.label_col")

    # One store for the whole split: every fold indexes into the same encoded images.
    store = EncodedStore(samples, cfg, tag="train")

    run_dir = ensure_dir(cfg.resolve(cfg.train.out_dir))
    run_name = make_run_name(cfg.train.run_name)
    fold_dir = ensure_dir(run_dir / run_name)

    if cfg.train.full_data:
        return _run_full_data(cfg, store, y, classes, device, run_dir, run_name)

    splits = _folds(cfg, y)
    results = []
    for k, (tr_idx, va_idx) in enumerate(splits):
        if len(splits) > 1:
            log.info("--- fold %d/%d | train %d / val %d ---", k + 1, len(splits), len(tr_idx), len(va_idx))
        else:
            log.info("train %d / val %d | classes: %s", len(tr_idx), len(va_idx), classes)

        seed_everything(cfg.data.seed + k)
        train_dl, val_dl = build_loaders(cfg, store, tr_idx, va_idx, y)

        model = build_model(cfg, num_classes=len(classes)).to(device)
        if k == 0:
            log.info("model %s | %.1fM trainable params", cfg.model.name, count_params(model) / 1e6)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        ckpt_path = fold_dir / f"fold{k}.pt"
        best, history = fit(model, train_dl, val_dl, cfg, device, classes, ckpt_path)
        results.append(best)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accs = np.array([r["acc"] for r in results])
    f1s = np.array([r["macro_f1"] for r in results])
    summary = {
        "acc": float(accs.mean()),
        "acc_std": float(accs.std()),
        "macro_f1": float(f1s.mean()),
        "macro_f1_std": float(f1s.std()),
        "folds": len(results),
    }
    if len(results) > 1:
        log.info("CV over %d folds | acc %.4f ± %.4f | macro_f1 %.4f ± %.4f",
                 len(results), summary["acc"], summary["acc_std"],
                 summary["macro_f1"], summary["macro_f1_std"])

    # `models/best.pt` is what predict reads by default; the run directory keeps every fold.
    best_fold = int(np.argmax(accs))
    (run_dir / "best.pt").write_bytes((fold_dir / f"fold{best_fold}.pt").read_bytes())

    (fold_dir / "summary.json").write_text(
        json.dumps({"config": cfg.to_dict(), "classes": classes,
                    "summary": summary, "folds": results}, indent=2),
        encoding="utf-8",
    )
    log.info("run complete: %s", fold_dir)
    return summary
