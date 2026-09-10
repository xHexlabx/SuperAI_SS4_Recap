"""Training and evaluation loops: AMP, gradient accumulation, cosine schedule, early stopping."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

log = logging.getLogger("uwb_pose.engine")


def build_optimizer(model: nn.Module, cfg):
    """AdamW with no weight decay on norms and biases — the usual recipe for ViT-family backbones."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.train.lr)


def lr_at(step: int, steps_per_epoch: int, cfg) -> float:
    """Linear warmup then cosine decay, evaluated per optimizer step."""
    warmup_steps = max(1, cfg.train.warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, cfg.train.epochs * steps_per_epoch)
    if step < warmup_steps:
        return cfg.train.lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return cfg.train.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))



def mix_batch(images: torch.Tensor, targets: torch.Tensor, cfg):
    """Blend the batch with a shuffled copy of itself — mixup or cutmix.

    With 647 training samples the model memorises the set long before it stops improving, and
    masking alone (SpecAugment) is a weak brake. Mixing pairs forces it to keep predicting
    sensibly on inputs that sit between two classes, which is the regularizer small datasets
    respond to most.

    Returns `(images, target_a, target_b, lam)`; the caller weights the two losses by `lam`.
    """
    use_mixup = cfg.train.mixup_alpha > 0
    use_cutmix = cfg.train.cutmix_alpha > 0
    if not (use_mixup or use_cutmix) or np.random.rand() > cfg.train.mix_prob:
        return images, targets, targets, 1.0

    if use_mixup and use_cutmix:
        use_cutmix = np.random.rand() < 0.5
    alpha = cfg.train.cutmix_alpha if use_cutmix else cfg.train.mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)

    if use_cutmix:
        # Cut along time only. A spectrogram's rows are Doppler velocities: splicing two clips
        # in time is a signal that could exist, splicing them in velocity is not.
        width = images.size(3)
        cut = int(round(width * (1.0 - lam)))
        if cut > 0:
            x0 = np.random.randint(0, width - cut + 1)
            images = images.clone()
            images[:, :, :, x0:x0 + cut] = images[perm][:, :, :, x0:x0 + cut]
            lam = 1.0 - cut / width
    else:
        images = lam * images + (1.0 - lam) * images[perm]

    return images, targets, targets[perm], lam


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, cfg, epoch, global_step):
    model.train()
    steps_per_epoch = max(1, len(loader) // cfg.train.grad_accum)
    running, seen = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.train.epochs} [train]", leave=False)
    for i, (images, targets) in enumerate(bar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, ta, tb, lam = mix_batch(images, targets, cfg)

        with torch.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
            logits = model(images)
            loss = lam * criterion(logits, ta) + (1.0 - lam) * criterion(logits, tb)
            loss = loss / cfg.train.grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % cfg.train.grad_accum == 0:
            lr = lr_at(global_step, steps_per_epoch, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            bar.set_postfix(loss=f"{running / max(1, seen):.4f}", lr=f"{lr:.2e}")

        running += loss.item() * cfg.train.grad_accum * images.size(0)
        seen += images.size(0)

    return running / max(1, seen), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg) -> dict:
    model.eval()
    total_loss, seen = 0.0, 0
    all_preds, all_targets = [], []

    for images, targets in tqdm(loader, desc="        [val]  ", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "loss": total_loss / max(1, seen),
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def fit(model, train_dl, val_dl, cfg, device, classes, ckpt_path):
    from .model import save_checkpoint

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")

    best = {"acc": -1.0}
    best_epoch, global_step = -1, 0
    history = []

    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_dl, optimizer, scaler, criterion, device, cfg, epoch, global_step
        )
        metrics = evaluate(model, val_dl, criterion, device, cfg)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        history.append(metrics)

        log.info(
            "epoch %2d/%d | train_loss %.4f | val_loss %.4f | acc %.4f | macro_f1 %.4f | %.1fs",
            epoch + 1, cfg.train.epochs, train_loss, metrics["loss"],
            metrics["acc"], metrics["macro_f1"], time.time() - t0,
        )

        if metrics["acc"] > best["acc"]:
            best, best_epoch = metrics, epoch
            save_checkpoint(ckpt_path, model, cfg, classes, metrics)
        elif epoch - best_epoch >= cfg.train.early_stop_patience:
            log.info("no improvement for %d epochs — stopping early", cfg.train.early_stop_patience)
            break

    log.info("best: epoch %d, acc %.4f, macro_f1 %.4f", best_epoch + 1, best["acc"], best["macro_f1"])
    return best, history
