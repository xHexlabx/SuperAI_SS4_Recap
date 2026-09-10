"""timm backbone wrapper + checkpoint I/O."""

from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn

log = logging.getLogger("uwb_pose.model")

# Rough VRAM guidance for this machine (RTX 3080 Ti Mobile, 16 GB) with AMP on.
# Larger backbones still fit — lower train.batch_size and raise train.grad_accum to compensate.
SUGGESTED_BATCH = {224: 32, 384: 8, 512: 4}


def build_model(cfg, num_classes: int) -> nn.Module:
    model = timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=num_classes,
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    _check_input_size(model, cfg)
    return model


def _check_input_size(model: nn.Module, cfg) -> None:
    """Fixed-resolution backbones (MaxViT, ViT) silently misbehave on a mismatched input."""
    native = getattr(model, "default_cfg", {}).get("input_size")
    if not native:
        return
    want = cfg.encode.img_size
    if native[-1] != want:
        log.warning(
            "%s expects %dx%d but encode.img_size=%d — set encode.img_size=%d "
            "(suggested train.batch_size=%s) or pick a resolution-agnostic backbone",
            cfg.model.name, native[-1], native[-1], want, native[-1],
            SUGGESTED_BATCH.get(native[-1], "?"),
        )


def save_checkpoint(path: str | Path, model: nn.Module, cfg, classes: list[str], metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "classes": classes,
            "metrics": metrics,
        },
        path,
    )
    log.info("saved checkpoint -> %s (val_acc=%.4f)", path, metrics.get("acc", float("nan")))


def load_checkpoint(path: str | Path, device: torch.device):
    """Rebuild the model exactly as trained, from the config stored in the checkpoint."""
    from .config import Config, _build, DataConfig, EncodeConfig, ModelConfig, TrainConfig, PredictConfig

    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt["config"]
    # Lenient: a checkpoint written by an older version may carry options this one dropped.
    cfg = Config(
        data=_build(DataConfig, raw.get("data"), strict=False),
        encode=_build(EncodeConfig, raw.get("encode"), strict=False),
        model=_build(ModelConfig, raw.get("model"), strict=False),
        train=_build(TrainConfig, raw.get("train"), strict=False),
        predict=_build(PredictConfig, raw.get("predict"), strict=False),
    )
    classes = ckpt["classes"]
    model = timm.create_model(cfg.model.name, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, classes
