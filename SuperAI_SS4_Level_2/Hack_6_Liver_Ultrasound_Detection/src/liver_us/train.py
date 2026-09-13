"""Train one ultralytics detector (YOLO or RT-DETR) from a Config and save run metadata."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from .config import Config
from .prepare import build_yolo_dataset
from .utils import seed_everything

log = logging.getLogger("liver_us.train")


def run_name(cfg: Config) -> str:
    if cfg.train.run_name:
        return cfg.train.run_name
    stem = Path(cfg.model.weights).stem
    tag = f"{stem}_{cfg.model.imgsz}"
    if cfg.data.mobile_oversample > 1:
        tag += f"_os{cfg.data.mobile_oversample}"
    if cfg.data.neg_fraction < 1:
        tag += f"_neg{cfg.data.neg_fraction:g}"
    if cfg.data.train_on_val:
        tag += "_tv"
    if cfg.data.synth_dir:
        tag += "_" + Path(cfg.data.synth_dir).name
    if cfg.train.seed != 42:
        tag += f"_s{cfg.train.seed}"
    return tag


def run(cfg: Config) -> Path:
    from ultralytics import RTDETR, YOLO

    seed_everything(cfg.train.seed)
    data_yaml = build_yolo_dataset(cfg)
    name = run_name(cfg)
    out_dir = cfg.resolve(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = (RTDETR if cfg.model.family == "rtdetr" else YOLO)(cfg.model.weights)
    t = cfg.train
    batch = int(t.batch) if float(t.batch) >= 1 else float(t.batch)
    kwargs = dict(
        data=str(data_yaml), imgsz=cfg.model.imgsz, epochs=t.epochs, patience=t.patience,
        batch=batch, seed=t.seed, optimizer=t.optimizer, lr0=t.lr0, cos_lr=t.cos_lr,
        close_mosaic=t.close_mosaic, workers=t.workers, mosaic=t.mosaic, mixup=t.mixup,
        degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear,
        perspective=t.perspective, fliplr=t.fliplr, flipud=t.flipud, hsv_h=t.hsv_h,
        hsv_s=t.hsv_s, hsv_v=t.hsv_v, project=str(out_dir), name=name, exist_ok=True,
        deterministic=False, plots=True, verbose=True,
    )
    kwargs.update(t.extra)
    log.info("training %s -> %s", cfg.model.weights, out_dir / name)
    model.train(**kwargs)
    run_dir = out_dir / name
    (run_dir / "liver_us_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    (run_dir / "liver_us_meta.json").write_text(json.dumps(dict(data_yaml=str(data_yaml), name=name)))
    log.info("done: %s", run_dir)
    return run_dir
