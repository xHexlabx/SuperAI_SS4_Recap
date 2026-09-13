"""Config loading: YAML -> nested dataclasses, with `--set a.b=value` overrides."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class DataConfig:
    kaggle_slug: str = "liver-ultrasound-detection"
    root: str = "datasets"
    yolo_dir: str = "datasets/yolo"     # symlink layout ultralytics understands
    mobile_oversample: int = 1          # how many times each mobile train image appears (1 = no oversampling)
    train_on_val: bool = False          # fold val into train (final fit)
    neg_fraction: float = 1.0           # keep this fraction of negative (no-lesion) train images
    synth_dir: str | None = None        # extra images/labels dir (simulated phone photos) added to train


@dataclass
class ModelConfig:
    family: str = "yolo"                # yolo | rtdetr
    weights: str = "yolo26m.pt"         # ultralytics checkpoint / model yaml
    imgsz: int = 1024


@dataclass
class TrainConfig:
    out_dir: str = "models"
    run_name: str | None = None
    epochs: int = 100
    patience: int = 30
    batch: float = 0.7                  # fraction of VRAM (ultralytics autobatch) or int
    seed: int = 42
    optimizer: str = "auto"
    lr0: float = 0.01
    cos_lr: bool = True
    close_mosaic: int = 15
    workers: int = 8
    # augmentation (ultralytics names)
    mosaic: float = 1.0
    mixup: float = 0.1
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0003
    fliplr: float = 0.5
    flipud: float = 0.0
    hsv_h: float = 0.01
    hsv_s: float = 0.3
    hsv_v: float = 0.4
    extra: dict = field(default_factory=dict)   # any other ultralytics train kwargs


@dataclass
class PredictConfig:
    runs: list[str] = field(default_factory=list)   # run dirs (models/<name>) to ensemble; empty = latest
    conf: float = 0.001
    iou: float = 0.6
    max_det: int = 100
    augment: bool = True                # ultralytics TTA (scales + flip)
    agnostic_nms: bool = True
    wbf_iou: float = 0.55
    out_csv: str = "submissions/submission.csv"
    threshold: float | None = None      # None = tune on val with the local metric
    max_boxes: int = 5                  # per-image cap in the submission
    tune_slice: str = "testlike"        # all | machine | mobile | testlike — which val slice to tune the threshold on
    class_thresholds: bool = True       # per-class threshold refinement after the global sweep


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _build(cls: type, raw: dict[str, Any] | None, strict: bool = True) -> Any:
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        if strict:
            raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
        logging.getLogger("liver_us.config").warning("%s: ignoring keys %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in raw.items() if k in known})


_SECTIONS = dict(data=DataConfig, model=ModelConfig, train=TrainConfig, predict=PredictConfig)


def from_dict(raw: dict[str, Any], strict: bool = True) -> Config:
    return Config(**{name: _build(cls, raw.get(name), strict) for name, cls in _SECTIONS.items()})


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = from_dict(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        _apply_override(cfg, key, value)
    return cfg


def _apply_override(cfg: Config, dotted: str, value: str) -> None:
    parts = dotted.split(".")
    target: Any = cfg
    for p in parts[:-1]:
        if not hasattr(target, p):
            raise ValueError(f"unknown config section {p!r} in {dotted!r}")
        target = getattr(target, p)
    leaf = parts[-1]
    if isinstance(target, dict):
        target[leaf] = yaml.safe_load(value)
        return
    if not hasattr(target, leaf):
        raise ValueError(f"unknown config key {leaf!r} in {dotted!r}")
    setattr(target, leaf, _coerce(value, getattr(target, leaf)))


def _coerce(value: str, like: Any) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(like, bool):
        return bool(parsed)
    if isinstance(like, int) and not isinstance(like, bool) and parsed is not None:
        return int(parsed)
    if isinstance(like, float) and parsed is not None:
        return float(parsed)
    return parsed
