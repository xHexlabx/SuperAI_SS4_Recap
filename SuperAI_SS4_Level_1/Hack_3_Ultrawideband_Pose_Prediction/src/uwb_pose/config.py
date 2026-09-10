"""Config loading: YAML -> nested dataclasses, with CLI overrides."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    kaggle_slug: str = ""
    kaggle_kind: str = "dataset"
    root: str = "datasets"
    # auto | wide_csv (one row per sample) | npy_dir (one .npy file per sample)
    layout: str = "auto"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    # npy_dir layout
    annotations_csv: str = "annotations.csv"
    classes_csv: str = "classes.csv"
    train_dir: str = "train/train"
    test_dir: str = "test/test"
    file_col: str | None = None
    npy_suffix: str = ".npy"
    label_col: str | None = "class"
    id_col: str | None = "id"
    signal_cols: list[str] | None = None
    n_channels: int = 1
    val_size: float = 0.2
    seed: int = 42


@dataclass
class EncodeConfig:
    method: str = "stack"
    stack_methods: list[str] = field(default_factory=lambda: ["rti", "range_doppler", "micro_doppler"])
    img_size: int = 224
    # radar encoders (rti / range_doppler / micro_doppler)
    n_frames: int = 2560
    range_bins: int = 56
    clutter_removal: bool = True
    dynamic_range: float = 40.0
    doppler_crop: float = 0.25
    md_nperseg: int = 256
    md_noverlap: int = 224
    rd_windows: int = 3
    range_bands: int = 3
    nperseg: int = 64
    noverlap: int = 48
    log_power: bool = True
    wavelet: str = "morl"
    n_scales: int = 64
    cache: bool = True
    cache_dir: str = "datasets/cache"


@dataclass
class ModelConfig:
    name: str = "maxvit_tiny_tf_224.in1k"
    pretrained: bool = True
    drop_rate: float = 0.1
    drop_path_rate: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    num_workers: int = 8
    amp: bool = True
    grad_accum: int = 1
    early_stop_patience: int = 5
    # mixup / cutmix — set an alpha > 0 to switch each on
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mix_prob: float = 0.5
    folds: int = 1
    # train on every labelled sample, with no validation split — see train.run()
    full_data: bool = False
    out_dir: str = "models"
    run_name: str | None = None


@dataclass
class PredictConfig:
    checkpoint: str = "models/best.pt"
    out_csv: str = "submissions/submission.csv"
    batch_size: int = 64


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def resolve(self, relative: str) -> Path:
        """Resolve a config path against the project root, so scripts work from any cwd."""
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
    """Build one config section.

    Strict for YAML files, so a typo in a key is an error rather than a silently ignored
    setting. Lenient for configs read back out of a checkpoint, where an option that has since
    been renamed or dropped should not make an old model unloadable.
    """
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        if strict:
            raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
        logging.getLogger("uwb_pose.config").warning(
            "%s: ignoring config keys not in this version: %s", cls.__name__, sorted(unknown)
        )
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, then apply `key.sub=value` CLI overrides."""
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    cfg = Config(
        data=_build(DataConfig, raw.get("data")),
        encode=_build(EncodeConfig, raw.get("encode")),
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
        predict=_build(PredictConfig, raw.get("predict")),
    )

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
    if not hasattr(target, leaf):
        raise ValueError(f"unknown config key {leaf!r} in {dotted!r}")
    current = getattr(target, leaf)
    setattr(target, leaf, _coerce(value, current))


def _coerce(value: str, like: Any) -> Any:
    """Parse an override string using YAML rules, keeping ints/floats/bools/lists sane."""
    parsed = yaml.safe_load(value)
    if isinstance(like, bool):
        return bool(parsed)
    if isinstance(like, int) and not isinstance(like, bool) and parsed is not None:
        return int(parsed)
    if isinstance(like, float) and parsed is not None:
        return float(parsed)
    return parsed
