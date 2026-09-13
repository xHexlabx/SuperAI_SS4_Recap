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
    kaggle_slug: str = "signal-processing-sleep-staging-classification"
    root: str = "datasets"
    cache_dir: str = "datasets/cache"
    n_jobs: int = 8                    # subjects processed in parallel while caching features


@dataclass
class FeatureConfig:
    fs: int = 64                       # every column is resampled to 64 Hz by the organiser
    epoch_samples: int = 1920          # 30 s
    bvp_band: list[float] = field(default_factory=lambda: [0.5, 8.0])
    peak_min_distance_s: float = 0.33  # 180 bpm ceiling for the systolic peak search
    nn_range_s: list[float] = field(default_factory=lambda: [0.33, 2.0])
    nn_ectopic_tol: float = 0.25       # drop an NN more than 25 % off the local median
    hrv_window_s: float = 300.0        # NN context used for the frequency-domain block
    hrv_interp_hz: float = 4.0
    eda_scr_band: list[float] = field(default_factory=lambda: [0.05, 1.0])
    acc_still_g: float = 0.02          # per-second std below this counts the second as "still"


@dataclass
class ContextConfig:
    per_subject_norm: bool = True      # robust z-score every feature inside its own recording
    keep_absolute: bool = True         # ...and keep the un-normalised copy as well
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    roll_windows: list[int] = field(default_factory=lambda: [5, 15, 31, 61])
    clock: bool = True                 # position-in-the-night features


@dataclass
class CVConfig:
    n_folds: int = 5                   # GroupKFold by subject — test is ten unseen people
    full_fit: bool = True


@dataclass
class ModelConfig:
    name: str = "bilstm"               # lgbm | bilstm | tcn | gru
    # lightgbm
    n_estimators: int = 1200
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 60
    subsample: float = 0.8
    colsample_bytree: float = 0.5
    reg_lambda: float = 1.0
    class_weight: str = "balanced"     # balanced | none
    # sequence nets
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.3
    epochs: int = 60
    lr: float = 0.002
    weight_decay: float = 0.0001
    batch_size: int = 8                # whole nights per batch
    crop_epochs: int = 0               # >0: train on random crops of this many 30 s epochs
    label_smoothing: float = 0.05
    loss_class_weight: str = "balanced"
    n_seeds: int = 3
    avg_last: int = 5              # average the softmax of the last N training epochs (cheap SWA)
    device: str = "auto"               # auto | cpu | cuda — sequence nets only
    tcn_kernel: int = 7
    tcn_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])


@dataclass
class SmoothConfig:
    method: str = "hmm"                # none | hmm
    self_bias: float = 0.0             # added to the log of the diagonal before Viterbi
    prior_power: float = 1.0           # divide log-likelihoods by the class prior ^ this
    calibrate: bool = True             # fit one bias per class on OOF to maximise macro F1


@dataclass
class TrainConfig:
    out_dir: str = "models"
    run_name: str | None = None
    seed: int = 42


@dataclass
class PredictConfig:
    runs: list[str] = field(default_factory=list)   # run dirs to ensemble; empty = latest
    source: str = "full"               # full | folds | both
    out_csv: str = "submissions/submission.csv"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)
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
        logging.getLogger("sleep_stage.config").warning("%s: ignoring keys %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in raw.items() if k in known})


def from_dict(raw: dict[str, Any], strict: bool = True) -> Config:
    return Config(**{name: _build(cls, raw.get(name), strict) for name, cls in _SECTIONS.items()})


_SECTIONS = dict(data=DataConfig, features=FeatureConfig, context=ContextConfig, cv=CVConfig,
                 model=ModelConfig, smooth=SmoothConfig, train=TrainConfig, predict=PredictConfig)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
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
