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
    kaggle_slug: str = "brain-motor-imagery-classification"
    root: str = "datasets"
    cache_dir: str = "datasets/cache"


@dataclass
class PreprocessConfig:
    highpass_hz: float = 0.5
    notch_hz: list[float] = field(default_factory=lambda: [50.0, 100.0])
    bandpass_hz: list[float] = field(default_factory=lambda: [8.0, 30.0])
    filter_order: int = 4
    pad_seconds: float = 2.0          # reflect-padding before filtfilt on a 7 s trial
    resample_hz: int = 125
    window_s: list[float] = field(default_factory=lambda: [0.5, 3.5])   # after cue
    clip_sigma: float = 0.0           # >0: clip |x| at clip_sigma * robust std per channel (after alignment)
    channels: list[int] | None = None  # subset of the 8 EEG channels, None = all


@dataclass
class AlignConfig:
    method: str = "euclidean"         # none | euclidean | riemann
    train_group: str = "sess"         # sess | subject | block  (column of train meta)
    test_group: str = "cluster"       # counter | cluster | global
    n_clusters: int = 2               # for test_group=cluster
    cov_window: str = "full"          # full | window  — which part of the trial the reference covariance uses


@dataclass
class CVConfig:
    scheme: str = "subject"           # subject | block
    n_folds: int = 5
    eval_application: bool = True     # score fold models on train_application (unseen session)
    test_oracle: bool = True          # score fold models on test with cue-order labels (EDA leak) — reporting only
    full_fit: bool = True             # also fit on all train data and predict test
    include_application: bool = False # add train_application trials to the full fit


@dataclass
class ModelConfig:
    name: str = "ts_lr"               # ts_lr | fbts_lr | csp_lr | eegnet | shallow | conformer | atcnet
    hierarchical: bool = False        # rest-vs-MI then L-vs-R
    lr_C: float = 1.0
    bands: list[list[float]] = field(default_factory=lambda: [[4, 8], [8, 13], [13, 20], [20, 30]])
    csp_filters: int = 6
    # deep nets
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    drop_prob: float = 0.5
    label_smoothing: float = 0.1
    n_seeds: int = 1
    aug_shift: int = 12               # max random roll in samples
    aug_scale: float = 0.1
    aug_noise: float = 0.05
    aug_channel_dropout: float = 0.0  # probability of zeroing each channel
    aug_reflect: bool = False         # C3<->C4, PO7<->PO8 with label swap 110<->120
    aug_freq_shift_hz: float = 0.0    # random frequency shift (braindecode)
    mixup_alpha: float = 0.0


@dataclass
class TrainConfig:
    out_dir: str = "models"
    run_name: str | None = None
    seed: int = 42


@dataclass
class PredictConfig:
    runs: list[str] = field(default_factory=list)   # run dirs to ensemble; empty = latest
    source: str = "full"              # full | folds | both  — which test probabilities to use
    out_csv: str = "submissions/submission.csv"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    cv: CVConfig = field(default_factory=CVConfig)
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
        logging.getLogger("brain_mi.config").warning("%s: ignoring keys %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in raw.items() if k in known})


def from_dict(raw: dict[str, Any], strict: bool = True) -> Config:
    return Config(**{name: _build(cls, raw.get(name), strict) for name, cls in _SECTIONS.items()})


_SECTIONS = dict(data=DataConfig, preprocess=PreprocessConfig, align=AlignConfig, cv=CVConfig,
                 model=ModelConfig, train=TrainConfig, predict=PredictConfig)


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
