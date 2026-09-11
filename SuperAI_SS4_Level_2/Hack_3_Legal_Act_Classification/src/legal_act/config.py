"""Config loading: YAML -> nested dataclasses, with CLI overrides."""

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
    kaggle_slug: str = "legal-act-classification"
    kaggle_kind: str = "competition"
    root: str = "datasets"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    committee_csv: str = "committee.csv"
    patterns_csv: str = "patterns.csv"
    sample_submission_csv: str = "sample_submission.csv"


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-14B-AWQ"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 3072
    # Qwen3 มี thinking mode ในตัว — "off" ให้ตอบ JSON ตรง ๆ (เร็วกว่ามาก),
    # "on" ให้คิดก่อนตอบ, "default" = ไม่ส่ง flag (สำหรับ API ที่ไม่รู้จักคีย์นี้)
    thinking: str = "off"
    # ส่งฟิลด์เพิ่มเข้า request body ตรง ๆ สำหรับ backend ที่มีพารามิเตอร์เฉพาะตัว
    # เช่น OpenRouter ปิด reasoning ด้วย {"reasoning": {"enabled": false}}
    extra_body: dict = field(default_factory=dict)
    timeout: float = 300.0
    max_retries: int = 4
    concurrency: int = 8
    cache: bool = True
    cache_dir: str = "datasets/llm_cache"


@dataclass
class CompileConfig:
    splits: list[str] = field(default_factory=lambda: ["train", "test"])
    out: str = "models/rules.json"
    scope_stage: bool = True
    self_repair_rounds: int = 2
    # ซ่อมกฎที่ขัดกับจำนวนผู้ลงนามที่ถูกถาม — ไม่ใช้ label จึงใช้กับ test ได้
    consistency_rounds: int = 2
    # กติกาลายเซ็นที่ใช้ตอนวัดผลใน repair loop — ต้องตรงกับ predict.semantics
    semantics: str = "exact"
    snap_names: bool = True
    snap_cutoff: float = 0.75


@dataclass
class EvaluateConfig:
    rules: str = "models/rules.json"
    report: str = "models/error_report.csv"
    semantics: str = "exact"


@dataclass
class PredictConfig:
    rules: str = "models/rules.json"
    out_csv: str = "submissions/submission.csv"
    fallback: int = 0
    # exact = ลายเซ็นต้องครบพอดี | at_least = เซ็นเกินได้ (ดู decide() ใน rules.py)
    semantics: str = "exact"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def resolve(self, relative: str) -> Path:
        """Resolve a config path against the project root, so commands work from any cwd."""
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
    """Build one config section. Strict, so a typo in a YAML key is an error not a silent no-op."""
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        if strict:
            raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
        logging.getLogger("legal_act.config").warning(
            "%s: ignoring unknown config keys: %s", cls.__name__, sorted(unknown)
        )
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, then apply `section.key=value` CLI overrides."""
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    if not Path(path).is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    cfg = Config(
        data=_build(DataConfig, raw.get("data")),
        llm=_build(LLMConfig, raw.get("llm")),
        compile=_build(CompileConfig, raw.get("compile")),
        evaluate=_build(EvaluateConfig, raw.get("evaluate")),
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
    setattr(target, leaf, _coerce(value, getattr(target, leaf)))


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
