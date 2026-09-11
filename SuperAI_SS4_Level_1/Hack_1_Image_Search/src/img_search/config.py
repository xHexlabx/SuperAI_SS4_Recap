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
    kaggle_slug: str = "image-search"
    kaggle_kind: str = "competition"
    root: str = "datasets"
    queries_dir: str = "queries/queries"
    train_dir: str = "train/train"
    test_dir: str = "test/images"
    sample_submission: str = "sample_submission.csv"
    id_col: str = "img_file"
    label_col: str = "class"
    # 22 query logos (class 0..21) + "none of them" = class 22
    n_classes: int = 22
    unknown_class: int = 22


@dataclass
class EmbedConfig:
    # ชื่อย่อจาก encoders.ENCODERS — ตัวแรกคือตัวที่ใช้จริงตอน predict
    models: list[str] = field(default_factory=lambda: ["siglip2-b16"])
    # ถ่วงน้ำหนักตอน concat หลาย model (ต้องยาวเท่า models หรือปล่อยว่าง = 1.0 หมด)
    weights: list[float] = field(default_factory=list)
    # วิธี fit ภาพให้เข้า input ของ backbone:
    #   pad    = เติมขอบให้เป็นสี่เหลี่ยมจัตุรัสก่อน resize (ไม่ตัด ไม่บิด) ← ปลอดภัยสุดกับโลโก้
    #   squash = resize ตรง ๆ (บิดสัดส่วน)
    #   crop   = resize ด้านสั้นแล้ว center-crop (ค่า default ของ processor, ตัดขอบทิ้ง)
    views: list[str] = field(default_factory=lambda: ["pad"])
    pad_color: str = "edge"       # edge | white | black
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True
    cache_dir: str = "datasets/emb"
    # ฟีเจอร์ที่ดึงจากโมเดลตระกูล DINO: cls | cls+avg
    dino_feature: str = "cls+avg"


@dataclass
class GalleryConfig:
    # queries   = โลโก้ต้นฉบับ 22 รูป (คลาส 0..21)
    # train     = รูปจากโฟลเดอร์ที่ query_map.yaml บอกว่าเป็นแบรนด์เดียวกับ query (คลาส 0..21)
    # negatives = รูปจากโฟลเดอร์ที่เหลือใน train/ ใช้เป็นตัวแทนของคลาส 22
    sources: list[str] = field(default_factory=lambda: ["queries", "train", "negatives"])
    # ไฟล์ map class id -> ชื่อโฟลเดอร์ใน train/train (สร้างด้วย `make map`)
    query_map: str = "configs/query_map.yaml"
    # รวม embedding ของแต่ละคลาสเป็นคะแนนยังไง: proto (mean) | max (1-NN) | topk
    aggregate: str = "max"
    topk: int = 3
    # query expansion: เอาภาพที่ทายได้มั่นใจสุดคลาสละ k รูปกลับเข้า gallery แล้วให้คะแนนใหม่
    # (ใช้แค่ *ภาพ* ของ test/validation ไม่ได้ใช้ label)
    #   expand_frac > 0  ->  k = frac x (จำนวนภาพในกอง / n_classes)  ← ใช้ตัวนี้
    #   expand_k    > 0  ->  จำนวนคงที่ (เพดานเมื่อใช้ร่วมกับ frac)
    # ที่ตั้งเป็นสัดส่วนเพราะ proxy มี ~2,000 ภาพ แต่ test มี 1,120 — k คงที่จะแรงเกินไปบน test
    expand_frac: float = 0.33
    expand_k: int = 0
    # ขยายฝั่ง negative ด้วยไหม (เอาภาพที่มั่นใจว่าเป็นคลาส 22 กลับเข้า gallery)
    expand_negatives: bool = False
    # กฎที่ใช้คัดภาพเข้า gallery ตอน query expansion:
    #   none    = เอา top-k ของแต่ละคลาสจาก argmax ล้วน ๆ
    #   gallery = ต้องชนะ negative gallery ก่อน (เข้มกว่า แต่กันภาพดี ๆ ออกไปด้วย)
    #   auto    = ตามค่า match.reject
    expand_reject: str = "auto"
    # ตอน query expansion ข้ามภาพที่คล้ายของเดิมใน gallery เกินค่านี้ (1.0 = ไม่ข้าม)
    # บังคับให้โควตา k ถูกใช้กับ "เวอร์ชันใหม่ของโลโก้" ไม่ใช่สำเนาของภาพที่มีอยู่แล้ว
    expand_max_sim: float = 1.0


@dataclass
class MatchConfig:
    # จะตัดสิน class 22 ยังไง
    #   threshold = คะแนนสูงสุดต้องเกิน match.threshold
    #   ratio     = คะแนนที่ 1 ต้องชนะคะแนนที่ 2 เกิน match.ratio (Lowe's ratio test)
    #   gallery   = ใส่รูปจากโฟลเดอร์ train ที่ "ไม่ใช่" 22 คลาสนี้เป็น gallery ของคลาส 22 ไปเลย
    #   gallery+threshold = ทำทั้งสองอย่าง
    reject: str = "gallery+threshold"
    # true = จูน threshold บน validation ที่สร้างจาก train/ ทุกครั้งก่อน predict (แนะนำ)
    auto_threshold: bool = True
    threshold: float = 0.5
    ratio: float = 1.02
    # margin ที่คลาสจริงต้องชนะ negative gallery (ใช้เมื่อ reject มี gallery)
    neg_margin: float = 0.0
    # สถิติที่ใช้สรุปคะแนนของ negative gallery: 1.0 = max (เอนเอียงตามขนาด gallery)
    # ค่าต่ำกว่า 1 เป็น quantile ซึ่งไม่โตตามจำนวนรูปในฝั่งลบ
    neg_quantile: float = 1.0
    # ปรับคะแนนก่อนตัดสิน: none | zscore (หักค่าเฉลี่ย/ส่วนเบี่ยงเบนของแต่ละคลาสบน test set)
    score_norm: str = "none"


@dataclass
class BenchmarkConfig:
    models: list[str] = field(default_factory=lambda: ["clip-b32", "clip-l14"])
    # จำนวน pseudo-task ที่สุ่มขึ้นมาจาก train/ (ดู img_search.proxy)
    folds: int = 5
    seed: int = 42
    # โฟลเดอร์ต้องมีรูปอย่างน้อยเท่านี้ถึงจะถูกสุ่มมาเป็นคลาสปลอมได้
    min_images: int = 4
    # สัดส่วนโฟลเดอร์ฝั่งลบที่ให้ "เคยเห็น" (รูปบางส่วนอยู่ใน gallery) ที่เหลือคือแบรนด์ที่ไม่เคยเห็น
    seen_ratio: float = 0.85
    # สัดส่วนรูปในโฟลเดอร์ "เคยเห็น" ที่เข้า gallery — คุมขนาด negative gallery ให้ใกล้ของจริง
    # (production มี 2,376 รูป) ถ้าเล็กเกินไป threshold ที่จูนได้จะย้ายมาใช้กับ test ไม่ตรง
    neg_gallery_frac: float = 0.95
    # เกณฑ์ที่ใช้เลือก threshold: accuracy (ตาม Kaggle) | balanced (เฉลี่ย known-acc กับ unknown-recall)
    # balanced ทนต่อการที่สัดส่วนคลาส 22 ใน proxy ไม่เท่ากับใน test จริง
    objective: str = "balanced"
    out_json: str = "models/benchmark.json"


@dataclass
class PredictConfig:
    out_csv: str = "submissions/submission.csv"
    # เขียนคะแนนดิบ/คลาสที่ 2 ไว้ตรวจงานด้วย
    debug_csv: str | None = "models/predict_debug.csv"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
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
    """Build one config section; a typo in a YAML key is an error, not a silent no-op."""
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        if strict:
            raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
        logging.getLogger("img_search.config").warning(
            "%s: ignoring unknown config keys %s", cls.__name__, sorted(unknown)
        )
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, then apply `section.key=value` CLI overrides."""
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    cfg = Config(
        data=_build(DataConfig, raw.get("data")),
        embed=_build(EmbedConfig, raw.get("embed")),
        gallery=_build(GalleryConfig, raw.get("gallery")),
        match=_build(MatchConfig, raw.get("match")),
        benchmark=_build(BenchmarkConfig, raw.get("benchmark")),
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
    """Parse an override string with YAML rules, keeping ints/floats/bools/lists sane."""
    parsed = yaml.safe_load(value)
    if isinstance(like, bool):
        return bool(parsed)
    if isinstance(like, int) and not isinstance(like, bool) and parsed is not None:
        return int(parsed)
    if isinstance(like, float) and parsed is not None:
        return float(parsed)
    return parsed
