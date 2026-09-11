"""End-to-end check on a tiny slice — รันก่อน commit ทุกครั้ง ใช้เวลาไม่ถึงนาที

    uv run python scripts/smoke_test.py

ใช้ clip-b32 (โมเดลเล็กสุด) เดินทุกขั้นตอนจริง: index -> embed -> proxy fold -> match ->
submission เพื่อจับ bug ประเภทไฟล์หาย/shape เพี้ยน/config พัง โดยไม่ต้องรอ benchmark เต็ม
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_search import benchmark as benchmark_mod  # noqa: E402
from img_search import data as data_mod  # noqa: E402
from img_search import gallery as gallery_mod  # noqa: E402
from img_search import match as match_mod  # noqa: E402
from img_search import predict as predict_mod  # noqa: E402
from img_search import proxy as proxy_mod  # noqa: E402
from img_search.config import load_config  # noqa: E402
from img_search.encoders import ENCODERS, make_view  # noqa: E402
from img_search.utils import setup_logging  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def main() -> int:
    setup_logging()
    cfg = load_config()
    cfg.embed.models = ["clip-b32"]
    cfg.benchmark.folds = 1

    print("\n[1] dataset layout")
    q, tr, te = data_mod.queries_index(cfg), data_mod.train_index(cfg), data_mod.test_index(cfg)
    check("queries = one image per class", len(q) == cfg.data.n_classes, f"{len(q)}")
    check("train folders present", tr["folder"].nunique() > cfg.data.n_classes,
          f"{tr['folder'].nunique()} folders / {len(tr)} images")
    check("test matches sample_submission", len(te) == len(data_mod.sample_submission(cfg)),
          f"{len(te)}")

    print("\n[2] preprocessing views")
    from PIL import Image

    with Image.open(q.iloc[0]["path"]) as im:
        for view in ("pad", "squash", "crop"):
            out = make_view(im, view, 224, cfg.embed.pad_color)
            check(f"view {view} -> 224x224", out.size == (224, 224), str(out.size))

    print("\n[3] encoder registry")
    check("registry non-empty", len(ENCODERS) > 0, f"{len(ENCODERS)} encoders")
    check("every spec has a known kind",
          all(e.kind in {"hf-clip", "hf-clip-pooler", "hf-siglip", "hf-dino", "open-clip"}
              for e in ENCODERS.values()))

    print("\n[4] proxy fold")
    fold = proxy_mod.build_folds(cfg, n_folds=1, seed=0, min_images=cfg.benchmark.min_images)[0]
    overlap = set(fold.gallery["path"]) & set(fold.val["path"])
    check("gallery and validation are disjoint", not overlap, f"{len(overlap)} shared")
    check("one reference image per class",
          (fold.gallery[fold.gallery["cls"] != cfg.data.unknown_class]
           .groupby("cls").size() == 1).all())
    check("validation covers both known and unknown",
          fold.val["cls"].nunique() > cfg.data.n_classes)

    print("\n[5] scoring + reject rules")
    scores, truth = benchmark_mod._fold_scores(cfg, fold, cfg.embed.models)
    check("score matrix shape", scores.known.shape == (len(truth), cfg.data.n_classes),
          str(scores.known.shape))
    check("negative gallery produced a score", scores.negative is not None)
    check("cosine scores stay in [-1, 1]", bool(np.abs(scores.known).max() <= 1.0 + 1e-4))
    for rule in ("gallery", "threshold", "gallery+threshold"):
        out = match_mod.decide(scores, reject=rule, threshold=0.8, ratio=1.02, neg_margin=0.0,
                               unknown_class=cfg.data.unknown_class)
        m = match_mod.evaluate(out["pred"], truth, n_classes=cfg.data.n_classes,
                               unknown_class=cfg.data.unknown_class)
        check(f"reject={rule} runs", 0.0 <= m["accuracy"] <= 1.0, f"acc={m['accuracy']:.3f}")

    print("\n[6] gallery + submission")
    gal = gallery_mod.build(cfg)
    check("production gallery has negatives",
          bool((gal.cls == cfg.data.unknown_class).any()), f"{len(gal.frame)} items")
    with tempfile.TemporaryDirectory() as tmp:
        cfg.predict.out_csv = str(Path(tmp) / "submission.csv")
        cfg.predict.debug_csv = str(Path(tmp) / "debug.csv")
        cfg.match.auto_threshold = False
        sub = predict_mod.run(cfg)
    check("submission has one row per test image", len(sub) == len(te), f"{len(sub)}")
    check("every label is a valid class",
          bool(sub[cfg.data.label_col].between(0, cfg.data.unknown_class).all()))

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
