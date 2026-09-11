"""Score the 1120 test images and write submissions/submission.csv."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import data as data_mod
from . import gallery as gallery_mod
from . import match as match_mod
from .embed import get_matrix
from .utils import ensure_dir

log = logging.getLogger("img_search.predict")


def tune_threshold(cfg) -> tuple[float, dict[str, float]]:
    """Calibrate the reject threshold on the pseudo-tasks built out of `train/`.

    โครงของ proxy เหมือนของจริง (โลโก้อ้างอิงคลาสละ 1 รูป + negative gallery) คะแนน cosine
    จึงอยู่ในสเกลเดียวกัน threshold ที่จูนได้เลยย้ายมาใช้กับ test ได้ตรง ๆ
    """
    from . import benchmark as benchmark_mod
    from . import proxy as proxy_mod

    folds = proxy_mod.build_folds(cfg, n_folds=cfg.benchmark.folds, seed=cfg.benchmark.seed,
                                  min_images=cfg.benchmark.min_images,
                                  seen_ratio=cfg.benchmark.seen_ratio,
                                  neg_gallery_frac=cfg.benchmark.neg_gallery_frac)
    metrics = benchmark_mod.evaluate_model(cfg, cfg.embed.models, folds)
    thr = metrics["threshold_production"]
    log.info("calibration over %d proxy image(s): acc=%.4f (known %.4f / unknown %.4f) "
             "-> threshold=%.4f", metrics["n"], metrics["accuracy"], metrics["known_acc"],
             metrics["unknown_recall"], thr)
    return thr, metrics


def run(cfg) -> pd.DataFrame:
    threshold = cfg.match.threshold
    if cfg.match.auto_threshold and "threshold" in cfg.match.reject:
        threshold, _ = tune_threshold(cfg)

    gal = gallery_mod.build(cfg)                       # gallery เต็ม: queries + train ทั้งหมด
    test = data_mod.test_index(cfg)
    log.info("encoding %d test images with %s", len(test), "+".join(cfg.embed.models))

    from .benchmark import score_pool

    scores = score_pool(cfg, get_matrix(cfg, test["path"].tolist()),
                        get_matrix(cfg, gal.paths), gal.cls)
    out = match_mod.decide(scores, reject=cfg.match.reject, threshold=threshold,
                           ratio=cfg.match.ratio, neg_margin=cfg.match.neg_margin,
                           unknown_class=cfg.data.unknown_class)

    id_col, label_col = cfg.data.id_col, cfg.data.label_col
    submission = pd.DataFrame({id_col: test[id_col], label_col: out["pred"]})

    dest = cfg.resolve(cfg.predict.out_csv)
    ensure_dir(dest.parent)
    submission.to_csv(dest, index=False)
    log.info("wrote %s", dest)

    if cfg.predict.debug_csv:
        debug = submission.assign(
            top1=out["top1"], top1_score=out["top1_score"],
            top2=out["top2"], top2_score=out["top2_score"],
            neg_score=out["neg_score"], rejected=out["rejected"],
        )
        dbg = cfg.resolve(cfg.predict.debug_csv)
        ensure_dir(dbg.parent)
        debug.to_csv(dbg, index=False)
        log.info("wrote %s", dbg)

    counts = submission[label_col].value_counts().sort_index()
    unknown = cfg.data.unknown_class
    log.info("predicted class 22 for %d/%d images (%.1f%%)",
             int(counts.get(unknown, 0)), len(submission),
             100 * float(counts.get(unknown, 0)) / len(submission))
    print(counts.to_string())
    return submission


def distribution_summary(submission: pd.DataFrame, label_col: str) -> np.ndarray:
    return submission[label_col].value_counts().sort_index().to_numpy()
