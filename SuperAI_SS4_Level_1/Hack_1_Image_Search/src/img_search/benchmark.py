"""Rank encoders on the pseudo-tasks that `img_search.proxy` carves out of `train/`.

threshold ถูกเลือกแบบ cross-fold — จูนบน fold อื่นแล้วเอามาใช้กับ fold นี้ ตัวเลขที่รายงานจึงเป็น
สิ่งที่คาดว่าจะได้ตอนเจอข้อมูลใหม่จริง ๆ ไม่ใช่ค่าที่ดีที่สุดเมื่อรู้คำตอบแล้ว
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import match as match_mod
from . import proxy as proxy_mod
from .embed import get_matrix
from .utils import ensure_dir

log = logging.getLogger("img_search.benchmark")


def _objective(metrics: dict[str, float], mode: str) -> float:
    if mode == "accuracy":
        return metrics["accuracy"]
    if mode == "balanced":
        return 0.5 * (metrics["known_acc"] + metrics["unknown_recall"])
    raise ValueError(f"unknown benchmark.objective {mode!r}")


def _reject_auroc(best_known: np.ndarray, truth: np.ndarray, unknown_class: int) -> float:
    """How well the top-1 score alone separates 'is one of the 22' from 'is not'."""
    pos = best_known[truth != unknown_class]
    neg = best_known[truth == unknown_class]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _fold_scores(cfg, fold, keys: list[str]) -> tuple[match_mod.Scores, np.ndarray, np.ndarray]:
    g_emb = get_matrix(cfg, fold.gallery["path"].tolist(), models=keys)
    v_emb = get_matrix(cfg, fold.val["path"].tolist(), models=keys)
    scores = score_pool(cfg, v_emb, g_emb, fold.gallery["cls"].to_numpy())
    return scores, fold.val["cls"].to_numpy(), v_emb


def score_pool(cfg, pool_emb, gallery_emb, gallery_cls) -> match_mod.Scores:
    """Class scores for a pool of images, with optional query expansion first.

    ใช้ร่วมกันระหว่าง benchmark (pool = validation) กับ predict (pool = test) เพื่อให้สอง
    เส้นทางคิดคะแนนด้วยวิธีเดียวกันเป๊ะ ๆ
    """
    k = expansion_k(cfg, len(pool_emb))
    if k > 0:
        gallery_emb, gallery_cls = match_mod.expand_gallery(
            pool_emb, gallery_emb, gallery_cls, k=k,
            n_classes=cfg.data.n_classes, unknown_class=cfg.data.unknown_class,
            aggregate=cfg.gallery.aggregate, topk=cfg.gallery.topk,
            reject=_expansion_reject(cfg),
            threshold=cfg.match.threshold, ratio=cfg.match.ratio,
            neg_margin=cfg.match.neg_margin, neg_quantile=cfg.match.neg_quantile,
            max_sim=cfg.gallery.expand_max_sim,
            expand_negatives=cfg.gallery.expand_negatives,
        )
    scores = match_mod.class_scores(
        pool_emb, gallery_emb, gallery_cls, n_classes=cfg.data.n_classes,
        unknown_class=cfg.data.unknown_class, aggregate=cfg.gallery.aggregate,
        topk=cfg.gallery.topk, neg_quantile=cfg.match.neg_quantile,
    )
    return match_mod.normalise(scores, cfg.match.score_norm)


def _expansion_reject(cfg) -> str:
    """Which reject rule gates what query expansion is allowed to add."""
    mode = cfg.gallery.expand_reject
    if mode != "auto":
        return mode
    # auto: ใช้กฎ gallery ล้วนถ้ามี เพราะยังไม่รู้ threshold ในรอบนี้
    return "gallery" if "gallery" in cfg.match.reject else cfg.match.reject


def expansion_k(cfg, pool_size: int) -> int:
    """How many images per class query expansion may fold back in."""
    k = cfg.gallery.expand_k
    if cfg.gallery.expand_frac > 0:
        frac_k = int(round(cfg.gallery.expand_frac * pool_size / max(cfg.data.n_classes, 1)))
        k = min(k, frac_k) if k > 0 else frac_k
    return max(int(k), 0)


def _threshold_grid(all_scores, n: int = 160) -> np.ndarray:
    best = np.concatenate([s.known.max(axis=1) for s in all_scores])
    return np.linspace(float(np.percentile(best, 0.5)), float(np.percentile(best, 99.5)), n)


def resolve_threshold(cfg, scores, base: float, pool_emb=None):
    """Turn the single tuned threshold into a per-class vector when that is switched on."""
    if cfg.match.per_class_gap <= 0:
        return base
    return match_mod.per_class_thresholds(
        scores.known, base, min_gap=cfg.match.per_class_gap,
        min_members=cfg.match.per_class_min_members, hi=cfg.match.per_class_hi,
        pool_emb=pool_emb, min_coherence=cfg.match.per_class_coherence,
    )


def _score_at(cfg, scores, truth, thr: float, pool_emb=None) -> dict[str, float]:
    out = match_mod.decide(scores, reject=cfg.match.reject,
                           threshold=resolve_threshold(cfg, scores, thr, pool_emb),
                           ratio=cfg.match.ratio,
                           neg_margin=cfg.match.neg_margin, unknown_class=cfg.data.unknown_class)
    return match_mod.evaluate(out["pred"], truth, n_classes=cfg.data.n_classes,
                              unknown_class=cfg.data.unknown_class)


def evaluate_model(cfg, keys: list[str], folds) -> dict[str, float]:
    """Cross-fold metrics for one encoder (or one `a+b` ensemble)."""
    per_fold = [_fold_scores(cfg, f, keys) for f in folds]
    tunable = "threshold" in cfg.match.reject
    grid = _threshold_grid([s for s, _, _ in per_fold]) if tunable else np.array([0.0])
    obj = cfg.benchmark.objective

    # threshold ที่จะใช้จริงตอน predict: ดีที่สุดเมื่อรวมทุก fold
    curve = np.array([
        np.mean([_objective(_score_at(cfg, s, t, thr, e), obj) for s, t, e in per_fold])
        for thr in grid
    ])
    production_threshold = float(grid[int(curve.argmax())])

    rows = []
    for i, (scores, truth, emb) in enumerate(per_fold):
        if tunable and len(folds) > 1:
            others = [x for j, x in enumerate(per_fold) if j != i]
            held = np.array([
                np.mean([_objective(_score_at(cfg, s, t, thr, e), obj) for s, t, e in others])
                for thr in grid
            ])
            thr = float(grid[int(held.argmax())])
        else:
            thr = production_threshold
        m = _score_at(cfg, scores, truth, thr, emb)
        known = truth != cfg.data.unknown_class
        m["retrieval_acc"] = float((scores.known.argmax(axis=1)[known] == truth[known]).mean())
        m["reject_auroc"] = _reject_auroc(scores.known.max(axis=1), truth, cfg.data.unknown_class)
        m["threshold"] = thr
        rows.append(m)

    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    # ส่วนเบี่ยงเบนระหว่าง fold — ตัวเลขที่ต่างกันน้อยกว่านี้ไม่ควรถือว่าต่างกันจริง
    mean["accuracy_std"] = float(np.std([r["accuracy"] for r in rows]))
    mean["threshold_production"] = production_threshold
    mean["n"] = int(sum(r["n"] for r in rows))
    return mean


def run(cfg, models: list[str] | None = None) -> pd.DataFrame:
    folds = proxy_mod.build_folds(cfg, n_folds=cfg.benchmark.folds, seed=cfg.benchmark.seed,
                                  min_images=cfg.benchmark.min_images,
                                  seen_ratio=cfg.benchmark.seen_ratio,
                                  neg_gallery_frac=cfg.benchmark.neg_gallery_frac)
    rows = []
    for entry in models or cfg.benchmark.models:
        t0 = time.time()
        try:
            m = evaluate_model(cfg, entry.split("+"), folds)
        except Exception as exc:
            log.error("skipping %s: %s", entry, exc)
            continue
        rows.append({"model": entry, **m, "seconds": round(time.time() - t0, 1)})
        log.info("%-30s acc=%.4f  bal=%.4f  retrieval=%.4f  AUROC=%.4f  thr=%.3f",
                 entry, m["accuracy"], 0.5 * (m["known_acc"] + m["unknown_recall"]),
                 m["retrieval_acc"], m["reject_auroc"], m["threshold_production"])

    if not rows:
        raise RuntimeError("no encoder ran successfully")
    table = pd.DataFrame(rows)
    table["balanced"] = 0.5 * (table["known_acc"] + table["unknown_recall"])
    table = table.sort_values(cfg.benchmark.objective if cfg.benchmark.objective != "balanced"
                              else "balanced", ascending=False, ignore_index=True)

    out = cfg.resolve(cfg.benchmark.out_json)
    ensure_dir(out.parent)
    out.write_text(json.dumps({"config": cfg.to_dict(), "results": table.to_dict("records")},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out)

    cols = ["model", "accuracy", "accuracy_std", "balanced", "known_acc", "unknown_recall",
            "retrieval_acc", "reject_auroc", "macro_f1", "threshold_production", "seconds"]
    print("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table
