"""Leave-subjects-out cross-validation, then an optional full fit for the submission.

The ten test subjects are people the model has never seen, so the only validation that
means anything is grouping by subject: `cv.n_folds` folds over whole nights, never
splitting one person across the split. Scores are reported both before and after Viterbi
smoothing, and the transition matrix of each fold is estimated on that fold's training
nights only.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold

from . import CLASSES
from .config import Config
from .context import build
from .data import load_dataset
from .models import build_model, needs_context
from .calibrate import apply_bias, fit_bias
from .smooth import class_prior, decode, transition_matrix
from .utils import ensure_dir, make_run_name, seed_everything

log = logging.getLogger("sleep_stage.train")


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    per = f1_score(y, pred, average=None, labels=range(len(CLASSES)), zero_division=0)
    return dict(macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
                acc=float(accuracy_score(y, pred)),
                kappa=float(cohen_kappa_score(y, pred)),
                per_class={c: float(v) for c, v in zip(CLASSES, per)})


def run(cfg: Config) -> dict:
    seed_everything(cfg.train.seed)
    ds = load_dataset(cfg)
    ctx = needs_context(cfg)
    tr_mats, te_mats, names = build(ds, cfg.context, with_context=ctx)
    ys = [n.y.astype(np.int64) for n in ds.train]

    run_name = make_run_name(cfg.train.run_name, cfg.model.name)
    out = ensure_dir(cfg.resolve(cfg.train.out_dir) / run_name)
    log.info("run %s | model=%s | %d nights, %d folds, %d features",
             run_name, cfg.model.name, len(tr_mats), cfg.cv.n_folds, len(names))

    oof = [np.zeros((len(M), len(CLASSES))) for M in tr_mats]
    oof_pred = [np.zeros(len(M), dtype=np.int8) for M in tr_mats]
    te_folds = [np.zeros((len(M), len(CLASSES))) for M in te_mats]
    folds = []
    t0 = time.time()
    for k, (tr, va) in enumerate(KFold(cfg.cv.n_folds, shuffle=True, random_state=cfg.train.seed)
                                 .split(np.arange(len(tr_mats)))):
        model = build_model(cfg, seed=cfg.train.seed + 100 * k)
        model.fit([tr_mats[i] for i in tr], [ys[i] for i in tr])
        probs = model.predict_proba([tr_mats[i] for i in va])
        A = transition_matrix([ys[i] for i in tr])
        prior = class_prior([ys[i] for i in tr])
        smoothed = decode(probs, A, prior, cfg.smooth)
        for j, i in enumerate(va):
            oof[i] = probs[j]
            oof_pred[i] = smoothed[j]
        for j, p in enumerate(model.predict_proba(te_mats)):
            te_folds[j] += p / cfg.cv.n_folds

        yv = np.concatenate([ys[i] for i in va])
        raw = score(yv, np.concatenate([p.argmax(1) for p in probs]))
        sm = score(yv, np.concatenate(smoothed))
        folds.append(dict(fold=k, n_nights=len(va), raw=raw, smoothed=sm))
        log.info("fold %d (%2d nights): macro-F1 %.4f -> %.4f after Viterbi | acc %.3f | %s  (%.0fs)",
                 k, len(va), raw["macro_f1"], sm["macro_f1"], sm["acc"],
                 {c: round(v, 3) for c, v in sm["per_class"].items()}, time.time() - t0)

    y_all = np.concatenate(ys)
    res = dict(run=run_name, config=cfg.to_dict(), n_features=len(names), folds=folds,
               oof=dict(raw=score(y_all, np.concatenate([p.argmax(1) for p in oof])),
                        smoothed=score(y_all, np.concatenate(oof_pred))))
    log.info("OOF: macro-F1 %.4f raw -> %.4f smoothed | weighted-F1 %.4f | acc %.4f | kappa %.3f",
             res["oof"]["raw"]["macro_f1"], res["oof"]["smoothed"]["macro_f1"],
             res["oof"]["smoothed"]["weighted_f1"], res["oof"]["smoothed"]["acc"],
             res["oof"]["smoothed"]["kappa"])
    log.info("OOF per class: %s", {c: round(v, 3) for c, v in res["oof"]["smoothed"]["per_class"].items()})

    A_all, prior_all = transition_matrix(ys), class_prior(ys)
    bias = np.zeros(len(CLASSES))
    if cfg.smooth.calibrate:
        bias, tuned = fit_bias(oof, ys, A_all, prior_all, cfg.smooth)
        res["oof"]["calibrated"] = score(y_all, np.concatenate(
            decode(apply_bias(oof, bias), A_all, prior_all, cfg.smooth)))
        log.info("OOF calibrated: macro-F1 %.4f | per class %s", res["oof"]["calibrated"]["macro_f1"],
                 {c: round(v, 3) for c, v in res["oof"]["calibrated"]["per_class"].items()})
    res["class_bias"] = {c: float(v) for c, v in zip(CLASSES, bias)}

    np.save(out / "class_bias.npy", bias)
    np.save(out / "oof_probs.npy", np.concatenate(oof))
    np.save(out / "test_probs_folds.npy", np.concatenate(te_folds))
    np.savez(out / "hmm.npz", A=A_all, prior=prior_all)

    if cfg.cv.full_fit:
        model = build_model(cfg, seed=cfg.train.seed)
        model.fit(tr_mats, ys)
        np.save(out / "test_probs_full.npy", np.concatenate(model.predict_proba(te_mats)))
        if hasattr(model, "feature_importance"):
            imp = sorted(zip(names, model.feature_importance), key=lambda t: -t[1])[:60]
            (out / "importance.csv").write_text("feature,gain\n" + "\n".join(f"{n},{v}" for n, v in imp))
        log.info("full fit done -> test_probs_full.npy")

    np.save(out / "test_lengths.npy", np.array([len(M) for M in te_mats]))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    return res
