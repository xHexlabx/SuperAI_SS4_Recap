"""Cross-validated training with three validation layers.

1. GroupKFold on train (by subject, or by block) -> out-of-fold score.
2. Fold models averaged on `train_application` (one unseen session, 180 labelled trials).
3. Fold models averaged on `test` scored against the cue-order labels found in the EDA.
   This is a leak-derived *oracle used for reporting only*; predictions never touch it.

Then an optional full fit on every train trial (+ application) producing the test
probabilities that `mi-predict` turns into a submission.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from .config import Config
from .data import CLASSES
from .models import build_model, proba_in_class_order
from .preprocess import prepare
from .utils import ensure_dir, make_run_name, seed_everything

log = logging.getLogger("brain_mi.train")


def _score(y, p):
    pred = np.array(CLASSES)[p.argmax(1)]
    return dict(acc=float(accuracy_score(y, pred)), f1=float(f1_score(y, pred, average="macro")))


def run(cfg: Config) -> dict:
    seed_everything(cfg.train.seed)
    P = prepare(cfg)
    groups = P.Mtr["subject" if cfg.cv.scheme == "subject" else "block"].values
    run_name = make_run_name(cfg.train.run_name, cfg.model.name)
    out = ensure_dir(cfg.resolve(cfg.train.out_dir) / run_name)
    log.info("run %s | model=%s hierarchical=%s | cv=%s x%d", run_name, cfg.model.name,
             cfg.model.hierarchical, cfg.cv.scheme, cfg.cv.n_folds)

    oof = np.zeros((len(P.Xtr), 3)); app_p = np.zeros((len(P.Xap), 3)); te_p = np.zeros((len(P.Xte), 3))
    folds = []
    t0 = time.time()
    for k, (tr, va) in enumerate(GroupKFold(cfg.cv.n_folds).split(P.Xtr, P.ytr, groups)):
        model = build_model(cfg, P.fs, seed=cfg.train.seed + 100 * k)
        model.fit(P.Xtr[tr], P.ytr[tr])
        oof[va] = proba_in_class_order(model, P.Xtr[va])
        r = _score(P.ytr[va], oof[va]); r["n_val"] = int(len(va))
        if cfg.cv.eval_application:
            pa = proba_in_class_order(model, P.Xap); app_p += pa / cfg.cv.n_folds; r["app_acc"] = _score(P.yap, pa)["acc"]
        if cfg.cv.test_oracle:
            pt = proba_in_class_order(model, P.Xte); te_p += pt / cfg.cv.n_folds; r["oracle_acc"] = _score(P.yte_oracle, pt)["acc"]
        folds.append(r)
        log.info("fold %d: val acc %.3f f1 %.3f | app %.3f | test-oracle %.3f  (%.0fs)", k, r["acc"], r["f1"],
                 r.get("app_acc", np.nan), r.get("oracle_acc", np.nan), time.time() - t0)

    res = dict(run=run_name, config=cfg.to_dict(), folds=folds,
               cv=dict(acc=float(np.mean([f["acc"] for f in folds])), acc_std=float(np.std([f["acc"] for f in folds])),
                       f1=float(np.mean([f["f1"] for f in folds])), oof=_score(P.ytr, oof)))
    if cfg.cv.eval_application:
        res["application"] = _score(P.yap, app_p)
    if cfg.cv.test_oracle:
        res["test_oracle_foldavg"] = _score(P.yte_oracle, te_p)
    np.save(out / "oof_probs.npy", oof); np.save(out / "app_probs.npy", app_p); np.save(out / "test_probs_folds.npy", te_p)

    if cfg.cv.full_fit:
        X, y = P.Xtr, P.ytr
        if cfg.cv.include_application:
            X, y = np.concatenate([X, P.Xap]), np.concatenate([y, P.yap])
        model = build_model(cfg, P.fs, seed=cfg.train.seed + 999)
        model.fit(X, y)
        pf = proba_in_class_order(model, P.Xte)
        np.save(out / "test_probs_full.npy", pf)
        if cfg.cv.test_oracle:
            res["test_oracle_full"] = _score(P.yte_oracle, pf)
        res["full_fit_n"] = int(len(y))

    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    log.info("== %s | CV acc %.3f±%.3f f1 %.3f | app acc %.3f | test-oracle foldavg %.3f full %.3f | %.0fs",
             run_name, res["cv"]["acc"], res["cv"]["acc_std"], res["cv"]["f1"],
             res.get("application", {}).get("acc", np.nan), res.get("test_oracle_foldavg", {}).get("acc", np.nan),
             res.get("test_oracle_full", {}).get("acc", np.nan), time.time() - t0)
    return res
