"""Apply the compiled rules to test and write the submission CSV."""

from __future__ import annotations

import logging

import pandas as pd

from .data import load_committees, load_split
from .engine import load_rules, predict_rows
from .utils import ensure_dir

log = logging.getLogger("legal_act.predict")


def run(cfg) -> "pd.DataFrame":
    test = load_split(cfg, "test")
    committees = load_committees(cfg)
    rules = load_rules(cfg.resolve(cfg.predict.rules))

    missing = sorted({cid for cid in test["clause_id"].unique()
                      if cid not in rules or not rules[cid].ok})
    if missing:
        log.warning("%d test clauses have no usable rule — falling back to answer=%d for %d rows",
                    len(missing), cfg.predict.fallback,
                    int(test["clause_id"].isin(missing).sum()))

    pred = predict_rows(test, rules, committees, fallback=cfg.predict.fallback)
    sub = pd.DataFrame({"id": test["id"], "answer": pred.astype(int)})

    out = cfg.resolve(cfg.predict.out_csv)
    ensure_dir(out.parent)
    sub.to_csv(out, index=False)
    log.info("wrote %d rows -> %s  (answer=1 on %.1f%%)",
             len(sub), out, 100.0 * sub["answer"].mean())
    return sub
