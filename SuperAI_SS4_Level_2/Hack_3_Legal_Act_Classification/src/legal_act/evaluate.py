"""Score compiled rules on train and write the rows that are still wrong."""

from __future__ import annotations

import logging

import pandas as pd

from .data import load_committees, load_split
from .engine import load_rules, score
from .utils import ensure_dir

log = logging.getLogger("legal_act.evaluate")


def run(cfg) -> dict:
    train = load_split(cfg, "train")
    committees = load_committees(cfg)
    rules = load_rules(cfg.resolve(cfg.evaluate.rules))

    stats = score(train, rules, committees, semantics=cfg.evaluate.semantics)
    log.info("train macro-F1 : %.4f   <-- the metric the competition scores", stats["macro_f1"])
    log.info("train accuracy : %.4f  (%d rows)", stats["accuracy"], stats["n"])
    log.info("clauses solved : %d/%d", stats["solved_clauses"], stats["total_clauses"])
    if stats["missing_rules"]:
        log.warning("clauses with no usable rule: %d", stats["missing_rules"])

    worst = stats["per_clause"][stats["per_clause"]["acc"] < 1.0]
    if len(worst):
        log.info("worst clauses (acc, rows):")
        for cid, row in worst.head(15).iterrows():
            ctx = train.loc[train["clause_id"] == cid, "context"].iloc[0]
            log.info("  %.2f  %3d rows  %s  %s", row["acc"], int(row["rows"]), cid, ctx[:90])

    out = cfg.resolve(cfg.evaluate.report)
    ensure_dir(out.parent)
    wrong = stats["wrong"].copy()
    wrong["signers"] = wrong["signers"].map(lambda v: " + ".join(v))
    wrong.to_csv(out, index=False)
    log.info("wrote %d wrong rows -> %s", len(wrong), out)
    return stats
