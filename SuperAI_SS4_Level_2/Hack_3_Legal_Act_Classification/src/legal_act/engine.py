"""Apply compiled rules to rows, and score them against the train labels.

This is the deterministic half of the pipeline: no model runs here, so a change in accuracy
always points at a clause the compiler read wrongly, never at sampling noise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .data import Committee
from .rules import ClauseRule, decide
from .utils import ensure_dir, normalise_name

log = logging.getLogger("legal_act.engine")


def director_keys(committees: dict[str, Committee], rg: str) -> set[str]:
    c = committees.get(rg)
    return set(c.by_key) if c else set()


def predict_rows(df: pd.DataFrame, rules: dict[str, ClauseRule],
                 committees: dict[str, Committee], fallback: int = 0,
                 semantics: str = "exact") -> pd.Series:
    preds = []
    for cid, rg, signers, act in zip(df["clause_id"], df["rg"], df["signers"], df["legal_act"]):
        rule = rules.get(cid)
        if rule is None or not rule.ok:
            preds.append(fallback)
            continue
        preds.append(int(decide(rule, signers, act, director_keys(committees, rg), semantics)))
    return pd.Series(preds, index=df.index, name="answer")


def score(df: pd.DataFrame, rules: dict[str, ClauseRule],
          committees: dict[str, Committee], fallback: int = 0,
          semantics: str = "exact") -> dict:
    """Accuracy overall and per clause, plus the rows that are still wrong."""
    pred = predict_rows(df, rules, committees, fallback, semantics)
    truth = df["answer"].astype(int)
    hit = pred == truth
    per_clause = (
        pd.DataFrame({"clause_id": df["clause_id"], "hit": hit})
        .groupby("clause_id")["hit"].agg(["mean", "size"])
        .rename(columns={"mean": "acc", "size": "rows"})
        .sort_values(["acc", "rows"], ascending=[True, False])
    )
    wrong = df.loc[~hit, ["id", "clause_id", "context", "legal_act", "signers"]].copy()
    wrong["truth"] = truth[~hit]
    wrong["pred"] = pred[~hit]
    return {
        "accuracy": float(hit.mean()),
        "macro_f1": macro_f1(truth, pred),
        "n": int(len(df)),
        "per_clause": per_clause,
        "wrong": wrong,
        "solved_clauses": int((per_clause["acc"] == 1.0).sum()),
        "total_clauses": int(len(per_clause)),
        "missing_rules": int(sum(1 for c in df["clause_id"].unique()
                                 if c not in rules or not rules[c].ok)),
    }


def macro_f1(truth: pd.Series, pred: pd.Series) -> float:
    """The competition scores macro-F1, not accuracy.

    Two probe submissions pin it down: an all-zero file scores 0.29597 and an all-one file
    0.36692. Those sum to 0.663, so the metric cannot be accuracy; solving p/(1+p) = 0.36692
    gives a positive rate of 0.5796, and (1-p)/(2-p) then reproduces 0.29597 exactly.
    """
    out = []
    for cls in (0, 1):
        tp = int(((pred == cls) & (truth == cls)).sum())
        fp = int(((pred == cls) & (truth != cls)).sum())
        fn = int(((pred != cls) & (truth == cls)).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(sum(out) / 2)


def save_rules(rules: dict[str, ClauseRule], path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(
        json.dumps({"rules": [r.to_dict() for r in rules.values()]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    log.info("wrote %d rules -> %s", len(rules), p)
    return p


def load_rules(path: str | Path) -> dict[str, ClauseRule]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run `make compile` first.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    rules = [ClauseRule.from_dict(r) for r in raw["rules"]]
    return {r.clause_id: r for r in rules}


def snap_rule_names(rule: ClauseRule, committee: Committee | None, cutoff: float = 0.75) -> int:
    """Pull every name the model wrote back onto a real director of that company.

    Safe because every signer in this competition is a registered director — 23,181 of 23,183
    `question` names match committee.csv exactly once whitespace is stripped. So a name the
    model garbled or a title it forgot to drop can be repaired without guessing.
    """
    if committee is None:
        return 0
    from .utils import best_match
    from .rules import Slot

    fixed = 0
    for branch in rule.branches:
        new_slots = []
        for slot in branch.slots:
            if slot.any_director:
                new_slots.append(slot)
                continue
            names = []
            for n in slot.names:
                hit = best_match(n, committee.by_key, cutoff)
                if hit is not None and normalise_name(hit) != normalise_name(n):
                    fixed += 1
                names.append(hit or n)
            new_slots.append(Slot(names=tuple(dict.fromkeys(names)), label=slot.label))
        branch.slots = new_slots
    return fixed
