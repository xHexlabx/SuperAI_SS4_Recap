"""Print one clause, the rule compiled from it, and the train rows it still gets wrong.

    uv run python scripts/inspect_clause.py                # the 10 worst clauses
    uv run python scripts/inspect_clause.py 105529030059   # by company
    uv run python scripts/inspect_clause.py <clause_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_act.config import load_config  # noqa: E402
from legal_act.data import load_committees, load_split  # noqa: E402
from legal_act.engine import load_rules, predict_rows, score  # noqa: E402


def show(cid, train, rules, committees) -> None:
    g = train[train["clause_id"] == cid]
    if g.empty:
        print(f"{cid}: no train rows")
        return
    rule = rules.get(cid)
    pred = predict_rows(g, rules, committees)
    acc = (pred == g["answer"].astype(int)).mean()
    com = committees.get(g["rg"].iloc[0])

    print("=" * 100)
    print(f"{cid}   pattern={g['pattern'].iloc[0]}   rows={len(g)}   acc={acc:.3f}")
    print(f"DIRECTORS: {', '.join(com.names) if com else '-'}")
    print(f"CLAUSE   : {g['context'].iloc[0]}")
    conds = sorted({c for c in g["condition"] if c})
    for c in conds:
        print(f"CONDITION: {c}")
    print("RULE     :", json.dumps(rule.to_dict(), ensure_ascii=False) if rule else "(none)")
    wrong = g.loc[pred != g["answer"].astype(int)]
    for i, r in wrong.head(12).iterrows():
        print(f"  ✗ truth={int(r['answer'])} pred={int(pred.loc[i])} | {r['legal_act']} "
              f"| {' + '.join(r['signers'])}")


def main() -> int:
    cfg = load_config()
    train = load_split(cfg, "train")
    committees = load_committees(cfg)
    rules = load_rules(cfg.resolve(cfg.evaluate.rules))

    if len(sys.argv) > 1:
        want = sys.argv[1]
        ids = [c for c in train["clause_id"].unique() if want in c]
        if not ids:
            print(f"no clause matching {want!r}")
            return 1
    else:
        stats = score(train, rules, committees)
        ids = list(stats["per_clause"][stats["per_clause"]["acc"] < 1.0].head(10).index)
        if not ids:
            print("every train clause is at 1.000 — nothing to inspect")
            return 0

    for cid in ids:
        show(cid, train, rules, committees)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
