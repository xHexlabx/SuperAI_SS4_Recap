"""Combine two or more compiled rule sets into one submission.

    uv run python scripts/ensemble.py models/rules_flash_v6.json models/rules_27b_v2.json \
        models/rules.json --mode vote --min-votes 2 --out submissions/vote.csv

Which combination is right depends on which way the rules are currently wrong, and the
leaderboard tells you: solve the macro-F1 back into precision/recall (see README).
  * recall is the bottleneck (few false 1s, many missed 1s)  -> union: 1 if ANY set says 1.
    That was the case before the prompt rewrite (P 0.978 / R 0.857) and union gave +0.012.
  * precision is the bottleneck (false 1s outnumber misses)  -> vote or intersect.
    After the rewrite (P 0.954 / R 0.985) a 2-of-3 vote scored 0.9731 / 0.9778 against
    0.9504 / 0.9629 for the best single set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_act.config import load_config  # noqa: E402
from legal_act.data import load_committees, load_split  # noqa: E402
from legal_act.engine import load_rules, macro_f1, predict_rows  # noqa: E402
from legal_act.utils import ensure_dir, setup_logging  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rules", nargs="+", help="two or more rules.json files")
    p.add_argument("--out", default="submissions/union.csv")
    p.add_argument("--semantics", default="at_least", choices=["exact", "at_least"])
    p.add_argument("--mode", default="union", choices=["union", "intersect", "vote"])
    p.add_argument("--min-votes", type=int, default=2, help="for --mode vote: 1s needed to answer 1")
    args = p.parse_args()

    setup_logging()
    cfg = load_config()
    committees = load_committees(cfg)
    train = load_split(cfg, "train")
    test = load_split(cfg, "test")

    votes_tr = votes_te = None
    for path in args.rules:
        rules = load_rules(cfg.resolve(path))
        ptr = predict_rows(train, rules, committees, semantics=args.semantics).astype(int)
        pte = predict_rows(test, rules, committees, semantics=args.semantics).astype(int)
        print(f"{Path(path).name:28} train macro-F1 {macro_f1(train['answer'].astype(int), ptr):.4f}"
              f"   test answer=1 {pte.mean():.4f}")
        votes_tr = ptr if votes_tr is None else votes_tr + ptr
        votes_te = pte if votes_te is None else votes_te + pte

    n = len(args.rules)
    need = {"union": 1, "intersect": n, "vote": args.min_votes}[args.mode]
    combined_tr = (votes_tr >= need).astype(int)
    combined_te = (votes_te >= need).astype(int)

    print(f"{args.mode:28} train macro-F1 "
          f"{macro_f1(train['answer'].astype(int), combined_tr):.4f}"
          f"   test answer=1 {combined_te.mean():.4f}")

    out = cfg.resolve(args.out)
    ensure_dir(out.parent)
    pd.DataFrame({"id": test["id"], "answer": combined_te.astype(int)}).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
