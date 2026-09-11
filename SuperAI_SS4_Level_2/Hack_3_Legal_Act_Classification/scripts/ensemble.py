"""Union two or more compiled rule sets into one submission.

    uv run python scripts/ensemble.py models/rules_a.json models/rules_b.json \
        --out submissions/union.csv

A row is answered 1 when ANY rule set accepts it. That is the right direction here because the
rules are lopsided: reading the score back out of the leaderboard gives precision 97.8% and
recall 85.7%, so almost every 1 we produce is correct and almost every mistake is a valid
signature we failed to allow. Taking the union trades a little of the precision we have plenty
of for the recall we are short of, and it beat either rule set alone (0.9137 / 0.9218 against
0.9017 / 0.9043 for the better single model).
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
    p.add_argument("--mode", default="union", choices=["union", "intersect"])
    args = p.parse_args()

    setup_logging()
    cfg = load_config()
    committees = load_committees(cfg)
    train = load_split(cfg, "train")
    test = load_split(cfg, "test")

    combined_tr = combined_te = None
    for path in args.rules:
        rules = load_rules(cfg.resolve(path))
        ptr = predict_rows(train, rules, committees, semantics=args.semantics)
        pte = predict_rows(test, rules, committees, semantics=args.semantics)
        print(f"{Path(path).name:28} train macro-F1 {macro_f1(train['answer'].astype(int), ptr):.4f}"
              f"   test answer=1 {pte.mean():.4f}")
        if combined_tr is None:
            combined_tr, combined_te = ptr, pte
        elif args.mode == "union":
            combined_tr, combined_te = combined_tr | ptr, combined_te | pte
        else:
            combined_tr, combined_te = combined_tr & ptr, combined_te & pte

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
