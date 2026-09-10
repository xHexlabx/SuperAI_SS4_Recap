"""Where the remaining train errors live: by pattern family, branch count and signer count.

    uv run python scripts/error_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_act.config import load_config  # noqa: E402
from legal_act.data import load_committees, load_split  # noqa: E402
from legal_act.engine import load_rules, predict_rows  # noqa: E402


def main() -> int:
    cfg = load_config()
    train = load_split(cfg, "train")
    committees = load_committees(cfg)
    rules = load_rules(cfg.resolve(cfg.evaluate.rules))

    pred = predict_rows(train, rules, committees)
    truth = train["answer"].astype(int)
    df = train.assign(
        pred=pred, truth=truth, hit=pred == truth,
        family=train["pattern"].astype(str).str[:2],
        n_signers=train["signers"].map(len),
        conditional=train["clause_id"].map(
            lambda c: bool(rules.get(c) and rules[c].needs_scope)),
        branches=train["clause_id"].map(lambda c: len(rules[c].branches) if c in rules else 0),
    )

    pd.set_option("display.width", 120)
    for col in ("family", "n_signers", "conditional"):
        print(f"\n=== accuracy by {col} ===")
        print(df.groupby(col).agg(acc=("hit", "mean"), rows=("hit", "size")).round(4))

    print("\n=== accuracy by branch count ===")
    print(df.groupby(df["branches"].clip(upper=6)).agg(acc=("hit", "mean"), rows=("hit", "size")).round(4))

    print("\n=== confusion ===")
    print(pd.crosstab(df["truth"], df["pred"]))
    print(f"\noverall {df['hit'].mean():.4f} on {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
