"""Score every saved run and greedy-build a probability-averaging ensemble.

    uv run python scripts/ensemble.py                 # rank runs, greedy ensemble by CV OOF macro-F1
    uv run python scripts/ensemble.py --by app        # select by train_application instead
    uv run python scripts/ensemble.py --runs a b c    # just score this fixed combination

Selection never looks at the test oracle; it is printed alongside as the third layer only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brain_mi.config import load_config  # noqa: E402
from brain_mi.data import CLASSES  # noqa: E402
from brain_mi.preprocess import load_dataset  # noqa: E402


def score(y, p):
    pred = np.array(CLASSES)[p.argmax(1)]
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", default="cv", choices=["cv", "app", "both"])
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--min-cv", type=float, default=0.0, help="ignore runs with OOF macro-F1 below this")
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--filter", default=None, help="only consider runs whose name contains this substring")
    args = ap.parse_args()
    cfg = load_config()
    ds = load_dataset(cfg)
    ytr, yap, yte = ds.train.y, ds.app.y, ds.test_oracle
    models = cfg.resolve(cfg.train.out_dir)
    runs = {}
    for d in sorted(models.iterdir()):
        if not (d / "results.json").exists() or not (d / "test_probs_full.npy").exists():
            continue
        r = json.loads((d / "results.json").read_text())
        runs[d.name] = dict(oof=np.load(d / "oof_probs.npy"), app=np.load(d / "app_probs.npy"),
                            te=np.load(d / "test_probs_full.npy"), cv_scheme=r["config"]["cv"]["scheme"])
    runs = {k: v for k, v in runs.items() if v["cv_scheme"] == "subject" and len(v["oof"]) == len(ytr)
            and (args.filter is None or args.filter in k)}
    rows = []
    for k, v in runs.items():
        a1, f1 = score(ytr, v["oof"]); a2, f2 = score(yap, v["app"]); a3, f3 = score(yte, v["te"])
        rows.append(dict(run=k, cv_acc=a1, cv_f1=f1, app_acc=a2, app_f1=f2, oracle_acc=a3, oracle_f1=f3))
    tab = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))

    def obj(sel):
        oof = np.mean([runs[k]["oof"] for k in sel], 0); app = np.mean([runs[k]["app"] for k in sel], 0)
        c, a = score(ytr, oof)[1], score(yap, app)[1]
        return {"cv": c, "app": a, "both": (c + a) / 2}[args.by]

    if args.runs:
        sel = args.runs
    else:
        cands = [k for k in tab.run if tab.set_index("run").cv_f1[k] >= args.min_cv]
        sel = []
        best = -1
        while len(sel) < args.max:
            trial = [(obj(sel + [k]), k) for k in cands]  # with replacement allowed (weights)
            v, k = max(trial)
            if v <= best + 1e-4:
                break
            best = v; sel.append(k)
    oof = np.mean([runs[k]["oof"] for k in sel], 0); app = np.mean([runs[k]["app"] for k in sel], 0); te = np.mean([runs[k]["te"] for k in sel], 0)
    print("\nensemble (%d members, selected by %s):" % (len(sel), args.by))
    for k in sel:
        print("  ", k)
    print("  cv acc/f1 %.3f/%.3f | app %.3f/%.3f | test-oracle %.3f/%.3f" % (*score(ytr, oof), *score(yap, app), *score(yte, te)))
    print("\nuse:  uv run mi-predict --set 'predict.runs=[%s]'" % ",".join(sel))


if __name__ == "__main__":
    main()
