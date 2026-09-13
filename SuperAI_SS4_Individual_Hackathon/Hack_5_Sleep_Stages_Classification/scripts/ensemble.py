"""Score every saved run and greedy-build a probability-averaging ensemble.

    uv run python scripts/ensemble.py                    # rank runs, greedy ensemble on OOF macro-F1
    uv run python scripts/ensemble.py --runs a b c       # score one fixed combination
    uv run python scripts/ensemble.py --write            # ...and write submissions/submission.csv

Selection is done on out-of-fold probabilities only. The blend is decoded with the same
Viterbi pass and the same per-class calibration used at predict time, because a blend that
wins before decoding is not necessarily the blend that wins after it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleep_stage import CLASSES  # noqa: E402
from sleep_stage.calibrate import apply_bias, fit_bias  # noqa: E402
from sleep_stage.config import load_config  # noqa: E402
from sleep_stage.data import load_dataset  # noqa: E402
from sleep_stage.smooth import class_prior, decode, transition_matrix  # noqa: E402
from sleep_stage.utils import setup_logging  # noqa: E402


def split_nights(flat: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    return list(np.split(flat, np.cumsum(lengths)[:-1]))


def evaluate(probas, ys, A, prior, cfg, calibrate: bool):
    y_true = np.concatenate(ys)
    bias = np.zeros(len(CLASSES))
    if calibrate:
        bias, _ = fit_bias(probas, ys, A, prior, cfg.smooth)
    pred = np.concatenate(decode(apply_bias(probas, bias), A, prior, cfg.smooth))
    return float(f1_score(y_true, pred, average="macro", zero_division=0)), bias


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="fixed set of run names to blend")
    ap.add_argument("--max", type=int, default=6, help="maximum runs in the greedy ensemble")
    ap.add_argument("--filter", default=None, help="only consider runs whose name contains this")
    ap.add_argument("--min-f1", type=float, default=0.0, help="ignore runs below this OOF macro-F1")
    ap.add_argument("--write", action="store_true", help="write the winning blend to predict.out_csv")
    args = ap.parse_args()
    setup_logging()

    cfg = load_config()
    ds = load_dataset(cfg)
    ys = [n.y.astype(int) for n in ds.train]
    tr_len = [len(n.X) for n in ds.train]
    te_len = [len(n.X) for n in ds.test]
    A, prior = transition_matrix(ys), class_prior(ys)

    models = cfg.resolve(cfg.train.out_dir)
    runs: dict[str, dict] = {}
    for d in sorted(models.iterdir()):
        if not (d / "oof_probs.npy").exists() or not (d / "test_probs_full.npy").exists():
            continue
        if args.filter and args.filter not in d.name:
            continue
        oof = split_nights(np.load(d / "oof_probs.npy"), tr_len)
        f1, _ = evaluate(oof, ys, A, prior, cfg, calibrate=False)
        if f1 < args.min_f1:
            continue
        runs[d.name] = dict(oof=oof, test=split_nights(np.load(d / "test_probs_full.npy"), te_len), f1=f1)
    if not runs:
        print(f"no usable runs in {models}")
        return 1

    table = pd.DataFrame([dict(run=k, oof_macro_f1=v["f1"]) for k, v in runs.items()]).sort_values(
        "oof_macro_f1", ascending=False)
    print(table.to_string(index=False))

    if args.runs:
        chosen = list(args.runs)
    else:
        chosen, best = [], -1.0
        pool = list(table.run)
        while len(chosen) < args.max:
            gains = []
            for r in pool:
                trial = chosen + [r]
                blend = [np.mean([runs[t]["oof"][i] for t in trial], axis=0) for i in range(len(ys))]
                gains.append((evaluate(blend, ys, A, prior, cfg, calibrate=False)[0], r))
            gains.sort(reverse=True)
            if gains[0][0] <= best + 1e-4:
                break
            best, pick = gains[0]
            chosen.append(pick)
            print(f"  + {pick}: blend OOF macro-F1 {best:.4f}")

    blend_oof = [np.mean([runs[t]["oof"][i] for t in chosen], axis=0) for i in range(len(ys))]
    f1, bias = evaluate(blend_oof, ys, A, prior, cfg, calibrate=cfg.smooth.calibrate)
    print(f"\nensemble {chosen}\n  OOF macro-F1 after calibration + Viterbi: {f1:.4f}"
          f"\n  class bias: {({c: round(float(v), 2) for c, v in zip(CLASSES, bias)})}")

    if args.write:
        blend_te = [np.mean([runs[t]["test"][i] for t in chosen], axis=0) for i in range(len(te_len))]
        pred = np.concatenate(decode(apply_bias(blend_te, bias), A, prior, cfg.smooth))
        ids = np.concatenate([n.ids for n in ds.test])
        sub = pd.DataFrame({"id": ids, "labels": np.array(CLASSES)[pred]})
        ss = pd.read_csv(cfg.resolve(cfg.data.root) / "sample_submission.csv", encoding="utf-8-sig")
        sub = ss[["id"]].merge(sub, on="id", how="left")
        assert sub.labels.notna().all(), "missing test ids"
        out = cfg.resolve(cfg.predict.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        print(f"  wrote {out} | class counts {sub.labels.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
