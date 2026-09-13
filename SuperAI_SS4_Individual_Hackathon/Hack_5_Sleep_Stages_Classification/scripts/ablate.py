"""Run a named suite of config variants and tabulate the leave-subjects-out scores.

    uv run python scripts/ablate.py seq
    uv run python scripts/ablate.py custom --set model.name=lgbm --set smooth.method=none

Every variant is a list of `--set` overrides on top of configs/default.yaml. Results are
appended to results/ablation_<suite>.csv, so a suite can be re-run incrementally — variants
already in the file are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleep_stage import train  # noqa: E402
from sleep_stage.config import load_config  # noqa: E402
from sleep_stage.utils import setup_logging  # noqa: E402

FAST = ["cv.full_fit=false"]
SWEEP_FOLDS = ["cv.n_folds=3"]   # sweeps use 3 folds; the final config is re-run at 5

SUITES: dict[str, list[tuple[str, list[str]]]] = {
    # --- how the sequence model should be calibrated and decoded ---------------------------
    "seq": [
        ("bilstm e60 balanced", ["model.name=bilstm"]),
        ("bilstm e60 plain loss", ["model.name=bilstm", "model.loss_class_weight=none"]),
        ("bilstm e60 no Viterbi", ["model.name=bilstm", "smooth.method=none"]),
        ("bilstm e150", ["model.name=bilstm", "model.epochs=150"]),
        ("tcn e150", ["model.name=tcn", "model.epochs=150"]),
        ("gru e150", ["model.name=gru", "model.epochs=150"]),
    ],
    # --- capacity / training budget ---------------------------------------------------------
    "capacity": [
        ("bilstm h128 L2 e60", ["model.name=bilstm"]),
        ("bilstm h128 L2 e150", ["model.name=bilstm", "model.epochs=150"]),
        ("bilstm h192 L2 e150", ["model.name=bilstm", "model.epochs=150", "model.hidden=192"]),
        ("bilstm h128 L3 e150", ["model.name=bilstm", "model.epochs=150", "model.layers=3"]),
        ("bilstm h128 L2 e150 drop0.5", ["model.name=bilstm", "model.epochs=150", "model.dropout=0.5"]),
        ("bilstm crop 240 e150", ["model.name=bilstm", "model.epochs=150", "model.crop_epochs=240"]),
        ("gru h128 L2 e150", ["model.name=gru", "model.epochs=150"]),
        ("tcn h128 e150", ["model.name=tcn", "model.epochs=150"]),
    ],
    # --- what the features are worth --------------------------------------------------------
    "features": [
        ("lgbm full", ["model.name=lgbm"]),
        ("lgbm no per-subject norm", ["model.name=lgbm", "context.per_subject_norm=false"]),
        ("lgbm no absolute copy", ["model.name=lgbm", "context.keep_absolute=false"]),
        ("lgbm no clock", ["model.name=lgbm", "context.clock=false"]),
        ("lgbm no rolling", ["model.name=lgbm", "context.roll_windows=[]"]),
        ("lgbm no lags", ["model.name=lgbm", "context.lags=[]"]),
        ("lgbm no context at all", ["model.name=lgbm", "context.roll_windows=[]", "context.lags=[]"]),
        ("lgbm no Viterbi", ["model.name=lgbm", "smooth.method=none"]),
    ],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", help="name of a suite in SUITES, or 'custom'")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging()

    variants = SUITES.get(args.suite) or [(" ".join(args.overrides) or "default", [])]
    out = Path(__file__).resolve().parents[1] / "results" / f"ablation_{args.suite}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["variant"]) if out.exists() else set()

    rows = []
    for name, overrides in variants:
        if name in done:
            print(f"skip (already in {out.name}): {name}")
            continue
        t0 = time.time()
        cfg = load_config(args.config, SWEEP_FOLDS + overrides + args.overrides + FAST + [f"train.run_name=abl_{abs(hash(name)) % 10**8}"])
        res = train.run(cfg)
        raw, sm = res["oof"]["raw"], res["oof"]["smoothed"]
        cal = res["oof"].get("calibrated", sm)
        rows.append(dict(variant=name, macro_f1=cal["macro_f1"], macro_f1_smoothed=sm["macro_f1"],
                         macro_f1_raw=raw["macro_f1"],
                         weighted_f1=cal["weighted_f1"], acc=cal["acc"], kappa=cal["kappa"],
                         f1_N=cal["per_class"]["N"], f1_R=cal["per_class"]["R"], f1_W=cal["per_class"]["W"],
                         secs=int(time.time() - t0), overrides=" ".join(overrides)))
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        rows = []
        print(f"[{args.suite}] {name}: macro-F1 {sm['macro_f1']:.4f} ({time.time() - t0:.0f}s)")

    if out.exists():
        df = pd.read_csv(out).sort_values("macro_f1", ascending=False)
        print(df[["variant", "macro_f1", "macro_f1_smoothed", "macro_f1_raw", "acc",
                  "f1_N", "f1_R", "f1_W", "secs"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
