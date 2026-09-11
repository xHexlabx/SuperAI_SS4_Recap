"""Run a named suite of config variants and tabulate CV / application / test-oracle scores.

    uv run python scripts/ablate.py riemann
    uv run python scripts/ablate.py deep
    uv run python scripts/ablate.py custom --set model.name=eegnet --set model.epochs=30

Every variant is a list of `--set` overrides on top of configs/default.yaml. Results are appended
to models/ablation_<suite>.csv so suites can be re-run incrementally (existing run names are skipped).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brain_mi import train  # noqa: E402
from brain_mi.config import load_config  # noqa: E402
from brain_mi.utils import setup_logging  # noqa: E402

SUITES: dict[str, list[tuple[str, list[str]]]] = {
    # --- plan steps 1-4 on the cheap model -------------------------------------------------
    "riemann": [
        ("ts_lr base (8-30, 0.5-3.5, EA sess, cluster)", []),
        ("no alignment", ["align.method=none"]),
        ("riemann alignment", ["align.method=riemann"]),
        ("EA test_group=global", ["align.test_group=global"]),
        ("EA test_group=counter", ["align.test_group=counter"]),
        ("EA train_group=subject", ["align.train_group=subject"]),
        ("EA cov_window=window", ["align.cov_window=window"]),
        ("no highpass/notch (band only)", ["preprocess.highpass_hz=0", "preprocess.notch_hz=[]"]),
        ("window 0-7", ["preprocess.window_s=[0,7]"]),
        ("window 0.5-2.5", ["preprocess.window_s=[0.5,2.5]"]),
        ("window 0.5-4.5", ["preprocess.window_s=[0.5,4.5]"]),
        ("window 1.0-3.5", ["preprocess.window_s=[1.0,3.5]"]),
        ("window 0.25-3.75", ["preprocess.window_s=[0.25,3.75]"]),
        ("band 4-40", ["preprocess.bandpass_hz=[4,40]"]),
        ("band 6-32", ["preprocess.bandpass_hz=[6,32]"]),
        ("band 8-13", ["preprocess.bandpass_hz=[8,13]"]),
        ("clip 6 sigma", ["preprocess.clip_sigma=6"]),
        ("250 Hz (no resample)", ["preprocess.resample_hz=250"]),
        ("lr_C 0.1", ["model.lr_C=0.1"]),
        ("lr_C 10", ["model.lr_C=10"]),
        ("fbts_lr 4 bands", ["model.name=fbts_lr", "preprocess.bandpass_hz=[4,40]"]),
        ("fbts_lr 4 bands C=0.1", ["model.name=fbts_lr", "preprocess.bandpass_hz=[4,40]", "model.lr_C=0.1"]),
        ("csp_lr", ["model.name=csp_lr"]),
        ("ts_lr hierarchical", ["model.hierarchical=true"]),
        ("fbts_lr hierarchical", ["model.name=fbts_lr", "preprocess.bandpass_hz=[4,40]", "model.hierarchical=true"]),
        ("drop Fz (blink)", ["preprocess.channels=[1,2,3,4,5,6,7]"]),
        ("motor only C3 Cz C4", ["preprocess.channels=[1,2,3]"]),
        ("cv by block (sanity: subject seen)", ["cv.scheme=block"]),
    ],
    # --- combine the winners of the riemann suite -------------------------------------------
    "riemann2": [
        ("RA ts_lr", ["align.method=riemann"]),
        ("RA ts_lr + clip6", ["align.method=riemann", "preprocess.clip_sigma=6"]),
        ("RA ts_lr + clip6 + cov_window=window", ["align.method=riemann", "preprocess.clip_sigma=6", "align.cov_window=window"]),
        ("RA csp_lr + clip6", ["align.method=riemann", "preprocess.clip_sigma=6", "model.name=csp_lr"]),
        ("RA csp_lr 8 filters + clip6", ["align.method=riemann", "preprocess.clip_sigma=6", "model.name=csp_lr", "model.csp_filters=8"]),
        ("RA ts_lr + clip6 window 0.5-2.5", ["align.method=riemann", "preprocess.clip_sigma=6", "preprocess.window_s=[0.5,2.5]"]),
        ("RA ts_lr + clip6 window 0.25-3.75", ["align.method=riemann", "preprocess.clip_sigma=6", "preprocess.window_s=[0.25,3.75]"]),
        ("RA ts_lr + clip6 band 7-30", ["align.method=riemann", "preprocess.clip_sigma=6", "preprocess.bandpass_hz=[7,30]"]),
        ("RA ts_lr + clip6 band 8-35", ["align.method=riemann", "preprocess.clip_sigma=6", "preprocess.bandpass_hz=[8,35]"]),
        ("RA fbts 8-13/13-20/20-30 (band 8-30)", ["align.method=riemann", "preprocess.clip_sigma=6", "model.name=fbts_lr", "model.bands=[[8,13],[13,20],[20,30]]"]),
        ("RA ts_lr + clip6 + include_application", ["align.method=riemann", "preprocess.clip_sigma=6", "cv.include_application=true"]),
    ],
    # --- plan steps 5-7: deep nets ---------------------------------------------------------
    "deep": [
        ("eegnet", ["model.name=eegnet"]),
        ("shallow", ["model.name=shallow"]),
        ("conformer", ["model.name=conformer"]),
        ("shallow 4-40", ["model.name=shallow", "preprocess.bandpass_hz=[4,40]"]),
        ("shallow window 0.5-4.5", ["model.name=shallow", "preprocess.window_s=[0.5,4.5]"]),
        ("shallow + reflect", ["model.name=shallow", "model.aug_reflect=true"]),
        ("shallow + chdrop 0.2", ["model.name=shallow", "model.aug_channel_dropout=0.2"]),
        ("shallow + mixup 0.4", ["model.name=shallow", "model.mixup_alpha=0.4"]),
        ("shallow + freqshift 2", ["model.name=shallow", "model.aug_freq_shift_hz=2"]),
        ("shallow + reflect + chdrop + mixup", ["model.name=shallow", "model.aug_reflect=true", "model.aug_channel_dropout=0.2", "model.mixup_alpha=0.4"]),
        ("shallow hierarchical", ["model.name=shallow", "model.hierarchical=true"]),
        ("shallow 120 epochs", ["model.name=shallow", "model.epochs=120"]),
        ("shallow no alignment", ["model.name=shallow", "align.method=none"]),
        ("conformer + reflect", ["model.name=conformer", "model.aug_reflect=true"]),
        ("eegnet + reflect", ["model.name=eegnet", "model.aug_reflect=true"]),
        ("atcnet", ["model.name=atcnet"]),
        ("eegnex", ["model.name=eegnex"]),
    ],
}


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--only", default=None, help="substring filter on variant names")
    ap.add_argument("--force", action="store_true", help="re-run variants already in the table")
    args = ap.parse_args()
    setup_logging()
    variants = SUITES[args.suite] if args.suite in SUITES else [("custom " + " ".join(args.overrides), [])]
    cfg0 = load_config(args.config, args.overrides)
    out = cfg0.resolve(cfg0.train.out_dir) / f"ablation_{args.suite}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for name, ov in variants:
        if args.only and args.only not in name:
            continue
        if not args.force and len(done) and name in set(done["variant"]):
            continue
        t0 = time.time()
        cfg = load_config(args.config, args.overrides + ov)
        cfg.train.run_name = f"abl_{args.suite}_{slug(name)}"
        r = train.run(cfg)
        row = dict(variant=name, cv_acc=r["cv"]["acc"], cv_std=r["cv"]["acc_std"], cv_f1=r["cv"]["f1"],
                   app_acc=r.get("application", {}).get("acc"), app_f1=r.get("application", {}).get("f1"),
                   oracle_folds=r.get("test_oracle_foldavg", {}).get("acc"), oracle_full=r.get("test_oracle_full", {}).get("acc"),
                   oracle_full_f1=r.get("test_oracle_full", {}).get("f1"), secs=round(time.time() - t0), overrides=" ".join(ov))
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(done.drop(columns=["overrides"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
