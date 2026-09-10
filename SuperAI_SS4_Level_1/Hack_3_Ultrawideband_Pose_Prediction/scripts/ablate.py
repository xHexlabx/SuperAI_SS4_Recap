#!/usr/bin/env python3
"""Compare encodings on the same split, so the choice of image is evidence rather than taste.

    uv run python scripts/ablate.py            # the default sweep
    uv run python scripts/ablate.py rti stack  # only the named variants
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uwb_pose import train  # noqa: E402
from uwb_pose.config import load_config  # noqa: E402
from uwb_pose.utils import setup_logging  # noqa: E402

VARIANTS: dict[str, list[str]] = {
    # single views — is any one of them enough on its own?
    "rti":              ["encode.method=rti"],
    "range_doppler":    ["encode.method=range_doppler"],
    "micro_doppler":    ["encode.method=micro_doppler"],
    # the 1-D baseline this hack is supposed to beat: flatten to a trace, then spectrogram
    "spectrogram_1d":   ["encode.method=spectrogram"],
    # three radar views in three RGB channels
    "stack":            ["encode.method=stack"],
    "stack_no_clutter": ["encode.method=stack", "encode.clutter_removal=false"],
    "stack_no_crop":    ["encode.method=stack", "encode.doppler_crop=0"],
    "stack_raw_wide":   ["encode.method=stack", "encode.clutter_removal=false",
                         "encode.doppler_crop=0"],

    # --- keeping the range axis through the time-frequency transform ---
    # micro_doppler sums complex across range BEFORE the STFT, so reflections at different
    # ranges can cancel; spectrogram2d takes the STFT per range bin and sums magnitudes after.
    "spectrogram2d":    ["encode.method=spectrogram2d"],
    "fft2":             ["encode.method=fft2"],
    # range-Doppler of three consecutive time slices, one per RGB channel — puts back the
    # time ordering that a whole-clip range-Doppler averages away.
    "rd_time":          ["encode.method=stack", "encode.stack_methods=[rd_t0, rd_t1, rd_t2]"],
    # the two strongest single views plus position
    "md_sg2d_rti":      ["encode.method=stack",
                         "encode.stack_methods=[rti, micro_doppler, spectrogram2d]"],
    "md_sg2d_rti_raw":  ["encode.method=stack", "encode.clutter_removal=false",
                         "encode.stack_methods=[rti, micro_doppler, spectrogram2d]"],

    # --- tuning the winning encoder rather than searching for new combinations ---
    "sg2d_no_clutter":  ["encode.method=spectrogram2d", "encode.clutter_removal=false"],
    # a shorter STFT window trades velocity resolution for time resolution
    "sg2d_fine":        ["encode.method=spectrogram2d", "encode.md_nperseg=128",
                         "encode.md_noverlap=112"],
    "sg2d_coarse":      ["encode.method=spectrogram2d", "encode.md_nperseg=512",
                         "encode.md_noverlap=448"],
    "sg2d_wide_db":     ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80"],
    # accuracy rose monotonically as the window shrank (512 -> 256 -> 128); push until it turns
    "sg2d_finer":       ["encode.method=spectrogram2d", "encode.md_nperseg=64",
                         "encode.md_noverlap=56"],
    "sg2d_finest":      ["encode.method=spectrogram2d", "encode.md_nperseg=32",
                         "encode.md_noverlap=28"],
    # a shorter window and a wider dB range each helped on their own — do they compose?
    "sg2d_128_60db":    ["encode.method=spectrogram2d", "encode.md_nperseg=128",
                         "encode.md_noverlap=112", "encode.dynamic_range=60"],
    "sg2d_64_60db":     ["encode.method=spectrogram2d", "encode.md_nperseg=64",
                         "encode.md_noverlap=56", "encode.dynamic_range=60"],
    "sg2d_128_80db":    ["encode.method=spectrogram2d", "encode.md_nperseg=128",
                         "encode.md_noverlap=112", "encode.dynamic_range=80"],
    "sg2d_128_100db":   ["encode.method=spectrogram2d", "encode.md_nperseg=128",
                         "encode.md_noverlap=112", "encode.dynamic_range=100"],
    "sg2d_128_140db":   ["encode.method=spectrogram2d", "encode.md_nperseg=128",
                         "encode.md_noverlap=112", "encode.dynamic_range=140"],

    # --- regularisation: 647 samples is small enough that mixing pairs should pay ---
    "mixup02":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2"],
    # --- how long to train: epoch cap vs early-stop patience ---
    # mixup04_long changed both at once (25->45 epochs AND patience 5->10) and gained 2.2
    # points, so which one mattered is unknown. Pin the cap high and vary only patience.
    "p05_e60":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.epochs=60", "train.early_stop_patience=5"],
    "p10_e60":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.epochs=60", "train.early_stop_patience=10"],
    "p20_e60":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.epochs=60", "train.early_stop_patience=20"],
    # does the un-augmented baseline want a long leash too, or is this a mixup-only effect?
    "p20_e60_nomix":    ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.epochs=60", "train.early_stop_patience=20"],
    # strongest mixing with the longest leash — mixup04 was the one most starved of epochs
    "mixup04_p20_e60":  ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.4", "train.epochs=60",
                         "train.early_stop_patience=20"],

    # accuracy rose monotonically with patience (5 -> 10 -> 20) and had not turned over;
    # `patience >= epochs` disables early stopping altogether, which is the end of the line.
    "p30_e80":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.epochs=80", "train.early_stop_patience=30"],
    "nostop_e60":       ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.epochs=60", "train.early_stop_patience=60"],
    "nostop_e100":      ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.epochs=100", "train.early_stop_patience=100"],
    # mixup 0.4 was the setting most starved of epochs — give it the longest run of all
    "mixup04_nostop_e100": ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.4", "train.epochs=100",
                            "train.early_stop_patience=100"],
    # the stability trade-off: does a bigger mix keep the variance down over a long run?
    "mixup03_p30_e80":  ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.3", "train.epochs=80",
                         "train.early_stop_patience=30"],

    # --- put the range axis back ---
    # spectrogram2d sums magnitudes over all 56 range bins, so the best encoding we have throws
    # away where the motion happened. Three contiguous range bands, one per RGB channel, keep
    # coarse position at no extra image size — and unlike the earlier failed stacks, all three
    # channels are the same kind of measurement, so none is a weak view dragging the rest down.
    "rb3_maxvit":    ["encode.method=stack", "encode.stack_methods=[sgband0, sgband1, sgband2]", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.6", "train.epochs=60", "train.early_stop_patience=20"],
    "rb3_cnx":       ["encode.method=stack", "encode.stack_methods=[sgband0, sgband1, sgband2]", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_tiny.fb_in1k",
                      "train.mixup_alpha=0.4", "train.epochs=60", "train.early_stop_patience=20"],
    "rb3_cnx_mix06": ["encode.method=stack", "encode.stack_methods=[sgband0, sgband1, sgband2]", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_tiny.fb_in1k",
                      "train.mixup_alpha=0.6", "train.epochs=60", "train.early_stop_patience=20"],
    "rb3_maxvit_mix08": ["encode.method=stack", "encode.stack_methods=[sgband0, sgband1, sgband2]", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.8", "train.epochs=60", "train.early_stop_patience=20"],
    "rb5_maxvit":    ["encode.method=stack",
                      "encode.stack_methods=[sgband0, sgband2, sgband4]",
                      "encode.range_bands=5", "encode.md_nperseg=128",
                      "encode.md_noverlap=112", "encode.dynamic_range=80",
                      "train.mixup_alpha=0.6", "train.epochs=60", "train.early_stop_patience=20"],

    # convnext at the long schedule is the best result so far. Patience saturated at 20 -
    # longer runs reproduce the identical checkpoint - so vary the mix strength instead.
    "cnx_mix06":     ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_tiny.fb_in1k",
                      "train.mixup_alpha=0.6", "train.epochs=60", "train.early_stop_patience=20"],
    "cnx_mix03":     ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_tiny.fb_in1k",
                      "train.mixup_alpha=0.3", "train.epochs=60", "train.early_stop_patience=20"],
    "cnx_base_long": ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_base.fb_in1k",
                      "train.mixup_alpha=0.4", "train.batch_size=12",
                      "train.grad_accum=3", "train.epochs=60", "train.early_stop_patience=20"],

    # convnext_tiny beat maxvit_tiny by 0.8 points at the short schedule, with a quarter of
    # the fold-to-fold spread. Give it the long schedule that lifted maxvit by 4 points.
    "cnx_long":         ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_tiny.fb_in1k",
                         "train.mixup_alpha=0.4", "train.epochs=60", "train.early_stop_patience=20"],
    "cnx_long_nomix":   ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_tiny.fb_in1k", "train.epochs=60", "train.early_stop_patience=20"],
    "cnx_nostop_e100":  ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_tiny.fb_in1k", "train.mixup_alpha=0.4",
                         "train.epochs=100", "train.early_stop_patience=100"],
    "cnx_small_long":   ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "model.name=convnext_small.fb_in1k",
                         "train.mixup_alpha=0.4", "train.batch_size=16",
                         "train.grad_accum=2", "train.epochs=60", "train.early_stop_patience=20"],

    # alpha 0.4 was the WORST setting at 25 epochs and the BEST at 60 — mix strength and
    # training length have to be raised together. Push the pairing further.
    "mixup06_nostop_e100": ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.6", "train.epochs=100",
                            "train.early_stop_patience=100"],
    "mixup08_nostop_e100": ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.8", "train.epochs=100",
                            "train.early_stop_patience=100"],
    "mixup04_nostop_e150": ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.4", "train.epochs=150",
                            "train.early_stop_patience=150"],

    # --- different architectures on the winning encoding ---
    # For an ensemble, models that make *different* mistakes beat models that are individually
    # better. Three unrelated backbone families should decorrelate more than three encodings
    # that all came out of the same STFT.
    "arch_convnext":    ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "model.name=convnext_tiny.fb_in1k"],
    "arch_swin":        ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "model.name=swin_tiny_patch4_window7_224.ms_in1k"],
    "arch_resnet50":    ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "model.name=resnet50.a1_in1k"],
    "arch_effnet":      ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "model.name=efficientnet_b0.ra_in1k"],

    # --- stop squashing the STFT into a square ---
    # The encoder resizes a (freq x time) map to img_size x img_size. At window 128 that is
    # 128x160 stretched to 224x224 — pick a window and hop whose native map is already square,
    # so the resize interpolates as little as possible.
    "sg2d_square":      ["encode.method=spectrogram2d", "encode.md_nperseg=224",
                         "encode.md_noverlap=213", "encode.dynamic_range=80", "train.mixup_alpha=0.2"],
    "sg2d_square_192":  ["encode.method=spectrogram2d", "encode.md_nperseg=192",
                         "encode.md_noverlap=179", "encode.dynamic_range=80", "train.mixup_alpha=0.2"],

    # --- more folds: more training data per model, and more ensemble members ---
    "folds10":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.folds=10"],

    # accuracy peaked at a *gentler* mix than the literature's default — find the real peak
    "mixup01":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.1"],
    "mixup015":         ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.15"],
    "mixup03":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.3"],
    "cutmix_light":     ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.cutmix_alpha=0.2"],
    # the same gentle mix, applied on only a quarter of the batches
    "mixup02_p25":      ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.mix_prob=0.25"],
    "mixup02_long":     ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.2", "train.epochs=45",
                         "train.early_stop_patience=10"],
    "mixup04":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.4"],
    "mixup08":          ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.8"],
    "cutmix":           ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.cutmix_alpha=1.0"],
    "mixup_cutmix":     ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.4", "train.cutmix_alpha=1.0"],
    # mixing lets the model train longer before it starts memorising
    "mixup04_long":     ["encode.method=spectrogram2d", "encode.md_nperseg=128", "encode.md_noverlap=112", "encode.dynamic_range=80", "train.mixup_alpha=0.4", "train.epochs=45",
                         "train.early_stop_patience=10"],

    # --- kept only for ensemble diversity: different encoder, different mistakes ---
    "enc_micro_doppler":["encode.method=micro_doppler", "encode.dynamic_range=60"],
    "enc_rti":          ["encode.method=rti", "encode.dynamic_range=60"],
    # more pixels for the same image, and a backbone that expects them
    "sg2d_384":         ["encode.method=spectrogram2d", "encode.img_size=384",
                         "model.name=maxvit_tiny_tf_384.in1k", "train.batch_size=8",
                         "train.grad_accum=4"],
}

COMMON = [
    "train.epochs=25",
    "train.lr=1e-4",
    "train.folds=5",
]


def main() -> int:
    setup_logging(logging.WARNING)  # keep the sweep readable; per-epoch logs are noise here
    wanted = sys.argv[1:] or list(VARIANTS)
    results = {}

    for name in wanted:
        if name not in VARIANTS:
            print(f"unknown variant {name!r}; known: {list(VARIANTS)}")
            return 2
        print(f"\n=== {name} ===", flush=True)
        # Each variant keeps its own run directory, so scripts/analyze.py can look at any of
        # them afterwards instead of only whichever ran last.
        cfg = load_config(overrides=COMMON + [f"train.run_name=ablate_{name}"] + VARIANTS[name])
        best = train.run(cfg)
        results[name] = best
        print(f"  -> acc {best['acc']:.4f} ± {best.get('acc_std', 0):.4f}  "
              f"macro_f1 {best['macro_f1']:.4f} ± {best.get('macro_f1_std', 0):.4f}", flush=True)

    print(f"\n{'variant':20s} {'acc':>16s} {'macro_f1':>16s}")
    print("-" * 54)
    for name, best in sorted(results.items(), key=lambda kv: -kv[1]["acc"]):
        acc = f"{best['acc']:.4f} ± {best.get('acc_std', 0):.3f}"
        f1 = f"{best['macro_f1']:.4f} ± {best.get('macro_f1_std', 0):.3f}"
        print(f"{name:20s} {acc:>16s} {f1:>16s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
