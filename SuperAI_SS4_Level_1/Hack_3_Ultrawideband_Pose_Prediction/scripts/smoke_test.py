#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data — no Kaggle download needed.

Run this straight after `make setup` to prove the install works (GPU, timm weights, encoders,
train loop, submission writer) before spending time on the real dataset:

    uv run python scripts/smoke_test.py

It exercises both data layouts: a wide CSV of 1-D traces with the generic encoders, and a
directory of complex .npy radar samples with the radar encoders the real competition uses.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uwb_pose import predict, train  # noqa: E402
from uwb_pose.config import load_config  # noqa: E402
from uwb_pose.utils import setup_logging  # noqa: E402

N_CLASSES = 4
N_TRAIN = 96
N_TEST = 16
SIG_LEN = 256
FRAMES, BINS = 160, 56


def _class_signature(cls: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Each 'pose' gets its own carrier frequency and burst position, so a classifier can
    separate them, but only by reading time-frequency structure — the same thing the encoders
    expose to the backbone on real UWB traces."""
    t = np.linspace(0, 1, n)
    burst = np.exp(-((t - (0.2 + 0.2 * cls)) ** 2) / 0.005)
    return np.sin(2 * np.pi * (5.0 + 7.0 * cls) * t) * burst + 0.05 * rng.standard_normal(n)


def wide_csv(tmp: Path, rng: np.random.Generator) -> None:
    for name, n, labelled in (("train.csv", N_TRAIN, True), ("test.csv", N_TEST, False)):
        rows = [_class_signature(i % N_CLASSES, SIG_LEN, rng) for i in range(n)]
        df = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=[f"s{i}" for i in range(SIG_LEN)])
        df.insert(0, "id", [f"sample_{i:04d}" for i in range(n)])
        if labelled:
            df["class"] = [f"pose_{i % N_CLASSES}" for i in range(n)]
        df.to_csv(tmp / name, index=False)


def npy_dir(tmp: Path, rng: np.random.Generator) -> None:
    (tmp / "train" / "train").mkdir(parents=True)
    (tmp / "test" / "test").mkdir(parents=True)
    rows = []
    for split, n, labelled in (("train", N_TRAIN, True), ("test", N_TEST, False)):
        for i in range(n):
            cls = i % N_CLASSES
            slow = _class_signature(cls, FRAMES, rng)
            profile = np.exp(-((np.arange(BINS) - (8 + 10 * cls)) ** 2) / 12.0)
            iq = (slow[:, None] * profile[None, :]).astype(np.complex128)
            iq += 0.01 * (rng.standard_normal((FRAMES, BINS)) + 1j * rng.standard_normal((FRAMES, BINS)))
            sid = f"{split}_{i:04d}"
            np.save(tmp / split / split / f"{sid}.npy", iq)
            if labelled:
                rows.append({"id": sid, "class": cls})
    pd.DataFrame(rows).to_csv(tmp / "annotations.csv", index=False)
    pd.DataFrame({"id": range(N_CLASSES), "class": [f"pose_{i}" for i in range(N_CLASSES)]}).to_csv(
        tmp / "classes.csv", index=False
    )


def check(name: str, tmp: Path, overrides: list[str]) -> None:
    cfg = load_config(overrides=[
        f"data.root={tmp}",
        f"encode.cache_dir={tmp}/cache",
        "encode.img_size=64",
        "model.name=resnet18",
        "train.epochs=2",
        "train.batch_size=16",
        "train.warmup_epochs=1",
        "train.num_workers=2",
        f"train.out_dir={tmp}/models",
        f"predict.checkpoint={tmp}/models/best.pt",
        f"predict.out_csv={tmp}/submission.csv",
    ] + overrides)

    best = train.run(cfg)
    sub = predict.run(cfg)
    assert len(sub) == N_TEST, f"{name}: expected {N_TEST} predictions, got {len(sub)}"
    print(f"  {name}: OK — val acc {best['acc']:.3f}, {len(sub)} predictions")


def main() -> int:
    setup_logging()
    rng = np.random.default_rng(0)

    for name, build, overrides in (
        ("wide_csv + 1-D encoders", wide_csv,
         ["data.layout=wide_csv", "encode.method=stack",
          "encode.stack_methods=[spectrogram, cwt, gaf]", "encode.n_scales=32"]),
        ("npy_dir + radar encoders", npy_dir,
         ["data.layout=npy_dir", "encode.method=stack", f"encode.n_frames={FRAMES}",
          "encode.md_nperseg=32", "encode.md_noverlap=24", "train.folds=2"]),
    ):
        tmp = Path(tempfile.mkdtemp(prefix="uwb_smoke_"))
        try:
            build(tmp, rng)
            check(name, tmp, overrides)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
