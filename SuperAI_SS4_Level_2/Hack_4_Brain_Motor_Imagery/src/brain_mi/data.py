"""Loading, epoching and test-block reconstruction for the Brain Motor Imagery dataset.

Device: g.tec Unicorn Hybrid Black, 250 Hz, 17 columns per sample:
    0-7   EEG  (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)  [uV, large DC offset]
    8-10  accelerometer x/y/z [g]
    11-13 gyroscope x/y/z [deg/s]
    14    battery level [%]
    15    sample counter (resets per recording, +1 per sample)
    16    validation indicator
Train: continuous blocks ``s{sess}_d{day}_p{subj}_{blk}`` with cue timestamps (30 cues, 7 s apart).
Test : 480 pre-cut 7 s trials (1750-1752 x 17) without subject id.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FS = 250
TRIAL_SAMPLES = 1750
EEG_CH = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
CLASSES = [110, 120, 150]
COL_BATTERY, COL_COUNTER = 14, 15

# The cue order used by the paradigm in 190/191 train blocks (the odd one is the same
# sequence with its first cue missing).  Kept for the leak analysis in the EDA only.
CUE_SEQUENCE = [120, 110, 150, 150, 120, 110, 120, 110, 150, 150, 120, 110, 120, 150, 110,
                150, 120, 110, 120, 110, 150, 150, 120, 110, 150, 120, 110, 150, 120, 110]


def _block_id(path: str) -> str:
    return re.sub(r"_(data|label)_time_(series|stamps)\.npy$", "", os.path.basename(path))


def list_blocks(root: str) -> list[str]:
    return sorted({_block_id(p) for p in glob.glob(os.path.join(root, "train", "train", "*.npy"))})


def load_block(root: str, block: str) -> dict:
    d = os.path.join(root, "train", "train", block)
    return dict(
        X=np.load(f"{d}_data_time_series.npy"),
        ts=np.load(f"{d}_data_time_stamps.npy"),
        y=np.load(f"{d}_label_time_series.npy").ravel().astype(int),
        lt=np.load(f"{d}_label_time_stamps.npy"),
    )


@dataclass
class Epochs:
    X: np.ndarray          # (n, 8, T) float32, EEG only, uV
    y: np.ndarray | None   # (n,) int labels in {110,120,150}
    meta: pd.DataFrame     # one row per trial


def load_train_epochs(root: str, n_samples: int = TRIAL_SAMPLES, pad_short: bool = True) -> Epochs:
    """Cut every train block at each cue timestamp into a [cue, cue + 7 s) epoch."""
    X, y, rows = [], [], []
    for b in list_blocks(root):
        blk = load_block(root, b)
        s, d, p, k = b.split("_")
        for j, (lab, t) in enumerate(zip(blk["y"], blk["lt"])):
            i0 = int(np.searchsorted(blk["ts"], t))
            seg = blk["X"][i0:i0 + n_samples, :8]
            if len(seg) < n_samples:
                if not pad_short or len(seg) < n_samples // 2:
                    continue
                seg = np.pad(seg, ((0, n_samples - len(seg)), (0, 0)), mode="edge")
            X.append(seg.T.astype(np.float32))
            y.append(int(lab))
            rows.append(dict(block=b, sess=f"{s}_{d}_{p}", subject=p, session=s, day=d,
                             blk=int(k), pos=j, cue_t=t, counter=blk["X"][i0, COL_COUNTER],
                             battery=float(blk["X"][i0:i0 + n_samples, COL_BATTERY].mean())))
    return Epochs(np.stack(X), np.asarray(y), pd.DataFrame(rows))


def load_test_epochs(root: str, n_samples: int = TRIAL_SAMPLES) -> Epochs:
    X, rows = [], []
    for f in sorted(glob.glob(os.path.join(root, "test", "*.npy"))):
        a = np.load(f)
        X.append(a[:n_samples, :8].T.astype(np.float32))
        rows.append(dict(id=os.path.basename(f)[:-4], counter=a[0, COL_COUNTER],
                         battery=float(a[:, COL_BATTERY].mean()), dc=a[:, :8].mean(0)))
    return Epochs(np.stack(X), None, pd.DataFrame(rows))


def load_application_epochs(root: str, n_samples: int = TRIAL_SAMPLES) -> tuple[Epochs, Epochs]:
    """Extra labelled (180) and unlabelled (60) trials from one further session (counter scrubbed)."""
    lab = pd.read_csv(os.path.join(root, "train_application", "label_application.csv")).set_index("id")["label"]
    out = []
    for sub, pat in [("train_application", "train/train"), ("test_application", "test_application")]:
        X, rows = [], []
        for f in sorted(glob.glob(os.path.join(root, sub, pat, "*.npy"))):
            a = np.load(f)
            i = os.path.basename(f)[:-4]
            seg = a[:n_samples, :8]
            if len(seg) < n_samples:
                seg = np.pad(seg, ((0, n_samples - len(seg)), (0, 0)), mode="edge")
            X.append(seg.T.astype(np.float32))
            rows.append(dict(id=i, battery=float(a[:, COL_BATTERY].mean()), dc=a[:, :8].mean(0)))
        meta = pd.DataFrame(rows)
        y = lab.reindex(meta.id).values.astype(int) if sub == "train_application" else None
        out.append(Epochs(np.stack(X), y, meta))
    return out[0], out[1]


def reconstruct_test_blocks(meta: pd.DataFrame, drift_scale: np.ndarray | None = None,
                            max_dc_dist: float = 15.0, max_batt_diff: float = 7.0) -> pd.DataFrame:
    """Group test trials into recording blocks using the device sample counter.

    Two trials belong to the same block when their counters differ by 1 or 2 trial lengths
    (+-15 samples), their per-channel DC offsets are close and their battery levels agree.
    Returns ``meta`` with ``blk`` (block id), ``pos`` (0-29 order within block) and
    ``tsess`` (recording session guessed from battery level).
    """
    if drift_scale is None:  # typical within-block std of 7 s DC means, per channel (uV)
        drift_scale = np.array([498, 620, 613, 606, 768, 801, 865, 463.0])
    m = meta.sort_values("counter").reset_index(drop=True)
    dc, c, batt = np.stack(m.dc.values), m.counter.values, m.battery.values
    parent = list(range(len(m)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            dcnt = c[j] - c[i]
            if dcnt > 2 * TRIAL_SAMPLES + 100:
                break
            k = round(dcnt / TRIAL_SAMPLES)
            if k in (1, 2) and abs(dcnt - TRIAL_SAMPLES * k) <= 15:
                dist = np.sqrt((((dc[j] - dc[i]) / drift_scale) ** 2).sum())
                if dist < max_dc_dist and abs(batt[j] - batt[i]) <= max_batt_diff:
                    parent[find(j)] = find(i)
    m["blk"] = pd.factorize(np.array([find(i) for i in range(len(m))]))[0]
    m["pos"] = m.groupby("blk").cumcount()
    return m
