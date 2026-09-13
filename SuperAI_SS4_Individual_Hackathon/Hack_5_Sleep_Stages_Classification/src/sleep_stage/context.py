"""Turning per-epoch features into what a model actually sees.

Two things matter far more here than the choice of classifier:

1. *Per-subject normalisation.* Resting heart rate, skin temperature and pulse amplitude
   differ more between two people than between wake and REM inside one person, and the ten
   test subjects are strangers. Every feature is therefore robust-z-scored against the
   median and IQR of its own recording.

2. *Temporal context.* Sleep stages come in cycles: an epoch is far easier to place once
   you can see the half hour around it. Centred rolling statistics and signed lags give a
   flat classifier that context; the sequence models get the whole night instead.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import ContextConfig
from .data import Dataset, Night

log = logging.getLogger("sleep_stage.context")

# Features whose neighbourhood carries the most stage information — the ones that get lags.
CORE_PREFIXES = (
    "hrv_hr", "hrv_rmssd", "hrv_sdnn", "hrv_sd_ratio", "hrv_sampen", "hrv_coverage",
    "frq_lf_hf", "frq_hf_nu", "frq_total", "frq_resp",
    "riiv_rate", "riiv_conc", "riiv_entropy", "rsa_rate", "rsa_conc", "rsa_entropy",
    "ppg_amp", "ppg_rel_resp", "ppg_sp_entropy", "ppg_f_card", "ppg_card_conc",
    "acc_count", "acc_still_frac", "acc_mag_std", "acc_angle_change",
    "tmp_mean", "tmp_slope", "eda_mean", "eda_scr_n",
    "e4_hr_mean", "e4_hr_std", "e4_hr_sdsd", "e4_hr_range", "e4_ibi_changes",
)
CLIP = 8.0


def robust_z(X: np.ndarray) -> np.ndarray:
    """Z-score each column against its own recording, using median/IQR so a few artefact
    epochs cannot rescale the whole night."""
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    scale = np.where(iqr > 1e-6, iqr * 0.7413, 1.0)
    return np.clip((X - med) / scale, -CLIP, CLIP).astype(np.float32)


def _rolling(X: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred rolling mean and std, edge-padded so the output keeps its length."""
    n, d = X.shape
    half = window // 2
    pad = np.pad(X, ((half, window - 1 - half), (0, 0)), mode="edge")
    csum = np.concatenate([np.zeros((1, d)), np.cumsum(pad, axis=0)])
    csq = np.concatenate([np.zeros((1, d)), np.cumsum(pad.astype(np.float64) ** 2, axis=0)])
    mean = (csum[window:] - csum[:-window]) / window
    var = (csq[window:] - csq[:-window]) / window - mean ** 2
    return mean[:n].astype(np.float32), np.sqrt(np.maximum(var[:n], 0.0)).astype(np.float32)


def _shift(X: np.ndarray, k: int) -> np.ndarray:
    """Shift rows by k (positive = look back), edge-padded."""
    out = np.empty_like(X)
    if k > 0:
        out[k:] = X[:-k]
        out[:k] = X[0]
    elif k < 0:
        out[:k] = X[-k:]
        out[k:] = X[-1]
    else:
        out[:] = X
    return out


def _clock(n: int) -> tuple[np.ndarray, list[str]]:
    """Where in the night this epoch sits — sleep pressure and REM density both drift."""
    i = np.arange(n, dtype=np.float32)
    block = np.column_stack([
        i / max(n - 1, 1),                 # relative position
        i * 0.5,                           # minutes from lights-off
        (n - 1 - i) * 0.5,                 # minutes to the end of the recording
        np.full(n, n * 0.5, dtype=np.float32),
        np.sin(2 * np.pi * i / 180.0),     # a 90 min sleep cycle, as a smooth phase pair
        np.cos(2 * np.pi * i / 180.0),
    ]).astype(np.float32)
    return block, ["clk_pos", "clk_min_from_start", "clk_min_to_end", "clk_len_min",
                   "clk_cycle_sin", "clk_cycle_cos"]


def build_night(X: np.ndarray, names: list[str], cfg: ContextConfig,
                with_context: bool) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    out_names: list[str] = []

    Z = robust_z(X) if cfg.per_subject_norm else X.astype(np.float32)
    blocks.append(Z)
    out_names += [f"z_{n}" for n in names]
    if cfg.keep_absolute and cfg.per_subject_norm:
        blocks.append(X.astype(np.float32))
        out_names += [f"abs_{n}" for n in names]

    if with_context:
        for w in cfg.roll_windows:
            mean, std = _rolling(Z, w)
            blocks += [mean, std]
            out_names += [f"rm{w}_{n}" for n in names] + [f"rs{w}_{n}" for n in names]
        core = [i for i, n in enumerate(names) if n.startswith(CORE_PREFIXES)]
        Zc, core_names = Z[:, core], [names[i] for i in core]
        for lag in cfg.lags:
            for sign, tag in ((lag, "lag"), (-lag, "lead")):
                blocks.append(_shift(Zc, sign))
                out_names += [f"{tag}{lag}_{n}" for n in core_names]

    if cfg.clock:
        block, clock_names = _clock(len(X))
        blocks.append(block)
        out_names += clock_names

    return np.nan_to_num(np.column_stack(blocks), copy=False), out_names


def build(ds: Dataset, cfg: ContextConfig, with_context: bool) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """-> (train matrices, test matrices, feature names), one matrix per night."""
    tr, te, names = [], [], []
    for night in ds.train:
        M, names = build_night(night.X, ds.names, cfg, with_context)
        tr.append(M)
    for night in ds.test:
        M, _ = build_night(night.X, ds.names, cfg, with_context)
        te.append(M)
    log.info("design matrix: %d features per epoch (context=%s)", len(names), with_context)
    return tr, te, names


def stack(mats: list[np.ndarray], nights: list[Night]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of nights into (X, y, subject groups) for a per-epoch model."""
    X = np.concatenate(mats, axis=0)
    y = np.concatenate([n.y for n in nights]) if nights[0].y is not None else None
    groups = np.concatenate([np.full(len(m), i) for i, m in enumerate(mats)])
    return X, y, groups
