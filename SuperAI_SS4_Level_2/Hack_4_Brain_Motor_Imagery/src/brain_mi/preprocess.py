"""Filtering, resampling, alignment and windowing.

Every trial (train, application, test) goes through the *same* per-trial path so the three
sets share one distribution: demean -> reflect-pad -> high-pass -> notch -> band-pass -> crop
-> resample.  Filtered 7 s epochs are cached; alignment and windowing are cheap and run per
experiment.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos

from .config import Config
from .data import (CLASSES, CUE_SEQUENCE, FS, Epochs, load_application_epochs, load_test_epochs,
                   load_train_epochs, reconstruct_test_blocks)

log = logging.getLogger("brain_mi.preprocess")


# ----------------------------------------------------------------------------- filtering
def _sos_chain(cfg: Config):
    p = cfg.preprocess
    chain = []
    if p.highpass_hz and p.highpass_hz > 0:
        chain.append(butter(p.filter_order, p.highpass_hz, btype="high", fs=FS, output="sos"))
    for f0 in p.notch_hz or []:
        b, a = iirnotch(f0, Q=30, fs=FS)
        chain.append(tf2sos(b, a))
    lo, hi = p.bandpass_hz
    chain.append(butter(p.filter_order, [lo, hi], btype="band", fs=FS, output="sos"))
    return chain


def filter_trials(X: np.ndarray, cfg: Config) -> np.ndarray:
    """X: (n, C, T) raw uV -> filtered + resampled (n, C, T')."""
    pad = int(cfg.preprocess.pad_seconds * FS)
    Y = X - X.mean(axis=2, keepdims=True)
    Y = np.pad(Y, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    for sos in _sos_chain(cfg):
        Y = sosfiltfilt(sos, Y, axis=-1)
    Y = Y[:, :, pad:-pad]
    if cfg.preprocess.resample_hz and cfg.preprocess.resample_hz != FS:
        g = np.gcd(FS, cfg.preprocess.resample_hz)
        Y = resample_poly(Y, cfg.preprocess.resample_hz // g, FS // g, axis=-1)
    return Y.astype(np.float32)


# ----------------------------------------------------------------------------- cache
@dataclass
class Dataset:
    """All three sets after filtering, before alignment/windowing."""
    train: Epochs
    app: Epochs           # train_application (labelled, one unseen session)
    test: Epochs
    fs: int

    @property
    def test_oracle(self) -> np.ndarray:
        """Cue-order labels of the test set recovered in the EDA (leak). Reporting only."""
        return self.test.meta["oracle"].values


def _cache_key(cfg: Config) -> str:
    p = cfg.preprocess
    payload = dict(hp=p.highpass_hz, notch=p.notch_hz, bp=p.bandpass_hz, order=p.filter_order,
                   pad=p.pad_seconds, fs=p.resample_hz)
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def _meta(arr) -> pd.DataFrame:
    return pd.read_json(io.StringIO(str(arr)), orient="split")


def load_dataset(cfg: Config) -> Dataset:
    root = str(cfg.resolve(cfg.data.root))
    cache = cfg.resolve(cfg.data.cache_dir) / f"filtered_{_cache_key(cfg)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        tr = Epochs(z["Xtr"], z["ytr"], _meta(z["Mtr"]))
        ap = Epochs(z["Xap"], z["yap"], _meta(z["Map"]))
        te = Epochs(z["Xte"], None, _meta(z["Mte"]))
        log.info("loaded filtered epochs from %s", cache.name)
    else:
        log.info("filtering raw epochs (hp %.1f, notch %s, band %s, %d Hz) ...", cfg.preprocess.highpass_hz,
                 cfg.preprocess.notch_hz, cfg.preprocess.bandpass_hz, cfg.preprocess.resample_hz)
        tr = load_train_epochs(root)
        ap, _ = load_application_epochs(root)
        te = load_test_epochs(root)
        # test block reconstruction (counter) -> session guess + oracle labels
        R = reconstruct_test_blocks(te.meta)
        R["oracle"] = np.array(CUE_SEQUENCE)[R.pos.values]
        R["tsess"] = np.where(R.battery < 70, "A", "B")
        te.meta = te.meta.merge(R[["id", "blk", "pos", "oracle", "tsess"]], on="id", how="left")
        for e in (tr, ap, te):
            e.X = filter_trials(e.X, cfg)
        for e in (tr, ap, te):
            e.meta = e.meta.drop(columns=[c for c in ("dc",) if c in e.meta.columns])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, Xtr=tr.X, ytr=tr.y, Mtr=tr.meta.to_json(orient="split"),
                            Xap=ap.X, yap=ap.y, Map=ap.meta.to_json(orient="split"),
                            Xte=te.X, Mte=te.meta.to_json(orient="split"))
        log.info("cached -> %s", cache.name)
    return Dataset(tr, ap, te, cfg.preprocess.resample_hz or FS)


# ----------------------------------------------------------------------------- alignment
def _mean_cov(X: np.ndarray) -> np.ndarray:
    return np.mean([x @ x.T / x.shape[1] for x in X], axis=0)


def _inv_sqrt(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(w ** -0.5) @ V.T


def _riemann_mean(X: np.ndarray) -> np.ndarray:
    from pyriemann.utils.mean import mean_riemann
    covs = np.stack([x @ x.T / x.shape[1] for x in X])
    return mean_riemann(covs)


def align_groups(X: np.ndarray, groups: np.ndarray, method: str, ref_X: np.ndarray | None = None) -> np.ndarray:
    """Whiten every trial by the (arithmetic|Riemannian) mean covariance of its group.

    ref_X: trials used to compute the reference (defaults to X itself, e.g. full 7 s trials
    while X is the cropped window).
    """
    if method == "none":
        return X
    out = np.empty_like(X)
    R = X if ref_X is None else ref_X
    for g in np.unique(groups):
        i = groups == g
        C = _riemann_mean(R[i]) if method == "riemann" else _mean_cov(R[i])
        out[i] = np.einsum("ij,njt->nit", _inv_sqrt(C), X[i])
    return out


def test_groups(ds: Dataset, cfg: Config, X_for_cluster: np.ndarray | None = None) -> np.ndarray:
    """Group test trials into recording sessions for unsupervised alignment."""
    how = cfg.align.test_group
    if how == "global":
        return np.zeros(len(ds.test.X), dtype=int)
    if how == "counter":
        return pd.factorize(ds.test.meta.tsess.values)[0]
    if how == "cluster":
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.cluster import KMeans
        Xc = ds.test.X if X_for_cluster is None else X_for_cluster
        T = TangentSpace(metric="riemann").fit_transform(Covariances("oas").fit_transform(Xc))
        lab = KMeans(cfg.align.n_clusters, n_init=20, random_state=0).fit_predict(T)
        truth = pd.factorize(ds.test.meta.tsess.values)[0]
        purity = pd.crosstab(lab, truth).max(axis=1).sum() / len(lab)
        log.info("test covariance clustering: k=%d, sizes=%s, purity vs counter-sessions=%.3f",
                 cfg.align.n_clusters, np.bincount(lab).tolist(), purity)
        return lab
    raise ValueError(how)


# ----------------------------------------------------------------------------- windowing
def window(X: np.ndarray, cfg: Config, fs: int) -> np.ndarray:
    t0, t1 = cfg.preprocess.window_s
    return X[:, :, int(t0 * fs):int(t1 * fs)]


def select_channels(X: np.ndarray, cfg: Config) -> np.ndarray:
    ch = cfg.preprocess.channels
    return X if not ch else X[:, ch, :]


def clip_robust(X: np.ndarray, sigma: float) -> np.ndarray:
    if not sigma or sigma <= 0:
        return X
    s = 1.4826 * np.median(np.abs(X), axis=(0, 2), keepdims=True)
    return np.clip(X, -sigma * s, sigma * s)


@dataclass
class Prepared:
    Xtr: np.ndarray; ytr: np.ndarray; Mtr: pd.DataFrame
    Xap: np.ndarray; yap: np.ndarray
    Xte: np.ndarray; yte_oracle: np.ndarray; Mte: pd.DataFrame
    fs: int


def prepare(cfg: Config, ds: Dataset | None = None) -> Prepared:
    """Align (per group) -> window -> channel subset -> clip, for all three sets."""
    ds = ds or load_dataset(cfg)
    fs = ds.fs
    a = cfg.align
    gtr = ds.train.meta[a.train_group].values
    gap = np.zeros(len(ds.app.X), dtype=int)
    gte = test_groups(ds, cfg)

    def run(X, groups):
        full = X
        win = window(X, cfg, fs)
        ref = full if a.cov_window == "full" else win
        out = align_groups(win, groups, a.method, ref_X=ref)
        out = select_channels(out, cfg)
        return clip_robust(out, cfg.preprocess.clip_sigma).astype(np.float32)

    Xtr, Xap, Xte = run(ds.train.X, gtr), run(ds.app.X, gap), run(ds.test.X, gte)
    log.info("prepared: train %s | app %s | test %s | align=%s by %s / test=%s | window %s s",
             Xtr.shape, Xap.shape, Xte.shape, a.method, a.train_group, a.test_group, cfg.preprocess.window_s)
    return Prepared(Xtr, ds.train.y, ds.train.meta, Xap, ds.app.y, Xte, ds.test_oracle, ds.test.meta, fs)
