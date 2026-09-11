"""Model zoo behind one interface: fit(X, y) / predict_proba(X) with X = (n, C, T) float32.

Riemannian pipelines are sklearn; deep nets (braindecode) are wrapped in TorchNet so the CV
loop, hierarchical wrapper and ensembling do not care which one they hold.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data import CLASSES

log = logging.getLogger("brain_mi.models")
DEEP = {"eegnet", "shallow", "conformer", "atcnet", "eegnex"}


# ----------------------------------------------------------------------------- riemannian
class FilterBankTS(BaseEstimator, TransformerMixin):
    """Per-band covariance -> tangent space, concatenated."""

    def __init__(self, bands, fs, estimator="oas"):
        self.bands, self.fs, self.estimator = bands, fs, estimator

    def _filt(self, X, lo, hi):
        hi = min(hi, self.fs / 2 - 1)
        return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=self.fs, output="sos"), X, axis=-1).astype(np.float32)

    def fit(self, X, y=None):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        self.ts_ = []
        for lo, hi in self.bands:
            p = make_pipeline(Covariances(self.estimator), TangentSpace(metric="riemann")).fit(self._filt(X, lo, hi))
            self.ts_.append(p)
        return self

    def transform(self, X):
        return np.concatenate([p.transform(self._filt(X, lo, hi)) for p, (lo, hi) in zip(self.ts_, self.bands)], axis=1)


def _riemann_model(name: str, cfg: Config, fs: int):
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from pyriemann.tangentspace import TangentSpace
    C = cfg.model.lr_C
    lr = LogisticRegression(C=C, max_iter=3000)
    if name == "ts_lr":
        return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), StandardScaler(), lr)
    if name == "fbts_lr":
        return make_pipeline(FilterBankTS(cfg.model.bands, fs), StandardScaler(), lr)
    if name == "csp_lr":
        return make_pipeline(Covariances("oas"), CSP(nfilter=cfg.model.csp_filters, log=True), StandardScaler(), lr)
    raise ValueError(name)


# ----------------------------------------------------------------------------- deep
class TorchNet(BaseEstimator, ClassifierMixin):
    """braindecode model + training loop with EEG augmentations, sklearn-style."""

    REFLECT_PAIRS = [(1, 3), (5, 7)]   # C3<->C4, PO7<->PO8 (indices in the 8-channel montage)

    def __init__(self, name: str, cfg: Config, fs: int, seed: int = 0):
        self.name, self.cfg, self.fs, self.seed = name, cfg, fs, seed

    def _build(self, n_chans, n_times, n_out):
        from braindecode import models as bm
        m = self.cfg.model
        if self.name == "eegnet":
            return bm.EEGNet(n_chans=n_chans, n_outputs=n_out, n_times=n_times, F1=16, D=2, F2=32, drop_prob=min(m.drop_prob, 0.5))
        if self.name == "shallow":
            return bm.ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_out, n_times=n_times, final_conv_length="auto", drop_prob=m.drop_prob)
        if self.name == "conformer":
            return bm.EEGConformer(n_chans=n_chans, n_outputs=n_out, n_times=n_times, final_fc_length="auto", drop_prob=m.drop_prob)
        if self.name == "atcnet":
            return bm.ATCNet(n_chans=n_chans, n_outputs=n_out, n_times=n_times, sfreq=self.fs)
        if self.name == "eegnex":
            return bm.EEGNeX(n_chans=n_chans, n_outputs=n_out, n_times=n_times)
        raise ValueError(self.name)

    def _augment(self, xb, yb, rng):
        import torch
        m = self.cfg.model
        n = len(xb)
        if m.aug_scale > 0:
            xb = xb * (1 + m.aug_scale * torch.randn(n, 1, 1))
        if m.aug_noise > 0:
            xb = xb + m.aug_noise * torch.randn_like(xb)
        if m.aug_shift > 0:
            xb = torch.roll(xb, int(rng.integers(-m.aug_shift, m.aug_shift + 1)), dims=2)
        if m.aug_channel_dropout > 0:
            keep = (torch.rand(n, xb.shape[1], 1) > m.aug_channel_dropout).float()
            xb = xb * keep
        if m.aug_reflect and self.reflect_ok_:
            flip = torch.rand(n) < 0.5
            if flip.any():
                xf = xb[flip].clone()
                for a, b in self.REFLECT_PAIRS:
                    xf[:, [a, b]] = xf[:, [b, a]]
                xb[flip] = xf
                yf = yb[flip].clone()
                l, r = self.class_index_[110], self.class_index_[120]
                yf[yb[flip] == l] = r
                yf[yb[flip] == r] = l
                yb = yb.clone(); yb[flip] = yf
        if m.aug_freq_shift_hz > 0:
            from braindecode.augmentation import FrequencyShift
            xb = FrequencyShift(probability=1.0, sfreq=self.fs, max_delta_freq=m.aug_freq_shift_hz)(xb)
        return xb, yb

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        m = self.cfg.model
        self.classes_ = np.unique(y)
        self.class_index_ = {c: i for i, c in enumerate(self.classes_)}
        self.reflect_ok_ = (110 in self.class_index_ and 120 in self.class_index_
                            and (self.cfg.preprocess.channels in (None, [], list(range(8)))))
        yi = np.array([self.class_index_[c] for c in y])
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net_ = self._build(X.shape[1], X.shape[2], len(self.classes_)).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=m.lr, weight_decay=m.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, m.epochs)
        Xt, yt = torch.tensor(X), torch.tensor(yi)
        for ep in range(m.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), m.batch_size):
                b = perm[i:i + m.batch_size]
                xb, yb = self._augment(Xt[b].clone(), yt[b], rng)
                xb, yb = xb.to(dev), yb.to(dev)
                if m.mixup_alpha > 0:
                    lam = float(rng.beta(m.mixup_alpha, m.mixup_alpha)); j = torch.randperm(len(xb), device=dev)
                    out = self.net_(lam * xb + (1 - lam) * xb[j])
                    loss = lam * F.cross_entropy(out, yb, label_smoothing=m.label_smoothing) + \
                        (1 - lam) * F.cross_entropy(out, yb[j], label_smoothing=m.label_smoothing)
                else:
                    loss = F.cross_entropy(self.net_(xb), yb, label_smoothing=m.label_smoothing)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        self.device_ = dev
        return self

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                out.append(torch.softmax(self.net_(torch.tensor(X[i:i + 256]).to(self.device_)), 1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class SeedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        self.classes_ = self.members[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ----------------------------------------------------------------------------- hierarchical
class Hierarchical(BaseEstimator, ClassifierMixin):
    """Stage 1: rest (150) vs MI.  Stage 2: 110 vs 120 on MI trials only."""

    def __init__(self, base_factory):
        self.base_factory = base_factory

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(CLASSES)
        self.stage1_ = self.base_factory().fit(X, (y == 150).astype(int))
        mi = y != 150
        self.stage2_ = self.base_factory().fit(X[mi], y[mi])
        return self

    def predict_proba(self, X):
        p1 = self.stage1_.predict_proba(X)            # columns: [MI, rest]
        p_rest = p1[:, list(self.stage1_.classes_).index(1)]
        p2 = self.stage2_.predict_proba(X)            # columns in stage2_.classes_ order (110, 120)
        c2 = list(self.stage2_.classes_)
        out = np.zeros((len(X), 3))
        out[:, 0] = (1 - p_rest) * p2[:, c2.index(110)]
        out[:, 1] = (1 - p_rest) * p2[:, c2.index(120)]
        out[:, 2] = p_rest
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ----------------------------------------------------------------------------- factory
def build_model(cfg: Config, fs: int, seed: int = 0):
    name = cfg.model.name

    def base():
        if name in DEEP:
            if cfg.model.n_seeds > 1:
                return SeedEnsemble([TorchNet(name, cfg, fs, seed=seed + k) for k in range(cfg.model.n_seeds)])
            return TorchNet(name, cfg, fs, seed=seed)
        return _riemann_model(name, cfg, fs)

    return Hierarchical(base) if cfg.model.hierarchical else base()


def proba_in_class_order(model, X) -> np.ndarray:
    """predict_proba with columns forced to CLASSES order."""
    p = model.predict_proba(X)
    cols = list(model.classes_)
    return np.stack([p[:, cols.index(c)] for c in CLASSES], axis=1)
