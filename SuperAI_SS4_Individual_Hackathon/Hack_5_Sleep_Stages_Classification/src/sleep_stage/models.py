"""Classifiers over a night of feature vectors.

Every model takes a *list of nights* rather than a flat epoch table, because the whole
point of this rebuild is that an epoch is read in the context of the night around it.
`lgbm` gets that context baked into its columns (lags and rolling windows); the sequence
models get the raw night and learn it.
"""
from __future__ import annotations

import logging

import numpy as np

from . import CLASSES
from .config import Config

log = logging.getLogger("sleep_stage.models")
N_CLASSES = len(CLASSES)


class NightModel:
    def fit(self, mats: list[np.ndarray], ys: list[np.ndarray]) -> "NightModel":
        raise NotImplementedError

    def predict_proba(self, mats: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError


def class_weights(ys: list[np.ndarray]) -> np.ndarray:
    y = np.concatenate(ys)
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    w = len(y) / (N_CLASSES * np.maximum(counts, 1.0))
    return w


# --------------------------------------------------------------------------- LightGBM

class LgbmModel(NightModel):
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.model = None

    def fit(self, mats, ys):
        import lightgbm as lgb
        m = self.cfg.model
        X = np.concatenate(mats).astype(np.float32)
        y = np.concatenate(ys)
        weight = None
        if m.class_weight == "balanced":
            weight = class_weights(ys)[y]
        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=N_CLASSES, n_estimators=m.n_estimators,
            learning_rate=m.learning_rate, num_leaves=m.num_leaves,
            min_child_samples=m.min_child_samples, subsample=m.subsample, subsample_freq=1,
            colsample_bytree=m.colsample_bytree, reg_lambda=m.reg_lambda,
            random_state=self.seed, n_jobs=self.cfg.data.n_jobs, verbose=-1,
        )
        self.model.fit(X, y, sample_weight=weight)
        return self

    def predict_proba(self, mats):
        return [self.model.predict_proba(M.astype(np.float32)) for M in mats]

    @property
    def feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_


# --------------------------------------------------------------------------- sequence nets

def _torch():
    import torch
    torch.set_num_threads(max(1, min(16, __import__("os").cpu_count() or 4)))
    return torch


def resolve_device(name: str):
    """`auto` picks CUDA when it is actually usable, otherwise CPU."""
    torch = _torch()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        log.warning("model.device=%s requested but no CUDA device is visible — falling back to cpu", name)
        name = "cpu"
    return torch.device(name)


def _build_net(kind: str, d_in: int, cfg) -> "object":
    import torch
    from torch import nn

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            h = cfg.hidden
            self.stem = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.dropout))
            if kind in ("bilstm", "gru"):
                rnn = nn.LSTM if kind == "bilstm" else nn.GRU
                self.body = rnn(h, h, num_layers=cfg.layers, batch_first=True, bidirectional=True,
                                dropout=cfg.dropout if cfg.layers > 1 else 0.0)
                self.head = nn.Linear(2 * h, N_CLASSES)
                self.kind = "rnn"
            elif kind == "tcn":
                self.blocks = nn.ModuleList()
                for d in cfg.tcn_dilations:
                    pad = d * (cfg.tcn_kernel - 1) // 2
                    self.blocks.append(nn.Sequential(
                        nn.Conv1d(h, h, cfg.tcn_kernel, dilation=d, padding=pad),
                        nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(cfg.dropout),
                        nn.Conv1d(h, h, 1), nn.BatchNorm1d(h), nn.GELU(),
                    ))
                self.head = nn.Linear(h, N_CLASSES)
                self.kind = "tcn"
            else:
                raise ValueError(f"unknown model.name {kind!r}")

        def forward(self, x):                     # x: (B, T, D)
            z = self.stem(x)
            if self.kind == "rnn":
                z, _ = self.body(z)
            else:
                z = z.transpose(1, 2)
                for blk in self.blocks:
                    z = z + blk(z)
                z = z.transpose(1, 2)
            return self.head(z)                   # (B, T, C)

    return Seq()


class SequenceModel(NightModel):
    """BiLSTM / GRU / dilated TCN over the whole night, one softmax per 30 s epoch."""

    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.mean = self.std = None
        self.nets: list = []
        self.device = None

    # -- normalisation across subjects (per-subject scaling already happened in context.py)
    def _standardise(self, mats):
        X = np.concatenate(mats)
        self.mean = X.mean(0, keepdims=True).astype(np.float32)
        self.std = (X.std(0, keepdims=True) + 1e-6).astype(np.float32)

    def _prep(self, mats):
        return [((M - self.mean) / self.std).astype(np.float32) for M in mats]

    def fit(self, mats, ys):
        torch = _torch()
        from torch import nn

        m = self.cfg.model
        self.device = resolve_device(m.device)
        self._standardise(mats)
        Xs = self._prep(mats)
        d_in = Xs[0].shape[1]
        w = torch.tensor(class_weights(ys) if m.loss_class_weight == "balanced"
                         else np.ones(N_CLASSES), dtype=torch.float32).to(self.device)
        log.info("sequence model %s on %s (%d nights, %d features)", m.name, self.device, len(Xs), d_in)

        self.nets = []
        for s in range(m.n_seeds):
            torch.manual_seed(self.seed + 1000 * s)
            rng = np.random.default_rng(self.seed + 1000 * s)
            net = _build_net(m.name, d_in, m).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=m.lr, weight_decay=m.weight_decay)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=m.lr, total_steps=m.epochs *
                                                        max(1, int(np.ceil(len(Xs) / m.batch_size))))
            lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=m.label_smoothing)
            net.train()
            for ep in range(m.epochs):
                order = rng.permutation(len(Xs))
                total = 0.0
                for i in range(0, len(order), m.batch_size):
                    idx = order[i:i + m.batch_size]
                    xb, yb, mask = _batch([Xs[j] for j in idx], [ys[j] for j in idx], m.crop_epochs, rng)
                    xb, yb, mask = xb.to(self.device), yb.to(self.device), mask.to(self.device)
                    logits = net(xb)
                    loss = lossf(logits[mask], yb[mask])
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step(); sched.step()
                    total += loss.item() * len(idx)
                if (ep + 1) % 10 == 0 or ep == m.epochs - 1:
                    log.debug("  seed %d epoch %2d/%d loss %.4f", s, ep + 1, m.epochs, total / len(order))
                if ep >= m.epochs - m.avg_last:
                    self.nets.append(_snapshot(net))   # kept on CPU so a run stays loadable anywhere
            net.train(False)
        return self

    def predict_proba(self, mats):
        torch = _torch()
        device = self.device or resolve_device(self.cfg.model.device)
        Xs = self._prep(mats)
        out = [np.zeros((len(X), N_CLASSES)) for X in Xs]
        net = _build_net(self.cfg.model.name, Xs[0].shape[1], self.cfg.model).to(device)
        with torch.no_grad():
            for state in self.nets:
                net.load_state_dict(state); net.eval()
                for k, X in enumerate(Xs):
                    logits = net(torch.from_numpy(X)[None].to(device))
                    out[k] += torch.softmax(logits, -1)[0].cpu().numpy() / len(self.nets)
        return out


def _snapshot(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _batch(Xs, ys, crop, rng):
    """Pad a list of nights into (B, T, D) with a boolean mask; optionally random-crop."""
    torch = _torch()
    if crop and crop > 0:
        cut = []
        for X, y in zip(Xs, ys):
            if len(X) > crop:
                s = rng.integers(0, len(X) - crop)
                cut.append((X[s:s + crop], y[s:s + crop]))
            else:
                cut.append((X, y))
        Xs, ys = [c[0] for c in cut], [c[1] for c in cut]
    T = max(len(X) for X in Xs)
    B, D = len(Xs), Xs[0].shape[1]
    xb = np.zeros((B, T, D), dtype=np.float32)
    yb = np.zeros((B, T), dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    for i, (X, y) in enumerate(zip(Xs, ys)):
        xb[i, :len(X)] = X
        yb[i, :len(y)] = y
        mask[i, :len(X)] = True
    return torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(mask)


# --------------------------------------------------------------------------- factory

def build_model(cfg: Config, seed: int) -> NightModel:
    if cfg.model.name == "lgbm":
        return LgbmModel(cfg, seed)
    return SequenceModel(cfg, seed)


def needs_context(cfg: Config) -> bool:
    """Only the flat classifier needs lags and rolling windows as columns."""
    return cfg.model.name == "lgbm"
