"""Viterbi smoothing over a night.

A classifier that scores each epoch on its own will happily emit a single REM epoch in the
middle of wake. Real hypnograms do not do that: stages persist for minutes at a time. The
epoch-to-epoch transition matrix estimated from the training hypnograms is a free, very
strong prior, and decoding the whole night through it fixes exactly the isolated flips that
cost macro F1 on the rare REM class.
"""
from __future__ import annotations

import numpy as np

from . import CLASSES

N = len(CLASSES)
EPS = 1e-12


def transition_matrix(ys: list[np.ndarray], smoothing: float = 1.0) -> np.ndarray:
    A = np.full((N, N), smoothing)
    for y in ys:
        np.add.at(A, (y[:-1].astype(int), y[1:].astype(int)), 1.0)
    return A / A.sum(1, keepdims=True)


def class_prior(ys: list[np.ndarray]) -> np.ndarray:
    counts = np.bincount(np.concatenate(ys).astype(int), minlength=N).astype(float)
    return counts / counts.sum()


def viterbi(proba: np.ndarray, A: np.ndarray, prior: np.ndarray,
            self_bias: float = 0.0, prior_power: float = 1.0) -> np.ndarray:
    """Most likely stage sequence given per-epoch posteriors.

    Posteriors already carry the training prior, so they are divided by it before being
    used as emission likelihoods — otherwise the prior is applied twice.
    """
    logB = np.log(proba + EPS) - prior_power * np.log(prior + EPS)
    logA = np.log(A + EPS)
    if self_bias:
        logA = logA + self_bias * np.eye(N)
    T = len(proba)
    dp = np.full((T, N), -np.inf)
    bp = np.zeros((T, N), dtype=np.int8)
    dp[0] = np.log(prior + EPS) + logB[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + logA
        bp[t] = scores.argmax(0)
        dp[t] = scores.max(0) + logB[t]
    path = np.zeros(T, dtype=np.int8)
    path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def decode(probas: list[np.ndarray], A: np.ndarray, prior: np.ndarray, cfg) -> list[np.ndarray]:
    if cfg.method == "none":
        return [p.argmax(1).astype(np.int8) for p in probas]
    if cfg.method != "hmm":
        raise ValueError(f"unknown smooth.method {cfg.method!r}")
    return [viterbi(p, A, prior, cfg.self_bias, cfg.prior_power) for p in probas]
