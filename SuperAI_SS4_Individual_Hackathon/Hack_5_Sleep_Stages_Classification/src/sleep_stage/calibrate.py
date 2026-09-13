"""Pick the decision rule, not just the model.

The competition is scored on macro F1, which weights REM — 9 % of the epochs — as heavily
as NREM. Argmax of a well-fitted posterior is the wrong rule for that objective: it trades
away REM recall to buy NREM precision that macro F1 does not reward. So after
cross-validation, one bias per class is fitted on the out-of-fold probabilities by direct
coordinate ascent on macro F1 *through the same Viterbi decoder used at predict time*, and
that bias travels with the run.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import f1_score

from . import CLASSES
from .smooth import decode

log = logging.getLogger("sleep_stage.calibrate")
N = len(CLASSES)


def apply_bias(probas: list[np.ndarray], bias: np.ndarray) -> list[np.ndarray]:
    """Reweight class probabilities by exp(bias) and renormalise."""
    w = np.exp(bias - bias.max())
    out = []
    for p in probas:
        q = p * w
        out.append(q / (q.sum(1, keepdims=True) + 1e-12))
    return out


def fit_bias(probas: list[np.ndarray], ys: list[np.ndarray], A: np.ndarray, prior: np.ndarray,
             smooth_cfg, grid: np.ndarray | None = None, rounds: int = 3) -> tuple[np.ndarray, float]:
    """Coordinate ascent on macro F1 over one log-weight per class."""
    grid = np.linspace(-1.2, 1.2, 25) if grid is None else grid
    y_true = np.concatenate(ys)

    def macro(bias: np.ndarray) -> float:
        pred = decode(apply_bias(probas, bias), A, prior, smooth_cfg)
        return float(f1_score(y_true, np.concatenate(pred), average="macro", zero_division=0))

    bias = np.zeros(N)
    best = macro(bias)
    log.info("calibration start: macro-F1 %.4f at bias %s", best, np.round(bias, 2).tolist())
    for r in range(rounds):
        improved = False
        for c in range(N):
            for v in grid:
                trial = bias.copy(); trial[c] = v
                s = macro(trial)
                if s > best + 1e-5:
                    best, bias, improved = s, trial, True
        if not improved:
            break
    bias = bias - bias.mean()
    log.info("calibration done after %d round(s): macro-F1 %.4f at bias %s",
             r + 1, best, {c: round(float(v), 2) for c, v in zip(CLASSES, bias)})
    return bias, best
