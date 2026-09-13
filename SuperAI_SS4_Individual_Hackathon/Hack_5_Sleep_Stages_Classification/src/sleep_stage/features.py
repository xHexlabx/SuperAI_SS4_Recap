"""Per-epoch features for a whole night of Empatica E4 signal.

The organiser resampled every column to 64 Hz, so one 30 s epoch is exactly 1920 rows.
Everything here is computed once per *recording* (not per epoch) and only then cut into
epochs — beat detection in particular needs the continuous signal, and the frequency-domain
HRV block needs a five-minute context that straddles epoch boundaries.

Blocks produced, in order:
  hrv_*   time-domain NN statistics from our own BVP peak detection
  frq_*   frequency-domain HRV (VLF/LF/HF) from the NN series resampled to 4 Hz
  ppg_*   pulse amplitude / morphology / quality from BVP itself
  acc_*   actigraphy: movement, posture, stillness
  tmp_*   skin temperature
  eda_*   electrodermal level and storm count
  e4_*    the HR and IBI columns the device already provides
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import FeatureConfig

COLUMNS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]
ACC_COUNTS_PER_G = 64.0
EPS = 1e-8


# --------------------------------------------------------------------------- helpers

def _butter(x: np.ndarray, band, fs: float, btype: str, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    wn = np.clip(np.atleast_1d(np.asarray(band, dtype=float)) / nyq, 1e-6, 0.999)
    b, a = sps.butter(order, wn if wn.size > 1 else wn[0], btype=btype)
    return sps.filtfilt(b, a, x)


def _agg(block: np.ndarray, prefix: str, out: dict, slope: bool = False) -> None:
    """mean/std/min/max (+ optional slope) of a (n_epochs, n_samples) block."""
    out[f"{prefix}_mean"] = block.mean(1)
    out[f"{prefix}_std"] = block.std(1)
    out[f"{prefix}_min"] = block.min(1)
    out[f"{prefix}_max"] = block.max(1)
    if slope:
        n = block.shape[1]
        t = np.linspace(-1.0, 1.0, n)
        out[f"{prefix}_slope"] = (block * t).mean(1) / (t * t).mean()


def _safe(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _sampen(x: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
    """Sample entropy of a short series; returns 0.0 when the series is too short."""
    n = x.size
    if n < m + 4:
        return 0.0
    r = r_scale * x.std()
    if r <= EPS:
        return 0.0

    def _count(k: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, k)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(-1)
        np.fill_diagonal(d, np.inf)
        return int((d <= r).sum())

    a, b = _count(m + 1), _count(m)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


# --------------------------------------------------------------------------- beats

def cardiac_frequency(bvp: np.ndarray, n_epochs: int, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Dominant pulse frequency per epoch, and how sharply it stands out.

    Wrist PPG during sleep is noisy enough that beat detection alone drifts (it doubles on
    dicrotic notches and skips on motion). The 30 s spectrum is far steadier, so it is used
    both as a feature in its own right and as the gate that tells the beat detector which
    intervals are physiologically possible for *this* epoch.
    """
    fs = float(cfg.fs)
    x = _butter(bvp, [0.5, 8.0], fs, "bandpass").reshape(n_epochs, cfg.epoch_samples)
    f, pxx = sps.welch(x, fs=fs, nperseg=cfg.epoch_samples, nfft=8192, axis=1)
    band = (f >= 0.55) & (f <= 2.5)
    fb, pb = f[band], pxx[:, band]
    i = np.clip(pb.argmax(1), 1, pb.shape[1] - 2)
    rows = np.arange(n_epochs)
    y0, y1, y2 = pb[rows, i - 1], pb[rows, i], pb[rows, i + 1]
    shift = np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - EPS), -1.0, 1.0)   # sub-bin peak
    f_card = fb[i] + shift * (fb[1] - fb[0])
    conc = pb.max(1) / (pb.sum(1) + EPS)                   # peak sharpness = pulse quality
    return f_card, conc


def detect_beats(bvp: np.ndarray, f_card: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Systolic peaks: coarse detection on the fundamental, refined on the full pulse band.

    Band-passing to 0.5-3.5 Hz removes the dicrotic notch that otherwise doubles the beat
    count; the refinement step puts each peak back on the true systolic maximum so the NN
    timing is not biased by the narrow filter.
    """
    fs = float(cfg.fs)
    narrow = _butter(bvp, [cfg.bvp_band[0], 3.5], fs, "bandpass")
    narrow = narrow / (np.median(np.abs(narrow)) * 1.4826 + EPS)
    period = 1.0 / max(np.median(f_card), 0.6)
    distance = max(1, int(0.6 * period * fs))
    peaks, _ = sps.find_peaks(narrow, distance=distance, prominence=0.4)

    wide = _butter(bvp, cfg.bvp_band, fs, "bandpass")
    wide = wide / (np.median(np.abs(wide)) * 1.4826 + EPS)
    span = max(1, int(0.08 * fs))
    lo = np.clip(peaks - span, 0, wide.size - 1)
    hi = np.clip(peaks + span + 1, 1, wide.size)
    refined = np.array([l + int(np.argmax(wide[l:h])) for l, h in zip(lo, hi)]) if peaks.size else peaks
    refined = np.unique(refined)
    if refined.size:                                       # drop points the wide band does not peak at
        inner = refined[(refined > 0) & (refined < wide.size - 1)]
        refined = inner[(wide[inner] >= wide[inner - 1]) & (wide[inner] >= wide[inner + 1])]
    prom = sps.peak_prominences(wide, refined)[0] if refined.size else np.zeros(0)
    return refined, prom


def clean_nn(peaks: np.ndarray, cfg: FeatureConfig,
             f_card: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NN intervals in seconds -> (time of the closing beat, interval, mask of kept beats)."""
    fs = float(cfg.fs)
    if peaks.size < 3:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    t = peaks / fs
    nn = np.diff(t)
    nn_t = t[1:]
    lo, hi = cfg.nn_range_s
    keep = (nn >= lo) & (nn <= hi)
    if f_card is not None and f_card.size:                  # reject beats the 30 s spectrum rules out
        epoch = np.clip((nn_t * fs / cfg.epoch_samples).astype(int), 0, f_card.size - 1)
        expect = 1.0 / np.maximum(f_card[epoch], 0.5)
        keep &= (nn > 0.65 * expect) & (nn < 1.5 * expect)
    if keep.sum() >= 5:                                     # ectopic rejection against a local median
        med = np.full(nn.size, np.median(nn[keep]))
        k = 11
        if nn.size >= k:
            pad = np.pad(nn, (k // 2, k // 2), mode="edge")
            med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=-1)
        keep &= np.abs(nn - med) <= cfg.nn_ectopic_tol * med
    return nn_t[keep], nn[keep], keep


# --------------------------------------------------------------------------- blocks

def _hrv_time_domain(nn_t: np.ndarray, nn: np.ndarray, n_epochs: int, cfg: FeatureConfig) -> dict:
    """Time-domain HRV per epoch, from the NN beats that close inside that epoch."""
    edges = np.arange(n_epochs + 1) * (cfg.epoch_samples / cfg.fs)
    idx = np.searchsorted(nn_t, edges)
    names = ["count", "coverage", "mean", "median", "sdnn", "rmssd", "pnn50", "pnn20", "cv",
             "iqr", "min", "max", "sd1", "sd2", "sd_ratio", "hr", "hr_std", "sampen"]
    out = {f"hrv_{n}": np.zeros(n_epochs) for n in names}
    for e in range(n_epochs):
        seg = nn[idx[e]:idx[e + 1]]
        if seg.size < 3:
            continue
        d = np.diff(seg)
        rmssd = float(np.sqrt((d * d).mean()))
        sd1 = rmssd / np.sqrt(2.0)
        sdnn = float(seg.std(ddof=1))
        sd2 = float(np.sqrt(max(2.0 * sdnn * sdnn - sd1 * sd1, 0.0)))
        hr = 60.0 / seg
        out["hrv_count"][e] = seg.size
        out["hrv_coverage"][e] = seg.sum() / (cfg.epoch_samples / cfg.fs)
        out["hrv_mean"][e] = seg.mean()
        out["hrv_median"][e] = np.median(seg)
        out["hrv_sdnn"][e] = sdnn
        out["hrv_rmssd"][e] = rmssd
        out["hrv_pnn50"][e] = float((np.abs(d) > 0.05).mean())
        out["hrv_pnn20"][e] = float((np.abs(d) > 0.02).mean())
        out["hrv_cv"][e] = sdnn / (seg.mean() + EPS)
        out["hrv_iqr"][e] = float(np.subtract(*np.percentile(seg, [75, 25])))
        out["hrv_min"][e] = seg.min()
        out["hrv_max"][e] = seg.max()
        out["hrv_sd1"][e] = sd1
        out["hrv_sd2"][e] = sd2
        out["hrv_sd_ratio"][e] = sd1 / (sd2 + EPS)
        out["hrv_hr"][e] = hr.mean()
        out["hrv_hr_std"][e] = hr.std()
        out["hrv_sampen"][e] = _sampen(seg)
    return out


def _hrv_frequency(nn_t: np.ndarray, nn: np.ndarray, n_epochs: int, cfg: FeatureConfig) -> dict:
    """VLF/LF/HF from the NN series resampled to 4 Hz, over a window centred on each epoch.

    HRV frequency bands need minutes of context, so `hrv_window_s` (300 s by default)
    deliberately straddles epoch boundaries — the same context every published HRV
    pipeline uses, and the sleep stage of a single epoch is never read from it.
    """
    epoch_s = cfg.epoch_samples / cfg.fs
    names = ["vlf", "lf", "hf", "total", "lf_nu", "hf_nu", "lf_hf", "hf_peak", "resp"]
    out = {f"frq_{n}": np.zeros(n_epochs) for n in names}
    if nn.size < 30:
        return out

    fs_i = cfg.hrv_interp_hz
    grid = np.arange(0.0, n_epochs * epoch_s, 1.0 / fs_i)
    series = np.interp(grid, nn_t, nn)
    series = series - series.mean()

    half = int(cfg.hrv_window_s * fs_i / 2)
    nperseg = min(512, max(128, half))
    for e in range(n_epochs):
        c = int((e + 0.5) * epoch_s * fs_i)
        seg = series[max(0, c - half): c + half]
        if seg.size < nperseg:
            continue
        f, pxx = sps.welch(seg, fs=fs_i, nperseg=nperseg, noverlap=nperseg // 2)
        band = lambda lo, hi: float(np.trapezoid(pxx[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]))
        vlf, lf, hf = band(0.003, 0.04), band(0.04, 0.15), band(0.15, 0.40)
        tot = vlf + lf + hf
        hfm = (f >= 0.15) & (f < 0.40)
        out["frq_vlf"][e] = np.log1p(vlf * 1e3)
        out["frq_lf"][e] = np.log1p(lf * 1e3)
        out["frq_hf"][e] = np.log1p(hf * 1e3)
        out["frq_total"][e] = np.log1p(tot * 1e3)
        out["frq_lf_nu"][e] = lf / (lf + hf + EPS)
        out["frq_hf_nu"][e] = hf / (lf + hf + EPS)
        out["frq_lf_hf"][e] = np.log1p(lf / (hf + EPS))
        peak = f[hfm][np.argmax(pxx[hfm])] if hfm.any() else 0.0
        out["frq_hf_peak"][e] = peak
        out["frq_resp"][e] = peak * 60.0                    # breaths per minute
    return out


def _band_shape(pxx: np.ndarray, f: np.ndarray, lo: float, hi: float) -> tuple[float, float, float]:
    """(peak frequency, concentration, entropy) of a power spectrum inside one band."""
    m = (f >= lo) & (f < hi)
    if not m.any():
        return 0.0, 0.0, 0.0
    p = pxx[m]
    tot = p.sum() + EPS
    q = p / tot
    return float(f[m][p.argmax()]), float(p.max() / tot), float(-(q * np.log(q + EPS)).sum())


def _respiration(bvp: np.ndarray, peaks: np.ndarray, prom: np.ndarray, nn_t: np.ndarray,
                 nn: np.ndarray, n_epochs: int, cfg: FeatureConfig) -> dict:
    """Breathing rate and — the part that matters for REM — how *regular* the breathing is.

    Two independent surrogates of respiration are read off the wrist: RIIV, the respiratory
    modulation of pulse amplitude, and RSA, the respiratory modulation of the beat interval.
    NREM breathing is metronomic and gives a sharp spectral peak; REM breathing is erratic
    and smears that peak out, which is the single most useful wearable cue for REM.
    """
    fs_i = cfg.hrv_interp_hz
    epoch_s = cfg.epoch_samples / cfg.fs
    grid = np.arange(0.0, n_epochs * epoch_s, 1.0 / fs_i)
    names = ["rate", "conc", "entropy", "power"]
    out = {f"{src}_{n}": np.zeros(n_epochs) for src in ("riiv", "rsa") for n in names}

    sources = {}
    if peaks.size > 10:
        sources["riiv"] = np.interp(grid, peaks / cfg.fs, prom)
    if nn.size > 10:
        sources["rsa"] = np.interp(grid, nn_t, nn)

    half = int(90.0 * fs_i / 2)                             # 90 s window: ~15 breaths to judge
    nperseg = 256
    for src, sig in sources.items():
        sig = _butter(sig - sig.mean(), [0.08, 0.6], fs_i, "bandpass", order=2)
        for e in range(n_epochs):
            c = int((e + 0.5) * epoch_s * fs_i)
            seg = sig[max(0, c - half): c + half]
            if seg.size < nperseg:
                continue
            f, pxx = sps.welch(seg, fs=fs_i, nperseg=nperseg, noverlap=nperseg // 2)
            rate, conc, ent = _band_shape(pxx, f, 0.1, 0.5)
            out[f"{src}_rate"][e] = rate * 60.0
            out[f"{src}_conc"][e] = conc
            out[f"{src}_entropy"][e] = ent
            out[f"{src}_power"][e] = np.log1p(pxx[(f >= 0.1) & (f < 0.5)].sum() * 1e3)
    return out


def _ppg_block(bvp: np.ndarray, peaks: np.ndarray, prom: np.ndarray, n_epochs: int,
               cfg: FeatureConfig) -> dict:
    """Pulse amplitude (vasoconstriction), spectral shape and a flat-line quality flag."""
    fs = float(cfg.fs)
    raw = bvp.reshape(n_epochs, cfg.epoch_samples)
    out: dict = {}
    _agg(np.abs(raw), "ppg_abs", out)
    out["ppg_flat"] = (raw.std(1) < 1e-3).astype(float)
    out["ppg_p2p"] = raw.max(1) - raw.min(1)

    ep = (peaks / cfg.epoch_samples).astype(int)
    ep = np.clip(ep, 0, n_epochs - 1)
    amp_mean = np.zeros(n_epochs)
    amp_std = np.zeros(n_epochs)
    if peaks.size:
        counts = np.bincount(ep, minlength=n_epochs).astype(float)
        sums = np.bincount(ep, weights=prom, minlength=n_epochs)
        sq = np.bincount(ep, weights=prom * prom, minlength=n_epochs)
        nz = counts > 0
        amp_mean[nz] = sums[nz] / counts[nz]
        amp_std[nz] = np.sqrt(np.maximum(sq[nz] / counts[nz] - amp_mean[nz] ** 2, 0.0))
    out["ppg_amp"] = amp_mean
    out["ppg_amp_std"] = amp_std

    f, pxx = sps.welch(raw, fs=fs, nperseg=256, axis=1)
    tot = pxx.sum(1) + EPS
    for lo, hi, name in [(0.1, 0.5, "resp"), (0.5, 2.0, "card"), (2.0, 8.0, "harm")]:
        m = (f >= lo) & (f < hi)
        out[f"ppg_pw_{name}"] = np.log1p(pxx[:, m].sum(1))
        out[f"ppg_rel_{name}"] = pxx[:, m].sum(1) / tot
    p = pxx / tot[:, None]
    out["ppg_sp_entropy"] = -(p * np.log(p + EPS)).sum(1)
    return out


def _acc_block(acc: np.ndarray, n_epochs: int, cfg: FeatureConfig) -> dict:
    """Actigraphy. `acc` is (3, n_samples) in E4 counts; 64 counts = 1 g."""
    g = acc / ACC_COUNTS_PER_G
    mag = np.sqrt((g * g).sum(0))
    out: dict = {}
    per = lambda v: v.reshape(n_epochs, cfg.epoch_samples)

    _agg(per(mag), "acc_mag", out)
    out["acc_mag_p90"] = np.percentile(per(mag), 90, axis=1)
    for i, ax in enumerate("xyz"):
        out[f"acc_{ax}_mean"] = per(g[i]).mean(1)
        out[f"acc_{ax}_std"] = per(g[i]).std(1)

    d = np.abs(np.diff(mag, prepend=mag[0]))
    out["acc_count"] = np.log1p(per(d).sum(1))              # classic activity count
    out["acc_jerk"] = np.log1p(per(d).max(1))
    out["acc_move_frac"] = (per(d) > 0.01).mean(1)

    sec = cfg.fs                                            # stillness, judged per second
    std_s = per(mag).reshape(n_epochs, -1, sec).std(-1)
    out["acc_still_frac"] = (std_s < cfg.acc_still_g).mean(1)
    out["acc_sec_std_med"] = np.median(std_s, axis=1)

    ang = np.arctan2(g[2], np.sqrt(g[0] ** 2 + g[1] ** 2) + EPS) * 180.0 / np.pi
    out["acc_angle_mean"] = per(ang).mean(1)
    out["acc_angle_std"] = per(ang).std(1)
    out["acc_angle_change"] = np.abs(np.diff(out["acc_angle_mean"], prepend=out["acc_angle_mean"][0]))
    return out


def _slow_blocks(temp: np.ndarray, eda: np.ndarray, hr: np.ndarray, ibi: np.ndarray,
                 n_epochs: int, cfg: FeatureConfig) -> dict:
    per = lambda v: v.reshape(n_epochs, cfg.epoch_samples)
    out: dict = {}
    _agg(per(temp), "tmp", out, slope=True)
    _agg(per(eda), "eda", out, slope=True)
    _agg(per(hr), "e4_hr", out, slope=True)

    scr = _butter(eda, cfg.eda_scr_band, cfg.fs, "bandpass")
    out["eda_scr_std"] = per(scr).std(1)
    out["eda_scr_amp"] = per(np.abs(scr)).max(1)
    peaks, _ = sps.find_peaks(scr, prominence=0.01, distance=cfg.fs)
    ep = np.clip((peaks / cfg.epoch_samples).astype(int), 0, n_epochs - 1)
    out["eda_scr_n"] = np.bincount(ep, minlength=n_epochs).astype(float)

    hr1 = hr[::cfg.fs][: n_epochs * 30].reshape(n_epochs, 30)   # the device stream at its own 1 Hz
    d1 = np.diff(hr1, axis=1)
    out["e4_hr_sdsd"] = d1.std(1)                           # smoothed HRV proxy: REM is restless
    out["e4_hr_absd"] = np.abs(d1).mean(1)
    out["e4_hr_range"] = hr1.max(1) - hr1.min(1)
    out["e4_hr_iqr"] = np.subtract(*np.percentile(hr1, [75, 25], axis=1))

    ib = per(ibi)
    out["e4_ibi_mean"] = ib.mean(1)
    out["e4_ibi_std"] = ib.std(1)
    out["e4_ibi_changes"] = (np.abs(np.diff(ib, axis=1)) > 1e-6).sum(1).astype(float)  # device confidence
    return out


# --------------------------------------------------------------------------- entry point

def night_features(raw: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """`raw` is (n_samples, 8) in COLUMNS order -> (n_epochs, n_features), feature names."""
    n_epochs = raw.shape[0] // cfg.epoch_samples
    raw = raw[: n_epochs * cfg.epoch_samples]
    bvp, accx, accy, accz, temp, eda, hr, ibi = (np.ascontiguousarray(raw[:, i], dtype=np.float64)
                                                 for i in range(8))

    f_card, card_conc = cardiac_frequency(bvp, n_epochs, cfg)
    peaks, prom = detect_beats(bvp, f_card, cfg)
    nn_t, nn, _ = clean_nn(peaks, cfg, f_card)

    feats: dict = {"ppg_f_card": f_card * 60.0, "ppg_card_conc": card_conc}
    feats.update(_hrv_time_domain(nn_t, nn, n_epochs, cfg))
    feats.update(_hrv_frequency(nn_t, nn, n_epochs, cfg))
    feats.update(_respiration(bvp, peaks, prom, nn_t, nn, n_epochs, cfg))
    feats.update(_ppg_block(bvp, peaks, prom, n_epochs, cfg))
    feats.update(_acc_block(np.stack([accx, accy, accz]), n_epochs, cfg))
    feats.update(_slow_blocks(temp, eda, hr, ibi, n_epochs, cfg))

    names = sorted(feats)
    X = np.column_stack([_safe(np.asarray(feats[n], dtype=np.float64)) for n in names]).astype(np.float32)
    return X, names
