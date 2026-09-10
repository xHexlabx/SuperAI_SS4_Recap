"""Signal -> image encoders.

The trick that made this hackathon work: a 1-D UWB channel-impulse-response trace carries
structure that a 2-D vision backbone reads far better than a 1-D CNN does. Each encoder below
turns one trace into a single-channel HxW map in [0, 1]; `encode_signal` stacks three of them
into an RGB image so a pretrained ImageNet backbone can be fine-tuned directly.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy import signal as sps

from . import radar

ENCODERS = ("raw2d", "spectrogram", "cwt", "gaf", "recurrence")


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize to (size, size) without pulling in PIL/cv2 for a 2-D float map."""
    h, w = img.shape
    if (h, w) == (size, size):
        return img.astype(np.float32)
    ys = np.linspace(0, h - 1, size)
    xs = np.linspace(0, w - 1, size)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def spectrogram_image(x: np.ndarray, size: int, nperseg: int, noverlap: int, log_power: bool) -> np.ndarray:
    """STFT magnitude — where energy sits in time and frequency."""
    nperseg = min(nperseg, len(x))
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0
    _, _, spec = sps.stft(x, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
    mag = np.abs(spec)
    if log_power:
        mag = np.log1p(mag)
    return _resize(_minmax(mag), size)


def cwt_image(x: np.ndarray, size: int, wavelet: str, n_scales: int) -> np.ndarray:
    """Continuous-wavelet scalogram — keeps sharp transients that STFT windows smear."""
    scales = np.arange(1, n_scales + 1)
    coef, _ = pywt.cwt(x, scales, wavelet)
    return _resize(_minmax(np.abs(coef)), size)


def gaf_image(x: np.ndarray, size: int) -> np.ndarray:
    """Gramian Angular Summation Field — encodes pairwise temporal correlation as texture."""
    # Downsample first: GAF is O(n^2) and a raw trace is long.
    if len(x) > size:
        idx = np.linspace(0, len(x) - 1, size).astype(int)
        x = x[idx]
    scaled = _minmax(x) * 2.0 - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    gaf = np.cos(phi[:, None] + phi[None, :])
    return _resize(_minmax(gaf), size)


def recurrence_image(x: np.ndarray, size: int) -> np.ndarray:
    """Recurrence plot — how often the trace revisits a similar amplitude."""
    if len(x) > size:
        idx = np.linspace(0, len(x) - 1, size).astype(int)
        x = x[idx]
    s = _minmax(x)
    dist = np.abs(s[:, None] - s[None, :])
    return _resize(_minmax(-dist), size)


def raw2d_image(x2d: np.ndarray, size: int, log_power: bool) -> np.ndarray:
    """Use the sample's own 2-D shape as the image.

    A UWB sample recorded as (frames x range-bins) is already a picture — a range-time map.
    Resizing it directly keeps the spatial layout the radar produced, which no 1-D transform
    can reconstruct once the frames have been collapsed.
    """
    mag = np.abs(np.asarray(x2d, dtype=np.float64))
    if mag.ndim == 1:
        side = int(np.sqrt(mag.size))
        mag = mag[: side * side].reshape(side, side)
    if log_power:
        mag = np.log1p(mag)
    return _resize(_minmax(mag), size)


def encode_one(x: np.ndarray, method: str, cfg) -> np.ndarray:
    """Run a single 1-D encoder named by `method`, using EncodeConfig `cfg`."""
    if method == "raw2d":
        return raw2d_image(x, cfg.img_size, cfg.log_power)
    if method == "spectrogram":
        return spectrogram_image(x, cfg.img_size, cfg.nperseg, cfg.noverlap, cfg.log_power)
    if method == "cwt":
        return cwt_image(x, cfg.img_size, cfg.wavelet, cfg.n_scales)
    if method == "gaf":
        return gaf_image(x, cfg.img_size)
    if method == "recurrence":
        return recurrence_image(x, cfg.img_size)
    raise ValueError(f"unknown encoder {method!r}; expected one of {ENCODERS + radar.ENCODERS}")


def encode_signal(x: np.ndarray, cfg) -> np.ndarray:
    """Encode one sample into a float32 CHW image with 3 channels, values in [0, 1].

    `method: stack` puts a different encoder in each RGB channel. For this dataset the default
    stack is the three radar views (`rti`, `range_doppler`, `micro_doppler`): the backbone gets
    position, velocity, and the time-ordering of velocity at once, for the cost of one image.
    """
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    methods = list(cfg.stack_methods) if cfg.method == "stack" else [cfg.method] * 3
    if cfg.method == "stack" and len(methods) != 3:
        raise ValueError(f"encode.stack_methods must have exactly 3 entries, got {methods}")

    trace = None
    planes = []
    for method in methods:
        if radar.is_radar_encoder(method):
            planes.append(radar.encode_one(x, method, cfg))
            continue
        if method == "raw2d":
            planes.append(encode_one(x, method, cfg))
            continue
        if trace is None:
            # Generic 1-D encoders need a single real trace; for radar IQ that is the
            # clutter-removed energy summed across range bins.
            trace = radar.range_profile_trace(x, cfg) if x.ndim == 2 or np.iscomplexobj(x) else np.real(x).ravel()
        planes.append(encode_one(trace, method, cfg))

    return np.stack(planes, axis=0).astype(np.float32)


def encode_multichannel(x: np.ndarray, n_channels: int, cfg) -> np.ndarray:
    """Encode one sample, optionally splitting an interleaved multi-antenna trace first.

    `n_channels > 1` only applies to flat 1-D samples; a 2-D radar sample already carries its
    antenna/range structure and is passed through untouched.
    """
    x = np.asarray(x)
    if n_channels <= 1 or x.ndim == 2 or np.iscomplexobj(x):
        return encode_signal(x, cfg)
    x = x.ravel()
    if len(x) % n_channels != 0:
        raise ValueError(f"signal length {len(x)} is not divisible by n_channels={n_channels}")
    traces = x.reshape(n_channels, -1)
    return np.mean([encode_signal(t, cfg) for t in traces], axis=0).astype(np.float32)
