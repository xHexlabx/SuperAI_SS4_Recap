"""Radar-specific views of a UWB sample.

A sample here is complex baseband IQ from a Novelda X4M03: `(slow-time frames, range bins)`,
2560 x 56 for almost every file. That is already a 2-D signal, so the useful thing is not to
flatten it but to render the three views a radar engineer would actually look at, and let a
pretrained vision backbone read them as an RGB image.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

ENCODERS = ("rti", "range_doppler", "micro_doppler", "spectrogram2d", "fft2")
# plus "rd_t<k>"    — range-Doppler of the k-th time window, e.g. rd_t0 / rd_t1 / rd_t2
# plus "sgband<k>"  — spectrogram2d over the k-th slice of range bins

EPS = 1e-10


def _to_db(mag: np.ndarray, dynamic_range: float) -> np.ndarray:
    """dB relative to this image's own peak, clipped to `dynamic_range` below it.

    An absolute floor (the baseline used -45 dB) only suits Range-Doppler, where the FFT adds
    ~68 dB of coherent gain. The same floor applied to a raw magnitude map buries every pixel,
    and it makes the encoding sensitive to per-recording gain differences. Referencing the peak
    keeps all three views on a comparable scale.
    """
    peak = float(np.max(mag)) + EPS
    db = 20.0 * np.log10((mag + EPS) / peak)
    return np.maximum(db, -dynamic_range)


def prepare(x: np.ndarray, n_frames: int, range_bins: int = 56) -> np.ndarray:
    """Normalise a raw sample to complex `(n_frames, range_bins)`.

    One file in this dataset is 7680 frames instead of 2560; keeping the *last* n_frames
    matches the baseline and keeps the end of the motion, which is where a fall resolves.
    """
    x = np.asarray(x)
    if x.ndim == 1:  # a flattened sample — put the range bins back
        if len(x) % range_bins != 0:
            raise ValueError(
                f"a radar encoder needs 2-D samples (frames x range bins); got a flat array of "
                f"length {len(x)} which is not divisible by encode.range_bins={range_bins}. "
                f"Use a 1-D encoder (spectrogram / cwt / gaf / recurrence) for this data."
            )
        x = x.reshape(-1, range_bins)
    if not np.iscomplexobj(x):
        x = x.astype(np.complex128)
    if x.shape[0] > n_frames:
        x = x[-n_frames:]
    elif x.shape[0] < n_frames:
        pad = n_frames - x.shape[0]
        x = np.pad(x, ((pad, 0), (0, 0)), mode="edge")
    return x


def remove_clutter(x: np.ndarray) -> np.ndarray:
    """Subtract each range bin's slow-time mean.

    Static reflections (walls, furniture) dominate the raw amplitude and are identical across
    every class, so leaving them in wastes most of the image's dynamic range on the background.
    """
    return x - x.mean(axis=0, keepdims=True)


def rti_image(x: np.ndarray, size: int, cfg) -> np.ndarray:
    """Range-Time Intensity: where the moving body is, over time.

    Rendered range-on-Y / time-on-X, which is how these plots are conventionally read.
    """
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)
    return _norm(_to_db(np.abs(x), cfg.dynamic_range).T, size)


def range_doppler_image(x: np.ndarray, size: int, cfg) -> np.ndarray:
    """Range-Doppler: how fast things move, at each distance.

    Same transform the organisers' baseline used — Hann window over slow time, FFT, dB floor —
    with an optional crop around zero Doppler, where human motion actually lives.
    """
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)
    windowed = np.hanning(x.shape[0])[:, None] * x
    rd = np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)
    rd = _to_db(np.abs(rd), cfg.dynamic_range)

    if 0 < cfg.doppler_crop < 1.0:
        half = max(1, int(rd.shape[0] * cfg.doppler_crop / 2))
        centre = rd.shape[0] // 2
        rd = rd[centre - half: centre + half]
    return _norm(rd, size)


def micro_doppler_image(x: np.ndarray, size: int, cfg) -> np.ndarray:
    """Micro-Doppler spectrogram: the velocity signature over time.

    A single Range-Doppler FFT averages the whole clip into one frame, which blurs together
    motions that differ mainly in *ordering* — sitting down versus standing up produce mirrored
    velocity traces but very similar global spectra. An STFT along slow time keeps that ordering,
    so it separates the reversed-motion class pairs this dataset is full of.
    """
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)

    # Collapse range coherently: the body spans a few bins and summing complex values keeps phase.
    trace = x.sum(axis=1)

    nperseg = min(cfg.md_nperseg, len(trace))
    noverlap = min(cfg.md_noverlap, nperseg - 1)
    _, _, spec = sps.stft(trace, nperseg=nperseg, noverlap=noverlap,
                          return_onesided=False, boundary=None, padded=False)
    spec = np.fft.fftshift(spec, axes=0)
    return _norm(_to_db(np.abs(spec), cfg.dynamic_range), size)



def spectrogram2d_image(x: np.ndarray, size: int, cfg) -> np.ndarray:
    """STFT every range bin, then sum the magnitudes across range.

    `micro_doppler` sums the complex data across range *before* the STFT. That is coherent, so
    two reflections at different ranges with opposing phase partially cancel — exactly what a
    body spanning several bins produces. Summing after taking magnitudes cannot cancel, at the
    cost of throwing away the phase relationship between bins.
    """
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)

    nperseg = min(cfg.md_nperseg, x.shape[0])
    noverlap = min(cfg.md_noverlap, nperseg - 1)
    _, _, spec = sps.stft(x, axis=0, nperseg=nperseg, noverlap=noverlap,
                          return_onesided=False, boundary=None, padded=False)
    mag = np.abs(spec).sum(axis=1)          # (freq, range, time) -> (freq, time)
    mag = np.fft.fftshift(mag, axes=0)
    return _norm(_to_db(mag, cfg.dynamic_range), size)


def fft2_image(x: np.ndarray, size: int, cfg) -> np.ndarray:
    """2-D FFT of the whole clip: Doppler on one axis, range-frequency on the other."""
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)
    window = np.hanning(x.shape[0])[:, None] * np.hanning(x.shape[1])[None, :]
    spec = np.fft.fftshift(np.fft.fft2(window * x))
    return _norm(_to_db(np.abs(spec), cfg.dynamic_range), size)


def rd_window_image(x: np.ndarray, size: int, cfg, k: int) -> np.ndarray:
    """Range-Doppler of the k-th of `cfg.rd_windows` equal time slices.

    A single Range-Doppler map averages the whole clip, so a motion and its reverse produce
    nearly the same picture — which is why `range_doppler` scores worst on a dataset whose
    classes include both sit->stand and stand->sit. Splitting the clip and putting each slice in
    its own RGB channel puts the ordering back without changing the model.
    """
    n = max(1, cfg.rd_windows)
    if not 0 <= k < n:
        raise ValueError(f"rd_t{k} is out of range for encode.rd_windows={n}")
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)

    span = x.shape[0] // n
    x = x[k * span: (k + 1) * span]
    windowed = np.hanning(x.shape[0])[:, None] * x
    rd = np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)
    rd = _to_db(np.abs(rd), cfg.dynamic_range)

    if 0 < cfg.doppler_crop < 1.0:
        half = max(1, int(rd.shape[0] * cfg.doppler_crop / 2))
        centre = rd.shape[0] // 2
        rd = rd[centre - half: centre + half]
    return _norm(rd, size)



def range_band_spectrogram(x: np.ndarray, size: int, cfg, k: int) -> np.ndarray:
    """spectrogram2d restricted to the k-th slice of range bins.

    `spectrogram2d` sums the magnitudes over *all* 56 range bins, which discards where the
    motion happened — the strongest encoding we have throws that axis away entirely. Splitting
    range into a few contiguous bands and giving each its own RGB channel puts coarse position
    back without enlarging the image, and unlike stacking different encoders the channels here
    carry the same kind of measurement, so none of them is the weak one dragging the rest down.
    """
    n = max(1, cfg.range_bands)
    if not 0 <= k < n:
        raise ValueError(f"sgband{k} is out of range for encode.range_bands={n}")
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)

    edges = np.linspace(0, x.shape[1], n + 1).astype(int)
    x = x[:, edges[k]:edges[k + 1]]

    nperseg = min(cfg.md_nperseg, x.shape[0])
    noverlap = min(cfg.md_noverlap, nperseg - 1)
    _, _, spec = sps.stft(x, axis=0, nperseg=nperseg, noverlap=noverlap,
                          return_onesided=False, boundary=None, padded=False)
    mag = np.fft.fftshift(np.abs(spec).sum(axis=1), axes=0)
    return _norm(_to_db(mag, cfg.dynamic_range), size)


def range_profile_trace(x: np.ndarray, cfg) -> np.ndarray:
    """A single real 1-D trace, for the generic (non-radar) encoders."""
    x = prepare(x, cfg.n_frames, cfg.range_bins)
    if cfg.clutter_removal:
        x = remove_clutter(x)
    return np.abs(x).sum(axis=1)


def _norm(img: np.ndarray, size: int) -> np.ndarray:
    from .signal2image import _minmax, _resize

    return _resize(_minmax(img), size)


def is_radar_encoder(method: str) -> bool:
    return (method in ENCODERS
            or (method.startswith("rd_t") and method[4:].isdigit())
            or (method.startswith("sgband") and method[6:].isdigit()))


def encode_one(x: np.ndarray, method: str, cfg) -> np.ndarray:
    if method == "rti":
        return rti_image(x, cfg.img_size, cfg)
    if method == "range_doppler":
        return range_doppler_image(x, cfg.img_size, cfg)
    if method == "micro_doppler":
        return micro_doppler_image(x, cfg.img_size, cfg)
    if method == "spectrogram2d":
        return spectrogram2d_image(x, cfg.img_size, cfg)
    if method == "fft2":
        return fft2_image(x, cfg.img_size, cfg)
    if method.startswith("rd_t") and method[4:].isdigit():
        return rd_window_image(x, cfg.img_size, cfg, int(method[4:]))
    if method.startswith("sgband") and method[6:].isdigit():
        return range_band_spectrogram(x, cfg.img_size, cfg, int(method[6:]))
    raise ValueError(f"unknown radar encoder {method!r}; expected one of {ENCODERS} or rd_t<k>")
