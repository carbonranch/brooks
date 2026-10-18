"""Spectrograms for display, finer than the analysis grid.

The analysis STFT is 512-point frames with a 128-point hop at 44.1 kHz (86 Hz rows,
2.9 ms steps), chosen for the modulation map. Drawn at that grid a 1 kHz band is twelve
rows and a 10 ms syllable three columns: pixels, not shapes (MBE, 2026-10-06). For
display the same frames are zero-padded to NFFT points (sinc interpolation of the
spectrum, no new information, no change to frequency resolution) and the hop is cut
to HOP_FINE samples, so the picture is smooth at the scale of a syllable. Short frames
(256 points, 5.8 ms) are used below ~200 ms where a syllable's glide is the subject.

Levels are shown in dB above each row's QUIET level (its 10th percentile over the
window or excerpt), never about its mean or median -- about the mean the silent part
of a period in the animal's rows clips to black and draws a false shadow.

paper_fonts() sets TrueType embedding so PDF text stays editable.
"""
from __future__ import annotations

import matplotlib
import numpy as np
from scipy.signal import stft

NPERSEG, HOP_FINE, NFFT = 512, 32, 4096          # 0.73 ms steps, 10.8 Hz display bins
NPERSEG_SHORT, HOP_SHORT, NFFT_SHORT = 256, 8, 4096


def paper_fonts() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def fine_spec(sig: np.ndarray, sr: int, nperseg: int = NPERSEG, hop: int = HOP_FINE, nfft: int = NFFT, pad_edges: bool = True):
    """(f, t, power) with t starting at 0 at the first sample when pad_edges (the signal is
    padded by half a frame so the picture covers the excerpt edge to edge)."""
    if pad_edges:
        pad = np.zeros(nperseg // 2)
        x = np.concatenate([pad, sig, pad])
    else:
        x = sig
    f, t, Z = stft(x, sr, nperseg=nperseg, noverlap=nperseg - hop, nfft=nfft, window="hann", boundary=None, padded=False)
    if pad_edges:
        t = t - (nperseg // 2) / sr
    return f, t, np.abs(Z) ** 2


def db_above_quiet(P: np.ndarray, quiet_db: np.ndarray | None = None, q: float = 10.0) -> np.ndarray:
    """dB above each row's quiet level. quiet_db: per-row level to use (e.g. from the whole
    300 s window, interpolated to this frequency grid); default the row's own q-th percentile."""
    L = 10 * np.log10(P + 1e-20)
    if quiet_db is None:
        quiet_db = np.percentile(L, q, axis=1, keepdims=True)
    return L - quiet_db


def quiet_on_grid(f_fine: np.ndarray, f_coarse: np.ndarray, quiet_coarse: np.ndarray) -> np.ndarray:
    """A per-row quiet level measured on the analysis grid, interpolated to the display grid."""
    return np.interp(f_fine, f_coarse, np.asarray(quiet_coarse).ravel())[:, None]
