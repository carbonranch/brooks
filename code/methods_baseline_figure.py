"""Methods figure: why the modulation map is excess over each row's own running-median baseline.

Idealized, three panels. (A) two carrier rows carrying the same 30 Hz pulse train, one on a
quiet background and one on a loud background whose level wanders; (B) their raw modulation
spectra from 2 to 100 Hz, which differ everywhere in level and slope, with each row's baseline
dashed -- the v2 baseline, a running median over BASE_OCT octaves of LOG rate, exactly as
window_blobs2._excess computes it; (C) the excess, raw minus baseline: both rows sit at zero, one
threshold serves both, and the peak height is the modulation signal-to-background ratio.

The old figure (baseline_figure.py, 2026-09-15) used the 15 Hz linear median of the first
detector and carried a real-window bottom row that F03 now covers at better resolution.

Usage:
  python analysis/callers/methods_baseline_figure.py analysis/callers/runs/figures_paper/methods/baseline_subtraction
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs2 as W  # noqa: E402
import specdisplay as SD  # noqa: E402

DT = W.HOP / 44100.0


def synth_row(T: float, R: float, duty: float, depth_db: float, level_db: float, slope: float, seed: int):
    """A row's dB level series: a tone whose power is modulated by a raised-cosine pulse train at rate R
    (on for `duty` of the period), on a colored background whose modulation spectrum falls as 1/f^slope,
    scaled so its power at 30 Hz is level_db."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, DT)
    ph = (t * R) % 1.0
    env = np.where(ph < duty, 0.5 - 0.5 * np.cos(2 * np.pi * ph / duty), 0.0)
    n = len(t)
    Fw = np.fft.rfft(rng.standard_normal(n)); fq = np.fft.rfftfreq(n, DT)
    shape = np.where(fq > 0, (np.maximum(fq, 0.5) / 30.0) ** (-slope / 2), 0.0)
    colored = np.fft.irfft(Fw * shape, n)
    fr, S = scipy.signal.welch(colored, fs=1 / DT, nperseg=W.WELCH_NPER)
    colored *= 10 ** ((level_db - 10 * np.log10(S[np.argmin(np.abs(fr - 30))])) / 20)
    return t, depth_db * env + colored


def spectrum_and_baseline(L_row: np.ndarray):
    """Raw Welch modulation spectrum (dB) of a de-meaned dB level series, its v2 baseline and the excess,
    on the 2-100 Hz grid the detector keeps."""
    fr, S = scipy.signal.welch(L_row - L_row.mean(), fs=1 / DT, nperseg=W.WELCH_NPER)
    fr_k, exc = W._excess(fr, S[None, :])
    sel = (fr >= W.AM_LO) & (fr <= W.AM_HI)
    Sdb = 10 * np.log10(S[sel] + 1e-30)
    return fr_k, Sdb, Sdb - exc[0], exc[0]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("out"); args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    SD.paper_fonts()
    R = 30.0
    t, quiet = synth_row(300, R, 0.45, depth_db=6.0, level_db=-22.0, slope=0.6, seed=1)
    _, loud = synth_row(300, R, 0.45, depth_db=6.0, level_db=-8.0, slope=1.6, seed=2)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    ax = axes[0]; s = slice(0, int(0.5 / DT))
    ax.plot(t[s], quiet[s] - quiet.mean() + 12, color="tab:blue", lw=0.8, label="quiet row (+12 dB offset for display)")
    ax.plot(t[s], loud[s] - loud.mean(), color="tab:red", lw=0.8, label="loud, wandering row")
    ax.set_xlabel("time (s)"); ax.set_ylabel("level (dB, de-meaned)"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_title(f"(A)  the same {R:.0f} Hz pulse train, 6 dB deep, on a quiet row\nand on a loud row whose level wanders\n", loc="left", fontsize=10, fontweight="bold")

    ax = axes[1]
    for L, col, lab in ((quiet, "tab:blue", "quiet"), (loud, "tab:red", "loud")):
        fr, Sdb, base, exc = spectrum_and_baseline(L)
        ax.plot(fr, Sdb, color=col, lw=0.8, label=f"{lab} row: raw modulation spectrum")
        ax.plot(fr, base, color=col, lw=1.8, ls="--", label=f"{lab} row: running median over {W.BASE_OCT:g} octaves of rate")
    ax.axhline(-10, color="k", ls=":", lw=1)
    ax.text(0.03, 0.42, "a fixed threshold (dotted) fires on the\nwhole loud row and misses the quiet\nrow's harmonics", transform=ax.transAxes, ha="left", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(W.AM_LO, W.AM_HI)
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50, 100])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel("modulation rate (Hz)"); ax.set_ylabel("modulation power (dB)"); ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
    ax.set_title("(B)  raw modulation spectra: the same peaks on backgrounds\nthat differ in level and slope; dashed = each row's own\nrunning-median baseline", loc="left", fontsize=10, fontweight="bold")

    ax = axes[2]
    for L, col, lab in ((quiet, "tab:blue", "quiet"), (loud, "tab:red", "loud")):
        fr, Sdb, base, exc = spectrum_and_baseline(L)
        ax.plot(fr, exc, color=col, lw=0.9, label=f"{lab} row: excess, peak {exc.max():.0f} dB")
    ax.axhline(W.THRESH_DB, color="k", ls=":", lw=1); ax.text(0.98, 0.3, f"peak threshold {W.THRESH_DB:.0f} dB", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(W.AM_LO, W.AM_HI)
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50, 100])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel("modulation rate (Hz)"); ax.set_ylabel("excess modulation power (dB above the row's baseline)"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    ax.set_title("(C)  excess = raw minus baseline: both rows sit at zero and\none threshold serves both; the peak height is the\nmodulation signal-to-background ratio", loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(args.out + ".pdf", bbox_inches="tight"); fig.savefig(args.out + ".png", dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
