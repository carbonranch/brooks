"""Why the modulation spectrum is measured against a 15 Hz running median.

Top row, idealised: two carrier rows carrying the same 30 Hz pulse train,
one on a quiet background and one on a loud, sloping one. Their raw
modulation spectra differ everywhere; after each row's own running-median
baseline is subtracted both sit at zero and one threshold serves both; the
peak height that remains is the modulation signal-to-background ratio.

Bottom row, real: one 300 s window (Back9 8613, 2024-08-23 20:40). The raw
modulation spectra of three rows with their baselines; the raw map; and the
excess map -- the modulation map -- with the 6 dB contour.

Usage:
    python analysis/callers/baseline_figure.py analysis/callers/runs/figures_2026-09-15/00c_baseline_subtraction
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal

import window_blobs as W

DT = W.HOP / 44100.0


def mod_spectrum(L_row: np.ndarray):
    """Raw modulation spectrum (dB) of a de-meaned dB level series, its 15 Hz
    running-median baseline, and the excess, exactly as am_map does it."""
    fr, S = scipy.signal.welch(L_row - L_row.mean(), fs=1 / DT, nperseg=W.WELCH_NPER)
    sel = (fr >= W.AM_LO) & (fr <= W.AM_HI)
    fr, Sdb = fr[sel], 10 * np.log10(S[sel] + 1e-30)
    k = int(round(W.BASE_HZ / (fr[1] - fr[0]))) | 1
    base = scipy.ndimage.median_filter(Sdb, size=k, mode="nearest")
    return fr, Sdb, base, Sdb - base


def synth_row(T: float, R: float, duty: float, depth_db: float, level_db: float, slope: float, seed: int):
    """A row's dB level series: a steady tone whose power is modulated by a
    raised-cosine pulse train at rate R (on for `duty` of the period), on a
    coloured background -- noise whose modulation spectrum falls as
    1/f^slope, scaled so its power at 30 Hz is `level_db` -- as real rows have
    (panel d)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, DT)
    ph = (t * R) % 1.0
    env = np.where(ph < duty, 0.5 - 0.5 * np.cos(2 * np.pi * ph / duty), 0.0)
    pulse_db = depth_db * env
    n = len(t)
    white = rng.standard_normal(n)
    Fw = np.fft.rfft(white)
    fq = np.fft.rfftfreq(n, DT)
    shape = np.where(fq > 0, (np.maximum(fq, 0.5) / 30.0) ** (-slope / 2), 0.0)
    coloured = np.fft.irfft(Fw * shape, n)
    # scale so that the Welch modulation power at 30 Hz is level_db
    fr, S = scipy.signal.welch(coloured, fs=1 / DT, nperseg=W.WELCH_NPER)
    ref = 10 * np.log10(S[np.argmin(np.abs(fr - 30))])
    coloured *= 10 ** ((level_db - ref) / 20)
    return t, pulse_db + coloured


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="baseline_subtraction")
    ap.add_argument("--recorder", default="BARLT_00008613")
    ap.add_argument("--site", default="Back9")
    ap.add_argument("--time", default="2024-08-23 20:40", help="start of the real 300 s window, local time")
    ap.add_argument("--rows", nargs="+", default=["3067:3067 Hz: tree cricket (O. latipennis) @ 49.7 Hz", "7244:7244 Hz: ground cricket @ 24.3 Hz", "1000:1000 Hz: noise, no singer"],
                    help="rows for panel D as 'freqHz:label' (up to three)")
    args = ap.parse_args()
    out = args.out
    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    # ---------------- idealised ----------------
    R = 30.0
    t, quiet = synth_row(300, R, 0.45, depth_db=6.0, level_db=-22.0, slope=0.6, seed=1)
    _, loud = synth_row(300, R, 0.45, depth_db=6.0, level_db=-8.0, slope=1.6, seed=2)
    ax = fig.add_subplot(gs[0, 0])
    s = slice(0, int(0.5 / DT))
    ax.plot(t[s], quiet[s] - quiet.mean() + 12, color="tab:blue", lw=0.7, label="quiet row (+12 dB offset for display)")
    ax.plot(t[s], loud[s] - loud.mean(), color="tab:red", lw=0.7, label="loud, wandering row")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("level (dB, de-meaned)")
    ax.set_title("(A)  idealised: the same 30 Hz pulse train (6 dB deep)\non a quiet and on a loud, wandering background", loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for L, col, lab in ((quiet, "tab:blue", "quiet"), (loud, "tab:red", "loud")):
        fr, Sdb, base, exc = mod_spectrum(L)
        ax.plot(fr, Sdb, color=col, lw=0.8, label=f"{lab} row: raw modulation spectrum")
        ax.plot(fr, base, color=col, lw=1.6, ls="--", label=f"{lab} row: 15 Hz running median")
    ax.set_xlabel("modulation rate (Hz)")
    ax.set_ylabel("modulation power (dB)")
    ax.axhline(-10, color="k", ls=":", lw=1)
    ax.text(99, -9.3, "a fixed threshold at −10 dB would fire on all of the\nloud row and miss the quiet row's harmonics", ha="right", fontsize=7.5)
    ax.set_title("(B)  raw modulation spectra: the same peaks on backgrounds that\ndiffer in level and slope; dashed = each row's 15 Hz running median", loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    for L, col, lab in ((quiet, "tab:blue", "quiet"), (loud, "tab:red", "loud")):
        fr, Sdb, base, exc = mod_spectrum(L)
        ax.plot(fr, exc, color=col, lw=0.9, label=f"{lab} row: excess = raw − baseline (peak {exc.max():.1f} dB)")
    ax.axhline(W.THRESH_DB, color="k", ls=":", lw=1)
    ax.text(99, W.THRESH_DB + 0.5, f"peak threshold {W.THRESH_DB:.0f} dB", ha="right", fontsize=8)
    ax.set_xlabel("modulation rate (Hz)")
    ax.set_ylabel("excess modulation power (dB above baseline)")
    ax.set_title("(C)  excess = raw − baseline: both rows now sit at zero, so one threshold\nworks; the peak height is the modulation signal-to-background ratio", loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # ---------------- real ----------------
    tw = pd.Timestamp(args.time, tz=W.TZ)
    sig, sr, t0, path = W.read_window(args.recorder, args.site, tw, 300.0)
    f, tt, Z = scipy.signal.stft(sig, sr, nperseg=W.NPERSEG, noverlap=W.NPERSEG - W.HOP, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2
    L = 10 * np.log10(P + 1e-20)
    L -= L.mean(axis=1, keepdims=True)
    fr, S = scipy.signal.welch(L, fs=1 / DT, nperseg=W.WELCH_NPER, axis=1)
    sel = (fr >= W.AM_LO) & (fr <= W.AM_HI)
    fr, Sdb = fr[sel], 10 * np.log10(S[:, sel] + 1e-30)
    k = int(round(W.BASE_HZ / (fr[1] - fr[0]))) | 1
    base = scipy.ndimage.median_filter(Sdb, size=(1, k), mode="nearest")
    exc = Sdb - base

    ax = fig.add_subplot(gs[1, 0])
    cols3 = ["tab:green", "tab:purple", "0.4"]
    rows = [(float(spec.split(":", 1)[0]), cols3[i % 3], spec.split(":", 1)[1]) for i, spec in enumerate(args.rows)]
    for fz, col, lab in rows:
        i = int(np.argmin(np.abs(f - fz)))
        ax.plot(fr, Sdb[i], color=col, lw=0.8, label=lab)
        ax.plot(fr, base[i], color=col, lw=1.6, ls="--")
    ax.set_xlabel("modulation rate (Hz)")
    ax.set_ylabel("modulation power (dB)")
    ax.set_title(f"(D)  real window ({args.site} {args.recorder[-4:]}, {tw:%-d %b %Y %H:%M}):\nraw modulation spectra of three rows, baselines dashed", loc="left", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fmax = 12000
    fm = f <= fmax
    ax = fig.add_subplot(gs[1, 1])
    im = ax.pcolormesh(fr, f[fm], Sdb[fm], cmap="magma", shading="auto", rasterized=True,
                       vmin=np.percentile(Sdb[fm], 5), vmax=np.percentile(Sdb[fm], 99.5))
    ax.set_xscale("log")
    ax.set_xlabel("modulation rate (Hz)")
    ax.set_ylabel("carrier (Hz)")
    ax.set_title("(E)  raw map (dB): rows differ in level, all rows are\nbright at low rates, singers are hard to pick out", loc="left", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="dB")

    ax = fig.add_subplot(gs[1, 2])
    im = ax.pcolormesh(fr, f[fm], exc[fm], cmap="magma", shading="auto", rasterized=True, vmin=0, vmax=np.percentile(exc[fm], 99.8))
    ax.contour(fr, f[fm], exc[fm], levels=[W.THRESH_DB], colors="cyan", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("modulation rate (Hz)")
    ax.set_title(f"(F)  modulation map: raw minus each row's running median;\ncontour = {W.THRESH_DB:.0f} dB; peaks are singers and their harmonics", loc="left", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="dB above baseline")
    for fz, col, lab in rows[:2]:
        ax.axhline(fz, color=col, lw=0.6, ls=":", alpha=0.9)

    fig.suptitle("Measuring modulation against each row's own background: the 15 Hz running-median baseline", fontsize=13)
    fig.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out + ".png", bbox_inches="tight", dpi=110)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
