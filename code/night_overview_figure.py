"""Figure 3: one real night at several time scales, as a data overview: the whole-night spectrogram, the same night
in the tree-cricket band with the air temperature (the carrier drifting with temperature), one hour, one 300 s
BROOKS window as recorded, and the same window row-normalized (as the detector sees it).

Layout and wording follow MBE's Figure 3 of 2026-10-10 (MBE_Figures/MBE_F03_one_night_audio.pdf): five full-width
panels on the letter page.
  (A) Full night 19:00 to 07:00, 0-12 kHz, mean power per 5 s block; the hour (C) and window (D) marked
  (B) Temperature dependence of carrier frequency: the band around the night's longest track, HRRR temperature cyan
  (C) Carrier frequency stability of one hour, 1 s columns; the window (D) marked
  (D) Example of 5 minute window used for analysis, 0.093 s columns, absolute dB
  (E) Row normalized 5 minute window: the same window, 1.5-12 kHz log axis, dB above each row's quiet level
Every panel is drawn from NPERSEG-point frames (2048: 21.5 Hz rows). The night is cached as .npz beside the output.

Usage:
    python analysis/callers/night_overview_figure.py --recorder BARLT_00008662 --site House --night 2025-09-20 \
        --tracks-dir analysis/callers/runs/v2/y2025/BARLT_00008662_House_20250920_night --track 51 \
        --weather analysis/callers/runs/v2/y2025/weather_hourly_hrrr.csv --out analysis/callers/runs/figures_paper/F03_one_night_audio
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.signal import stft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs as W  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402
from window_figure import khz_log  # noqa: E402

NPERSEG = 2048        # 21.5 Hz rows for every panel
BLOCK_S = 5.0         # the night: mean power per 5 s block (8640 columns over 12 h)
HOUR_COL_S = 1.0      # the hour: 1 s columns
WIN_COL_S = 0.093     # the window: four half-overlapped 2048-point frames per column
FMAX_LTS = 12000.0
FMIN_LOG = 1500.0     # (E)
# page geometry (inches from the page's top-left), after MBE's layout: axes X0..X1, rows (top, height)
X0, X1 = 1.40, 7.50
ROWS = {"A": (1.17, 1.50), "B": (3.20, 0.85), "C": (4.60, 0.85), "D": (6.00, 0.85), "E": (7.45, 1.05)}


def block_mean_db(P, tt, block_s, n_blocks, offset_blocks=0, S=None):
    """Mean power per block_s block of the STFT power P (columns at times tt), in dB, written into S at column
    offset_blocks + b (a new array when S is None). Columns with no frames stay NaN."""
    if S is None:
        S = np.full((P.shape[0], n_blocks), np.nan)
    for b in range(int(np.ceil(tt[-1] / block_s))):
        cols = (tt >= b * block_s) & (tt < (b + 1) * block_s)
        col = offset_blocks + b
        if col < n_blocks and cols.any():
            S[:, col] = 10 * np.log10(P[:, cols].mean(axis=1) + 1e-20)
    return S


def read_600(recorder, site, tk):
    x, sr, _, _ = W.read_window(recorder, site, tk, 600.0)
    if sr != 44100:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(44100, sr)
        x = resample_poly(x, 44100 // g, sr // g)
    return x


def night_lts(recorder, site, night, cache):
    """Long-term spectrogram 19:00 -> 07:00 local, dB, rows = frequency bins (NPERSEG-point frames, half
    overlapped), columns = BLOCK_S blocks. Missing audio -> NaN columns."""
    if os.path.exists(cache):
        z = np.load(cache)
        return z["S"], z["f"], pd.to_datetime(z["t"].astype("int64"), utc=True).tz_convert(W.TZ)
    t0 = pd.Timestamp(f"{night} 19:00", tz=W.TZ)
    n_blocks = int(12 * 3600 / BLOCK_S)
    f = np.fft.rfftfreq(NPERSEG, 1 / 44100)
    S = np.full((len(f), n_blocks), np.nan)
    for h in range(12):
        th = t0 + pd.Timedelta(hours=h)
        for k in range(6):
            tk = th + pd.Timedelta(minutes=10 * k)
            try:
                x = read_600(recorder, site, tk)
            except Exception as e:
                print(f"  {tk:%H:%M}: no audio ({type(e).__name__})")
                continue
            _, tt, Z = stft(x, 44100, nperseg=NPERSEG, noverlap=NPERSEG // 2, window="hann", boundary=None, padded=False)
            block_mean_db(np.abs(Z) ** 2, tt, BLOCK_S, n_blocks, int((h * 3600 + k * 600) / BLOCK_S), S)
        print(f"  {th:%H:%M} done")
    t = t0 + pd.to_timedelta(np.arange(n_blocks) * BLOCK_S + BLOCK_S / 2, unit="s")
    np.savez_compressed(cache, S=S, f=f, t=t.tz_convert("UTC").asi8)
    return S, f, t


def hour_spec(recorder, site, zh):
    """One hour from zh in HOUR_COL_S columns, NPERSEG-point frames, dB. Returns (f, t, S)."""
    n_cols = int(3600 / HOUR_COL_S)
    f = np.fft.rfftfreq(NPERSEG, 1 / 44100)
    S = np.full((len(f), n_cols), np.nan)
    for k in range(6):
        tk = zh + pd.Timedelta(minutes=10 * k + 5)             # read_window centers on its target: the 600 s from zh + 10k min
        try:
            x = read_600(recorder, site, tk)
        except Exception as e:
            print(f"  hour {tk:%H:%M}: no audio ({type(e).__name__})")
            continue
        _, tt, Z = stft(x, 44100, nperseg=NPERSEG, noverlap=NPERSEG // 2, window="hann", boundary=None, padded=False)
        block_mean_db(np.abs(Z) ** 2, tt, HOUR_COL_S, n_cols, int(k * 600 / HOUR_COL_S), S)
    t = zh + pd.to_timedelta(np.arange(n_cols) * HOUR_COL_S + HOUR_COL_S / 2, unit="s")
    return f, t, S


def khz_axis(ax, fmax=FMAX_LTS):
    ax.set_ylim(0, fmax); ax.set_yticks(np.arange(0, fmax + 1, 2000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}")); ax.set_ylabel("frequency (kHz)")


def mark(ax, x0, width, letter, y_frac=0.97):
    """A translucent white box over a span of the x axis with its letter at the top."""
    ax.axvspan(x0, x0 + width, color="w", alpha=0.18, lw=0)
    ax.text(x0, ax.get_ylim()[1] * y_frac, f" {letter}", color="w", fontsize=BODY_PT, fontweight="bold", va="top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder", default="BARLT_00008662")
    ap.add_argument("--site", default="House")
    ap.add_argument("--night", default="2025-09-20")
    ap.add_argument("--zoom-hour", default="23:00")
    ap.add_argument("--window", default="23:05", help="center of the 300 s BROOKS window")
    ap.add_argument("--drift-band", type=float, nargs=2, default=[2300, 3600], help="band for the whole-night carrier-drift panel")
    ap.add_argument("--weather", default="analysis/callers/runs/v2/y2025/weather_hourly_hrrr.csv")
    ap.add_argument("--tracks-dir", default=None, help="a v2 unit directory; its longest global track chooses the window and band")
    ap.add_argument("--track", type=int, default=None, help="use this track id from --tracks-dir instead of the longest")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.tracks_dir:
        Tt = pd.read_csv(os.path.join(args.tracks_dir, "night_night_global_tracks.csv"))
        Wa = pd.read_csv(os.path.join(args.tracks_dir, "night_night_global_window_animals.csv"))
        tr = args.track if args.track is not None else int(Tt[Tt.R.notna()].sort_values("span_h", ascending=False).track.iloc[0])
        g = Wa[(Wa.track == tr) & Wa.F_fix.notna()].sort_values("depth", ascending=False)
        b = g.iloc[0]
        tb = pd.Timestamp(b.t).tz_convert(W.TZ)
        args.window = f"{tb:%H:%M}"
        args.zoom_hour = f"{tb.floor('h'):%H:%M}"
        args.drift_band = [max(300.0, float(g.F_fix.min()) - 700), float(g.F_fix.max()) + 700]
        print(f"track {tr}: {float(Tt.set_index('track').loc[tr, 'span_h']):.1f} h, deepest window {tb:%H:%M}, band {b.F_lo:.0f}-{b.F_hi:.0f} Hz, R {b.R:.2f} Hz")
    out_dir = os.path.dirname(args.out); os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f"night_lts{NPERSEG}_{BLOCK_S:g}s_{args.recorder}_{args.night}.npz")
    if not os.path.exists(cache):                                     # the cache of the earlier F02_night_overview run, if present
        old = glob.glob(os.path.join(out_dir, f"*_lts{NPERSEG}_{BLOCK_S:g}s_{args.recorder}_{args.night}.npz"))
        if old:
            cache = old[0]
    S, f, t = night_lts(args.recorder, args.site, args.night, cache)
    Wx = pd.read_csv(args.weather)
    Wx["t"] = pd.to_datetime(Wx.t, utc=True).dt.tz_convert(W.TZ)
    Wx = Wx.dropna(subset=["T_C"]).sort_values("t")

    zh = pd.Timestamp(f"{args.night} {args.zoom_hour}", tz=W.TZ)
    if zh.hour < 12:
        zh += pd.Timedelta(days=1)
    tw = pd.Timestamp(f"{args.night} {args.window}", tz=W.TZ)
    if tw.hour < 12:
        tw += pd.Timedelta(days=1)
    tw0 = tw - pd.Timedelta(seconds=150)                            # the window starts 150 s before its center
    hhmm = mdates.DateFormatter("%H:%M", tz=W.TZ)

    page = PaperPage(number=3, title="Example night audio")
    Wc, Hc = 7.2, page.size[1]                                    # a canvas wider than the margins: room for (B)'s right axis label and the last tick labels
    fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def panel(key):
        top, h = ROWS[key]
        return fig.add_axes([(X0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (X1 - X0) / Wc, h / Hc]), h

    # ---- (A) the night ----
    ax, h = panel("A")
    fm = f <= FMAX_LTS
    Sn = S[fm]; ref = np.nanpercentile(Sn, 99.5)
    ax.pcolormesh(t, f[fm], Sn, cmap="magma", vmin=ref - 45, vmax=ref, shading="auto", rasterized=True)
    khz_axis(ax); ax.set_xlim(t[0], t[-1])
    mark(ax, zh, pd.Timedelta(hours=1), "C", 0.97); mark(ax, tw0, pd.Timedelta(minutes=5), "D", 0.86)
    ax.xaxis.set_major_formatter(hhmm)
    page.label(ax, "A", "Full night 19:00 to 07:00", y=1 + 0.05 / h)

    # ---- (B) the night in one band: the carrier drifts with temperature ----
    ax, h = panel("B")
    fb = (f >= args.drift_band[0]) & (f <= args.drift_band[1])
    Sb = S[fb]; refb = np.nanpercentile(Sb, 99.5)
    ax.pcolormesh(t, f[fb], Sb, cmap="magma", vmin=refb - 35, vmax=refb, shading="auto", rasterized=True)
    ax.set_ylim(args.drift_band); ax.set_xlim(t[0], t[-1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:.1f}")); ax.set_ylabel("frequency (kHz)")
    axT = ax.twinx()
    wn = Wx[(Wx.t >= t[0]) & (Wx.t <= t[-1])]
    axT.plot(wn.t, wn.T_C, "-", color="cyan", lw=1.0)
    axT.set_ylabel("temperature (°C), cyan line"); axT.set_ylim(np.floor(wn.T_C.min()) - 1, np.ceil(wn.T_C.max()) + 1)
    ax.xaxis.set_major_formatter(hhmm)
    page.label(ax, "B", "Temperature dependence of carrier frequency", y=1 + 0.05 / h)

    # ---- (C) one hour, read again from the audio at 1 s columns ----
    ax, h = panel("C")
    fh, th_, Sh = hour_spec(args.recorder, args.site, zh)
    fmh = fh <= FMAX_LTS; refh = np.nanpercentile(Sh[fmh], 99.5)
    ax.pcolormesh(th_, fh[fmh], Sh[fmh], cmap="magma", vmin=refh - 45, vmax=refh, shading="auto", rasterized=True)
    khz_axis(ax); ax.set_xlim(zh, zh + pd.Timedelta(hours=1))
    mark(ax, tw0, pd.Timedelta(minutes=5), "D", 0.97)
    ax.xaxis.set_major_formatter(hhmm)
    page.label(ax, "C", "Carrier frequency stability of one hour", y=1 + 0.05 / h)

    # ---- (D) one BROOKS window (300 s), as recorded ----
    sig, sr, t0w, _ = W.read_window(args.recorder, args.site, tw, 300.0)
    fw, tt, Zw = stft(sig, sr, nperseg=NPERSEG, noverlap=NPERSEG // 2, window="hann", boundary=None, padded=False)
    Lw = 10 * np.log10(np.abs(Zw) ** 2 + 1e-20)
    kcol = max(1, int(round(WIN_COL_S / (tt[1] - tt[0])))); nb = Lw.shape[1] // kcol
    te = np.arange(nb + 1) * kcol * (tt[1] - tt[0]); dfw = fw[1] - fw[0]
    ax, h = panel("D")
    fmw = fw <= FMAX_LTS
    Sd = 10 * np.log10((10 ** (Lw[fmw][:, :nb * kcol] / 10)).reshape(fmw.sum(), nb, kcol).mean(axis=2) + 1e-20)
    refw = np.nanpercentile(Sd, 99.5)
    ax.pcolormesh(te, np.concatenate([[fw[fmw][0] - dfw / 2], fw[fmw] + dfw / 2]), Sd, cmap="magma", vmin=refw - 45, vmax=refw, shading="flat", rasterized=True)
    khz_axis(ax); ax.set_xlim(0, 300); ax.set_xlabel("seconds into the window")
    page.label(ax, "D", "Example of 5 minute window used for analysis", y=1 + 0.05 / h)

    # ---- (E) the same window row-normalized: dB above each row's quiet level, log axis (as F03 A / the detector) ----
    ax, h = panel("E")
    LA = Lw - np.percentile(Lw, 10, axis=1, keepdims=True)
    DA = LA[:, :nb * kcol].reshape(LA.shape[0], nb, kcol).mean(axis=2)
    selE = (fw >= FMIN_LOG) & (fw <= FMAX_LTS)
    vlo, vhi = np.percentile(DA[selE], 5), np.percentile(DA[selE], 99.7)
    ax.pcolormesh(te, np.concatenate([[fw[selE][0] - dfw / 2], fw[selE] + dfw / 2]), DA[selE], cmap="magma", vmin=vlo, vmax=vhi, shading="flat", rasterized=True)
    khz_log(ax, FMIN_LOG, FMAX_LTS); ax.set_ylabel("frequency (kHz)")
    ax.set_xlim(0, 300); ax.set_xlabel("seconds into the window")
    page.label(ax, "E", "Row normalized 5 minute window", y=1 + 0.05 / h)

    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png  (window {t0w.tz_convert(W.TZ):%H:%M:%S} to {(t0w + pd.Timedelta(seconds=300)).tz_convert(W.TZ):%H:%M:%S})")


if __name__ == "__main__":
    main()
