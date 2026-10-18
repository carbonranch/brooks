"""Figure 1: the parts of an orthopteran calling song, as the paper uses the terms: carrier frequency, syllable
(pulse), repeat period and rate, duty, glide and harmonic, trills and chirps, each as a waveform and a spectrogram.

Layout and wording follow MBE's Figure 1 of 2026-10-10 (MBE_Figures/MBE_F01_song_anatomy.pdf): seven full-width
panels on the letter page, paired rows on one time axis, then a zoom-out, twice.
  (A) A syllable: carrier cycles under an envelope, the carrier gliding 500 Hz down through the syllable
  (B) A trill: syllables repeated without pause (waveform, 200 ms)
  (C) Spectrogram of trill: the same 200 ms, aligned with (B), log-frequency axis (the 2F harmonic has the same shape)
  (D) Trilling song over 4 s
  (E) A chirp: multi-syllable groups as the repeating unit (waveform, 620 ms)
  (F) Chirp spectrogram, aligned with (E)
  (G) Extended chirping song, 4 s

Synthetic signals; nothing here is measured.

Usage:
    python analysis/callers/call_anatomy_figure.py analysis/callers/runs/figures_paper/F01_song_anatomy
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from scipy.signal import spectrogram

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_page import PaperPage, BODY_PT  # noqa: E402

SR = 44100
F0 = 3000.0          # carrier, Hz
R = 40.0             # syllable rate, Hz  -> period 25 ms
DUTY = 0.45          # fraction of the period the syllable occupies
GLIDE = -500.0       # Hz across the syllable (a down-glide, as the tree crickets show)
H2 = 0.35            # amplitude of the second carrier harmonic
CH_N, CH_GAP = 3, 0.12   # syllables per chirp, gap between chirps (s)
INK = "0.15"
ACC = "tab:red"
ACC2 = "tab:blue"
FLO, FHI = 1500.0, 8000.0   # shared log-frequency axis for every spectrogram panel
# panel boxes on the page (inches from the top edge, after MBE's layout): (top, height); axes span X0..X1 across the page
X0, X1 = 1.54, 7.20
ROWS = {"A": (1.20, 0.55), "B": (2.45, 0.60), "C": (3.65, 0.75), "D": (5.00, 0.60), "E": (6.35, 0.60), "F": (7.50, 0.75), "G": (8.85, 0.60)}


def syllable(t, t0, dur, f0, glide=0.0):
    u = (t - t0) / dur
    env = np.where((u >= 0) & (u <= 1), 0.5 - 0.5 * np.cos(2 * np.pi * u), 0.0)
    f_inst = f0 + glide * (u - 0.5)
    phase = 2 * np.pi * np.cumsum(f_inst) / SR
    return env * np.sin(phase)


def trill(t, rate, duty, f0, t_start=0.0, n=None, glide=0.0):
    per = 1 / rate
    y = np.zeros_like(t); k = 0; t0 = t_start
    while t0 + per * duty <= t[-1] and (n is None or k < n):
        y += syllable(t, t0, per * duty, f0, glide); t0 += per; k += 1
    return y


def chirps(t, rate, duty, f0, n_per, gap, t_start=0.02, glide=0.0):
    per = 1 / rate
    y = np.zeros_like(t); t0 = t_start
    while t0 + n_per * per <= t[-1]:
        y += trill(t, rate, duty, f0, t_start=t0, n=n_per, glide=glide); t0 += n_per * per + gap
    return y


def song(t, kind, t_start):
    """fundamental with a glide plus a weaker 2F harmonic with twice the glide (a tooth-strike carrier)."""
    if kind == "trill":
        return trill(t, R, DUTY, F0, t_start=t_start, glide=GLIDE) + H2 * trill(t, R, DUTY, 2 * F0, t_start=t_start, glide=2 * GLIDE)
    return chirps(t, R, DUTY, F0, CH_N, CH_GAP, t_start=t_start, glide=GLIDE) + H2 * chirps(t, R, DUTY, 2 * F0, CH_N, CH_GAP, t_start=t_start, glide=2 * GLIDE)


def brace(ax, x0, x1, y, text, color=INK, dy=0.0, above=True, fs=BODY_PT, ha="center"):
    ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", color=color, lw=0.8, shrinkA=0, shrinkB=0))
    xt = (x0 + x1) / 2 if ha == "center" else (x1 if ha == "left" else x0)
    ax.text(xt, y + (0.06 if above else -0.06) + dy, text, ha=ha, va="bottom" if above else "top", fontsize=fs, color=color)


def wave_axes(ax):
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def spec_panel(ax, y, t_ms_scale=True, nperseg=512, hop=64, vrange=55):
    """log-frequency spectrogram on the shared FLO-FHI axis; x in ms or s."""
    pad = np.zeros(nperseg // 2)                       # so the spectrogram covers the whole axis, edge to edge, aligned with the waveform
    f, tt, S = spectrogram(np.concatenate([pad, y, pad]), SR, nperseg=nperseg, noverlap=nperseg - hop, window="hann")
    tt = tt - (nperseg // 2) / SR
    Sdb = 10 * np.log10(S + 1e-12)
    x = tt * (1000 if t_ms_scale else 1)
    sel = (f >= FLO * 0.9) & (f <= FHI * 1.1)
    ax.pcolormesh(x, f[sel], Sdb[sel], cmap="magma", vmin=Sdb.max() - vrange, vmax=Sdb.max(), shading="auto", rasterized=True)
    ax.set_yscale("log"); ax.set_ylim(FLO, FHI)
    ax.yaxis.set_major_locator(FixedLocator([2000, 3000, 4000, 6000, 8000])); ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_ylabel("kHz")
    return x


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "analysis/callers/runs/figures_paper/F01_song_anatomy"
    per = 1 / R; dur = per * DUTY
    page = PaperPage(number=1, title="")
    Wc, Hc = page.size
    fig = plt.figure(figsize=(Wc, Hc))

    def panel(key):
        top, h = ROWS[key]
        ax = fig.add_axes([(X0 - 1.0) / Wc, 1 - (top - 1.0 + h) / Hc, (X1 - X0) / Wc, h / Hc])
        return ax, h

    # (A) one syllable
    ax, h = panel("A")
    t = np.arange(0, dur * 1.5, 1 / SR); y = syllable(t, dur * 0.25, dur, F0, glide=GLIDE); tm = t * 1000
    ax.plot(tm, y, color=INK, lw=0.5)
    env = np.where((t >= dur * 0.25) & (t <= dur * 1.25), 0.5 - 0.5 * np.cos(2 * np.pi * (t - dur * 0.25) / dur), 0)
    ax.plot(tm, env, color=ACC, lw=0.8, ls="--"); ax.plot(tm, -env, color=ACC, lw=0.8, ls="--")
    mid = dur * 0.75; c0 = mid - 0.25 / F0
    ax.annotate("", xy=(c0 * 1000, 1.12), xytext=((c0 + 1 / F0) * 1000, 1.12), arrowprops=dict(arrowstyle="<->", color=ACC2, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(mid * 1000, 1.25, f"one cycle of the carrier at mid-syllable: 1/F = {1000 / F0:.2f} ms (F = {F0:.0f} Hz)", ha="center", va="bottom", fontsize=BODY_PT, color=ACC2)
    ax.text(dur * 0.25 * 1000 - 0.2, -0.25, f"carrier {F0 - GLIDE / 2:.0f} Hz at start", ha="right", va="top", fontsize=BODY_PT, color=ACC2)
    ax.text(dur * 1.25 * 1000 + 0.2, -0.25, f"carrier {F0 + GLIDE / 2:.0f} Hz at the end", ha="left", va="top", fontsize=BODY_PT, color=ACC2)
    brace(ax, dur * 0.25 * 1000, dur * 1.25 * 1000, -1.3, f"syllable (pulse) duration, {dur * 1000:.1f} ms: one closing stroke of the wings", above=False)
    page.label(ax, "A", "A syllable", y=1 + 0.05 / h)
    ax.set_ylim(-2.1, 1.9); ax.set_xlim(0, tm[-1]); ax.set_xlabel("time (ms)"); wave_axes(ax)

    # (B) trill waveform, 200 ms
    T = 0.2; t = np.arange(0, T, 1 / SR); ys = song(t, "trill", 0.01)
    ax, h = panel("B")
    ax.plot(t * 1000, ys, color=INK, lw=0.3); ax.fill_between(t * 1000, np.abs(ys), -np.abs(ys), color=INK, alpha=0.15, lw=0)
    s0 = 0.01 + 2 * per
    brace(ax, s0 * 1000, (s0 + per) * 1000, 1.45, f"repeat period 1/R = {per * 1000:.0f} ms, so the rate R = {R:.0f} Hz", color=ACC)
    brace(ax, s0 * 1000, (s0 + dur) * 1000, -1.45, "on", above=False, color=ACC2)
    brace(ax, (s0 + dur) * 1000, (s0 + per) * 1000, -1.45, "off", above=False)
    ax.text((s0 + per + 0.006) * 1000, -1.75, f"duty = on / period = {DUTY:.2f}", ha="left", va="top", fontsize=BODY_PT, color=INK)
    page.label(ax, "B", "A trill: syllables repeated without pause", y=1 + 0.05 / h)
    ax.set_ylim(-2.9, 2.3); ax.set_xlim(0, T * 1000); ax.set_xlabel("time (ms)"); wave_axes(ax)

    # (C) the same 200 ms as a spectrogram, aligned with (B)
    ax, h = panel("C")
    spec_panel(ax, ys)
    ax.set_xlim(0, T * 1000); ax.set_xlabel("time (ms)")
    page.label(ax, "C", "Spectrogram of trill", y=1 + 0.05 / h)
    s3 = 0.01 + 3 * per
    ax.annotate(f"carrier band: each syllable glides {abs(GLIDE):.0f} Hz downward;\nthe band's power-weighted center is the carrier F",
                xy=((s3 + dur * 0.5) * 1000, F0 * 0.93), xytext=(6, 2000), fontsize=BODY_PT, color="w", arrowprops=dict(arrowstyle="->", color="w", lw=0.6))
    ax.annotate("second carrier harmonic at 2F: the same syllables, the same\nglide as a fraction of frequency, so the same shape on a log axis",
                xy=((s3 + per + dur * 0.5) * 1000, 2 * F0 * 0.96), xytext=(6, 4300), fontsize=BODY_PT, color="w", arrowprops=dict(arrowstyle="->", color="w", lw=0.6))
    ax.annotate("", xy=(s3 * 1000, 1700), xytext=((s3 + per) * 1000, 1700), arrowprops=dict(arrowstyle="<->", color="w", lw=0.8, shrinkA=0, shrinkB=0))
    ax.text((s3 + per / 2) * 1000, 1760, "period 1/R", ha="center", fontsize=BODY_PT, color="w")

    # (D) the trill over several seconds
    T4 = 4.0; t4 = np.arange(0, T4, 1 / SR); y4 = song(t4, "trill", 0.05)
    ax, h = panel("D")
    spec_panel(ax, y4, t_ms_scale=False, nperseg=512, hop=128)
    ax.set_xlim(0, T4); ax.set_xlabel("time (s)")
    page.label(ax, "D", f"Trilling song over {T4:.0f} s: {R:.0f} syllables per second", y=1 + 0.05 / h)

    # (E) chirps waveform, 620 ms
    T5 = 0.62; t5 = np.arange(0, T5, 1 / SR); yc = song(t5, "chirp", 0.02)
    ax, h = panel("E")
    ax.plot(t5 * 1000, yc, color=INK, lw=0.25, alpha=0.85)
    c_start = 0.02 + (CH_N * per + CH_GAP)
    brace(ax, c_start * 1000, (c_start + CH_N * per) * 1000, 1.45, f"chirp = {CH_N} syllables", color=ACC2)
    brace(ax, (c_start + CH_N * per) * 1000, (c_start + CH_N * per + CH_GAP) * 1000, 1.45, "silence between chirps", color=INK)
    brace(ax, c_start * 1000, (c_start + CH_N * per + CH_GAP) * 1000, -1.5,
          f"chirp period {(CH_N * per + CH_GAP) * 1000:.0f} ms, chirp rate {1 / (CH_N * per + CH_GAP):.1f} Hz", above=False, color=ACC2, ha="left")
    brace(ax, c_start * 1000, (c_start + per) * 1000, -1.5, "syllable period 1/R ", above=False, ha="right")
    page.label(ax, "E", "A chirp: multi-syllable groups as repeating unit", y=1 + 0.05 / h)
    ax.set_ylim(-2.9, 2.3); ax.set_xlim(0, T5 * 1000); ax.set_xlabel("time (ms)"); wave_axes(ax)

    # (F) the same 620 ms as a spectrogram, aligned with (E)
    ax, h = panel("F")
    spec_panel(ax, yc)
    ax.set_xlim(0, T5 * 1000); ax.set_xlabel("time (ms)")
    page.label(ax, "F", "Chirp spectrogram", y=1 + 0.05 / h)

    # (G) chirps over several seconds
    y7 = song(t4, "chirp", 0.05)
    ax, h = panel("G")
    spec_panel(ax, y7, t_ms_scale=False, nperseg=512, hop=128)
    ax.set_xlim(0, T4); ax.set_xlabel("time (s)")
    page.label(ax, "G", "Extended chirping song", y=1 + 0.05 / h)

    page.finish(fig, out)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
