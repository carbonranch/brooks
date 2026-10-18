"""Rate-filtered views of the F03 animals: for each of the N animals of the one-window figure (same window, same
selection and numbering as window_figure.py), 30 s of context, a short excerpt from inside the stretch of song
the detector folded, and that excerpt rate-filtered at the animal's own refined rate.

Each animal gets three panels on one log-frequency band (its fundamental band +/- PAD_HZ, the same band as its
fold in F03 C), all in dB above each row's quiet level over the whole 300 s window: (1) a context strip, 30 s
around the excerpt in 50 ms columns (block maximum of the analysis columns, so it is an envelope and does not
alias the syllables), with the detector's on-periods for the animal as a bar and the excerpt boxed; (2) the raw
excerpt as a fine display spectrogram (specdisplay: 512-point frames zero-padded to 4096, 0.73 ms steps);
(3) the same excerpt after a period-synchronous average over 2k+1 periods at the animal's rate
(streak_grid.sync_average): each pixel becomes the mean of itself and the same phase point in the k periods
before and after, so syllables repeating at that rate pass unchanged while anything else is divided by 2k+1 or
smears as its phase drifts. The excerpt is the EXCERPT_S seconds centered where the animal's rate-locked
amplitude over its band peaks inside its on-period runs. One color scale serves every panel. Where a louder
animal of the figure shares the rows (animal 4 here, a 45 Hz singer under the 29 Hz animal 3 that owns 3.0-3.9
kHz), the title says so: the raw views show the louder singer's syllables, the filtered view the animal's own.

Companion to F03 (window_figure.py), written as its own file while the paper figures are frozen (2026-10-07);
MBE can place it after F03 or it can become F03 panel (E) later.

Usage:
    python analysis/callers/window_rate_filtered_figure.py --recorder BARLT_00008662 --site House --time "2025-09-20 23:05" \
        --out analysis/callers/runs/figures_paper/F03b_rate_filtered [--n 8] [--excerpt 1.5] [--k 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_blobs2 as W2  # noqa: E402
import specdisplay as SD  # noqa: E402
from window_figure import choose_animals, excerpt_start, logband, PAD_HZ, DF  # noqa: E402
from streak_grid import sync_average  # noqa: E402

SD.paper_fonts()
STRIP_COL_S = 0.05       # context strip columns: block maximum over 50 ms of the analysis grid


def title(ax, text, chars, fs=9, bold=False, color=None):
    ax.set_title(textwrap.fill(text, chars), loc="left", fontsize=fs, fontweight="bold" if bold else "normal", color=color)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder", required=True); ap.add_argument("--site", required=True); ap.add_argument("--time", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--excerpt", type=float, default=1.5, help="excerpt length (s)")
    ap.add_argument("--k", type=int, default=4, help="periods each side in the period-synchronous average (2k+1 in all). 4: in this window a 29 Hz "
                    "singer 5 dB louder than the 45 Hz animal sharing its rows drops from 23 to 14 dB over the median while the 45 Hz rhythm rises from 18 to 36 dB; k = 2 leaves it at 19")
    ap.add_argument("--context", type=float, default=30.0, help="length of the context strip (s)")
    ap.add_argument("--vmax", type=float, default=None, help="color ceiling (dB above quiet); default the 99.5th percentile of the raw excerpts")
    args = ap.parse_args()
    t = pd.Timestamp(args.time, tz=W2.TZ)
    sig, sr, t0, _ = W2.read_window(args.recorder, args.site, t, 300.0)
    X = W2.analyse_signal(sig, sr, 300.0)
    f, dt, P, B, per_blob, A = (X[k] for k in ("f", "dt", "P", "B", "per_blob", "A"))
    t0 = t0.tz_convert(W2.TZ)
    chosen, num, col, dropped, D, parts, _, nondom = choose_animals(X, args.n)
    quiet_w = np.percentile(10 * np.log10(P + 1e-20), 10, axis=1)       # the whole window's quiet level per analysis row
    Aidx = A.set_index("animal")

    L = 10 * np.log10(P + 1e-20) - quiet_w[:, None]                     # analysis grid, dB above each row's quiet level
    panels = []
    for p in chosen:
        d = per_blob[p.blob]; R = float(p.R)
        s0 = excerpt_start(d, f, p, dt, args.excerpt, 300.0)
        i0, i1 = int(s0 * sr), int((s0 + args.excerpt) * sr)
        fd, td, Pd = SD.fine_spec(sig[i0:i1], sr)
        Dd = SD.db_above_quiet(Pd, SD.quiet_on_grid(fd, f, quiet_w))
        lo, hi = p.f_c - PAD_HZ, p.f_c + PAD_HZ
        band = (fd >= lo) & (fd <= hi)
        filt = sync_average(Dd, td[1] - td[0], R, args.k)
        # the context strip: CONTEXT_S around the excerpt, block MEAN over STRIP_COL_S of the analysis columns (an envelope
        # of level, as F03 A's 0.5 s columns: a 30 Hz train cannot be drawn in 50 ms columns, and the block maximum
        # saturates every pulsing row)
        c0 = float(np.clip(s0 + args.excerpt / 2 - args.context / 2, 0, 300 - args.context))
        j0, j1 = int(c0 / dt), int((c0 + args.context) / dt)
        rows = np.flatnonzero((f >= lo - DF) & (f <= hi + DF))
        j1 = min(j1, L.shape[1])                                             # the STFT has no columns past the last full frame
        k = max(1, int(round(STRIP_COL_S / dt))); n = (j1 - j0) // k
        strip = L[rows][:, j0:j0 + n * k].reshape(len(rows), n, k).mean(axis=2)
        # animals of the figure whose fundamental band shares rows with this one and whose rhythm is the stronger one
        # in the shared rows over this animal's on-periods (band loudness cannot decide: the quieter singer's rows
        # are lit by the louder one, so animal 4 here ranks above animal 3 by loudness while 3 owns every row)
        shares = []
        for q in chosen:
            if q.animal == p.animal or q.f_lo > p.f_hi or q.f_hi < p.f_lo:
                continue
            rr = np.flatnonzero((f >= max(p.f_lo, q.f_lo)) & (f <= min(p.f_hi, q.f_hi)))
            own = np.mean([W2.locked_series(P[rr, sl], dt, R).mean() for sl in d["cols"]])
            theirs = np.mean([W2.locked_series(P[rr, sl], dt, float(q.R)).mean() for sl in d["cols"]])
            if theirs > own:
                shares.append(q)
        panels.append(dict(p=p, R=R, s0=s0, td=td, fd=fd[band], raw=Dd[band], filt=filt[band], lo=lo, hi=hi,
                           c0=c0, strip=strip, strip_f=f[rows], strip_dt=k * dt, runs=[(sl.start * dt, sl.stop * dt) for sl in d["cols"]],
                           shares=shares, rc=Aidx.loc[p.animal, "chirp_rate"] if p.animal in Aidx.index else np.nan))
    vmax = args.vmax or max(6.0, float(np.percentile(np.concatenate([q["raw"].ravel() for q in panels]), 99.5)))

    per_row = 2                                   # animals per row, three panels each: context strip, raw excerpt, filtered excerpt
    nrow = int(np.ceil(len(panels) / per_row))
    fig = plt.figure(figsize=(18, 2.5 * nrow + 1.6))
    gs = fig.add_gridspec(nrow, per_row * 3, width_ratios=[2.3, 1, 1] * per_row, hspace=1.0, wspace=0.25,
                          left=0.04, right=0.99, top=0.865, bottom=0.065)
    for i, q in enumerate(panels):
        p = q["p"]; r, cbase = i // per_row, (i % per_row) * 3
        unit = "chirps" if (pd.notna(q["rc"]) and abs(q["rc"] / q["R"] - 1) < 0.1) else "syllables"   # a chirper read at its chirp rate alone
        extra = f"; chirps at {q['rc']:.1f} Hz" if pd.notna(q["rc"]) else ""
        share_note = ("; shares rows with the stronger " + ", ".join(f"{num[s.animal]} ({s.R:.2f} Hz)" for s in q["shares"])) if q["shares"] else ""
        # context strip
        ax = fig.add_subplot(gs[r, cbase])
        te = q["c0"] + np.arange(q["strip"].shape[1] + 1) * q["strip_dt"]
        fe = np.concatenate([[q["strip_f"][0] - DF / 2], q["strip_f"] + DF / 2])
        ax.pcolormesh(te, fe, q["strip"], cmap="magma", vmin=0, vmax=vmax, shading="flat", rasterized=True)
        logband(ax, q["lo"], q["hi"])
        ax.axhline(p.f_lo - DF / 2, color="w", lw=0.5, ls=":"); ax.axhline(p.f_hi + DF / 2, color="w", lw=0.5, ls=":")
        ax.add_patch(plt.Rectangle((q["s0"], q["lo"]), args.excerpt, q["hi"] - q["lo"], fill=False, color="w", lw=1.2))
        for a, b in q["runs"]:
            a, b = max(a, q["c0"]), min(b, q["c0"] + args.context)
            if b > a:
                ax.plot([a, b], [q["hi"] * 0.985] * 2, color=col[p.animal], lw=4, solid_capstyle="butt")
        ax.set_xlim(q["c0"], q["c0"] + args.context); ax.tick_params(labelsize=7); ax.set_ylabel("kHz", fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(col[p.animal]); sp.set_linewidth(2.5)
        title(ax, f"{num[p.animal]}:  {p.f_c:.0f} Hz @ {q['R']:.2f} Hz, period {1000 / q['R']:.1f} ms{extra}; {p.loud_db:.0f} dB above quiet, on {p.on_frac:.0%}{share_note}. "
                  f"{args.context:g} s of the window; bar = on-periods, box = the excerpt", 78, color=col[p.animal])
        if r == nrow - 1:
            ax.set_xlabel("seconds into the window", fontsize=8)
        for j, (img, what) in enumerate(((q["raw"], f"{args.excerpt:g} s from {q['s0']:.1f} s, raw"),
                                         (q["filt"], f"rate-filtered at {q['R']:.2f} Hz: {unit} at that rate pass"))):
            ax = fig.add_subplot(gs[r, cbase + 1 + j])
            ax.pcolormesh(q["td"], q["fd"], img, cmap="magma", vmin=0, vmax=vmax, shading="auto", rasterized=True)
            logband(ax, q["lo"], q["hi"]); ax.set_yticklabels([])
            ax.axhline(p.f_lo - DF / 2, color="w", lw=0.5, ls=":"); ax.axhline(p.f_hi + DF / 2, color="w", lw=0.5, ls=":")
            ax.set_xlim(0, args.excerpt); ax.tick_params(labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(col[p.animal]); sp.set_linewidth(2.5)
            title(ax, what, 30, color=col[p.animal])
            if r == nrow - 1:
                ax.set_xlabel("s", fontsize=8)
    for j in range(len(panels), nrow * per_row):
        for k in range(3):
            fig.add_subplot(gs[j // per_row, (j % per_row) * 3 + k]).set_axis_off()

    fig.suptitle(textwrap.fill(
        f"The {len(panels)} animals of the one-window figure ({args.site} {args.recorder[-4:]}, {t0:%d %B %Y}, the 300 s centered on {t:%H:%M}), each in three views on its "
        f"fundamental band +/- {PAD_HZ:.0f} Hz (dotted lines = the band), dB above each row's quiet level over the whole window, one color scale to {vmax:.0f} dB. "
        f"Left: {args.context:g} s of the window around the excerpt, columns of {STRIP_COL_S * 1000:.0f} ms (mean level, an envelope), the bar at the top the detector's "
        f"on-periods for the animal (its fold and refined rate come from these), the box the excerpt. Middle: the excerpt, {args.excerpt:g} s centered where the animal's "
        f"rate-locked amplitude over its band peaks inside its on-periods. Right: the same excerpt after a period-synchronous average over {2 * args.k + 1} periods at the "
        f"animal's refined rate: each pixel is the mean of itself and the same phase point {args.k} periods before and after, so what repeats at that rate is kept and "
        f"everything else is divided by {2 * args.k + 1} or smeared. Where another animal of the figure shares the rows with the stronger rhythm there, the raw views show its syllables and the filtered view the animal's own.", 250),
        fontsize=10, fontweight="bold", x=0.04, ha="left", y=0.99)
    fig.savefig(args.out + ".pdf", bbox_inches="tight"); fig.savefig(args.out + ".png", dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}.pdf / .png")
    for q in panels:
        p = q["p"]; print(f"  {num[p.animal]}: animal {p.animal}, {p.f_lo:.0f}-{p.f_hi:.0f} Hz @ {q['R']:.2f}, excerpt from {q['s0']:.1f} s")


if __name__ == "__main__":
    main()
