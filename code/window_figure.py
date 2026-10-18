"""One 300 s window, as BROOKS sees it: the spectrogram, the modulation map with its peaks and the animals they
resolve into, and for each of the N chosen animals a zoom of the modulation map around its peak beside a short
excerpt of the audio, raw and rate-filtered at the animal's rate. From the v2 detector
(window_blobs2.analyse_signal), drawn with the display spectrograms of specdisplay. Chains from the night
overview figure (same recorder, night and window).

  (A) the window at 2048-point frames (21.5 Hz rows) in 0.093 s columns (mean level), dB above each row's quiet
      level over the window, percentile color scale; log frequency
  (B) the modulation map: excess modulation power (dB) over carrier (log) x rate (log); the N animals'
      fundamental parts outlined solid in their colors, their absorbed rate harmonics dotted, unabsorbed rate
      harmonics (a peak at m x R in the animal's rows that the detector left as a separate part) dashed; every
      other part a thin gray outline
  (C) per animal, three panels: its modulation spectrum (the map's excess averaged over its fundamental rows,
      against rate on a log axis, with its rate, the rate's multiples, the peak threshold and the rates of other
      shown animals sharing its rows marked); a 1.5 s excerpt from inside its on-periods, raw; the same excerpt
      rate-filtered at its refined rate (period-synchronous average over 2k+1 periods, streak_grid.sync_average).
      The excerpts are dB above each row's quiet level over the window on one color scale.

The N animals are the loudest among those whose OWN rhythm is the strongest one in their fundamental rows
(rate-locked amplitude over their on-periods against every other fundamental part sharing those rows): a quieter
singer under a louder one in the same rows is real in the map but cannot be shown in a spectrogram excerpt, raw
or filtered (the 45 Hz animal of this window under the 29 Hz one came out as a moire of the louder one; MBE,
2026-10-08), so such animals are counted in the caption and left to the map. Carrier harmonics (2x, 3x at the
same rate), same-band same-rate twins, split pieces and rate harmonics of a chosen animal are skipped as before.

Redesign of 2026-10-08 (MBE: "(A) higher resolution, (B) the same, then for each chosen peak a zoomed-in region
along with a modulation map for each"); the fold panel and the on/off heat map of the first version are gone
(the folds live in F03b's companion views and the on/off series in METHOD 19 O).

Usage:
    python analysis/callers/window_figure.py --recorder BARLT_00008662 --site House --time "2025-09-20 23:05" \
        --out analysis/callers/runs/figures_paper/F03_window [--n 8] [--fmin 1500 --fmax 12000] [--excerpt 1.5] [--k 4]
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
import scipy.signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_blobs2 as W2  # noqa: E402
import specdisplay as SD  # noqa: E402
import postfilter  # noqa: E402
from streak_grid import sync_average  # noqa: E402

SD.paper_fonts()
PAD_HZ = 600.0
DF = 44100 / 512
CHIRP_HZ = 10.0          # a rate below this is a chirp rate
CHIRP_HARM_TOL = 0.08    # relative tolerance for a chirp-rate harmonic (the detector's 1.5 % misses them when the rate wanders)
SPLIT_GAP_HZ = 600.0     # same-rate pieces this close in carrier are one singer split by a neighbor's rows
A_NPERSEG, A_COL_S = 2048, 0.093   # panel A: 21.5 Hz rows, columns of 4 half-overlapped frames (0.093 s), mean level


def title(ax, text, chars, fs=10, bold=True, color=None):
    kw = {"color": color} if color is not None else {}
    ax.set_title(textwrap.fill(text, chars), loc="left", fontsize=fs, fontweight="bold" if bold else "normal", **kw)


def khz_log(ax, lo, hi, ticks=(500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 20000)):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    tk = [t for t in ticks if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))


def logband(ax, lo, hi):
    """Log-frequency axis for a narrow band: three to five round ticks."""
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ticks = [v for v in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 10000, 12000] if lo <= v <= hi]
    if len(ticks) < 3:
        ticks = [round(v, -2) for v in np.geomspace(lo * 1.03, hi * 0.97, 3)]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))


def rate_log(ax, lo, hi):
    ax.set_xscale("log"); ax.set_xlim(lo, hi)
    cands = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    tk = [v for v in cands if lo <= v <= hi]
    if len(tk) < 2:
        tk = [round(v, 1) for v in np.geomspace(lo * 1.05, hi * 0.95, 3)]
    ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))


def excerpt_start(d: dict, f: np.ndarray, p, dt: float, excerpt_s: float, dur: float) -> float:
    """Start (s) of an animal's excerpt: centered on the SLOW_S chunk of its rate-locked series, averaged over
    its band rows, that is highest inside its on-period runs."""
    fk = f[d["p_lo"]:d["p_hi"] + 1]
    rows = np.flatnonzero((fk >= p.f_lo) & (fk <= p.f_hi))
    s = d["raw"][rows].mean(axis=0) if len(rows) else d["raw"].mean(axis=0)
    chunk = W2.SLOW_S / dt
    inside = np.zeros(len(s), bool)
    for sl in d["cols"]:
        inside[int(sl.start / chunk): int(np.ceil(sl.stop / chunk))] = True
    if not inside.any():
        inside[:] = True
    j = int(np.argmax(np.where(inside, s, -np.inf)))
    c = (j + 0.5) * W2.SLOW_S
    return float(np.clip(c - excerpt_s / 2, 0, dur - excerpt_s))


MIN_LOUD_DB = 8.0        # an animal below this (75th percentile of its band's mean dB above quiet) is too faint for an excerpt to show


def choose_animals(X: dict, n: int, dominant: bool = True, fmax: float = 12000.0, min_loud: float = MIN_LOUD_DB):
    """The figure's N animals from an analysed window X (window_blobs2.analyse_signal): one fundamental part per
    animal, ranked by loudness, skipping carrier harmonics, same-band twins, split pieces and rate harmonics of
    one already chosen, and (dominant=True) animals whose rhythm is not the strongest in their own rows; numbered
    1..N by carrier. Returns (chosen parts, {animal: number}, {animal: color}, dropped (part, kept part) pairs,
    D = dB above each row's quiet level, parts with a rate column, rate_harmonic test, non-dominant parts skipped)."""
    f, P, B, parts, A, fr, excess = (X[k] for k in ("f", "P", "B", "parts", "A", "fr", "excess"))
    L = 10 * np.log10(P + 1e-20)
    quiet = np.percentile(L, 10, axis=1, keepdims=True)
    D = L - quiet
    parts = parts.copy(); parts["R"] = parts.blob.map(B.R_fine)

    # the N strongest animals: one fundamental part per animal (its lowest band), deepest first,
    # skipping an animal that is a carrier harmonic (2x, 3x, same rate) or a same-band same-rate twin of one already chosen
    # rank animals by LOUDNESS (mean dB above quiet over the fundamental part's rows), not by modulation
    # depth: a loud, nearly continuous singer has a low relative depth (the 2.5 kHz tree cricket here, 0.30)
    # and was left out when depth decided (MBE, 2026-10-06)
    def loud(p):
        rows = np.flatnonzero((f >= p.f_lo) & (f <= p.f_hi))
        return float(np.percentile(D[rows].mean(axis=1), 75)) if len(rows) else 0.0   # the band's louder rows, so a wide band is not penalized
    # harmonics-as-own and shadows (postfilter, by band power) are not candidates
    Af = postfilter.flag_harmonics(A.copy(), parts)
    bad = set(Af.animal[(Af.harm_of >= 0) | (Af.sub_of >= 0)])
    fund = parts[(parts.am_harm_of < 0) & ~parts.animal.isin(bad)].sort_values("f_c").groupby("animal").first().reset_index()
    fund = fund[fund.f_hi + PAD_HZ <= fmax]                            # the figure's band (an animal at its top edge is usually a chorus member's carrier harmonic)
    fund["loud_db"] = [loud(p) for p in fund.itertuples()]
    fund = fund[fund.loud_db >= min_loud].sort_values("loud_db", ascending=False)

    dfr = fr[1] - fr[0]

    def map_strength(rows, R):
        """Median over rows of the map's excess at rate R (the best of +/- 2 bins): how strongly R rules those rows
        over the whole window."""
        m = (fr >= R - 2.5 * dfr) & (fr <= R + 2.5 * dfr)
        return float(np.median(excess[rows][:, m].max(axis=1))) if len(rows) and m.any() else 0.0

    def dominant_in_own_rows(p):
        """p's rhythm is the strongest one in its OWN fundamental rows: no other fundamental part's rate has a
        higher median excess over those rows. Judged on the modulation map (0.17 Hz resolution over the whole
        window), not on band loudness, which cannot decide (a quiet singer's rows are lit by the louder one: the
        45 Hz animal here ranked second of 29 by loudness while the 29 Hz animal owned every row), and not on
        0.5 s rate-locked series, which cannot separate chorus members 1 Hz apart or hold a 3 Hz chirp rhythm."""
        rows = np.flatnonzero((f >= p.f_lo) & (f <= p.f_hi))
        own = map_strength(rows, float(p.R))
        for q in fund.itertuples():
            if q.animal == p.animal or q.f_lo > p.f_hi or q.f_hi < p.f_lo or abs(q.R / p.R - 1) < 0.03:
                continue
            if map_strength(rows, float(q.R)) > own:
                return False
        return True

    def rate_harmonic(p, q):
        """p's rate is an integer multiple (2..6) of q's in overlapping rows. The detector's rule is
        R_TOL (1.5 %) on each peak's own refined rate; a chirp rate wanders ~5 % within a window
        (here 3.16 Hz in one 30 s run, 2.99 Hz in another), so below CHIRP_HZ the tolerance is
        CHIRP_HARM_TOL (MBE, 2026-10-06: 8 was the 2x chirp-rate harmonic of 7)."""
        m = p.R / q.R
        tol = CHIRP_HARM_TOL if q.R < CHIRP_HZ else 0.03
        return abs(m - round(m)) < tol * round(m) and 2 <= round(m) <= 6 and p.f_lo <= q.f_hi + DF and p.f_hi >= q.f_lo - DF

    chosen, dropped, nondom = [], [], []
    for p in fund.itertuples():
        dup = False
        for q in chosen:
            ratio = p.f_c / q.f_c; m = round(ratio)
            same_rate = abs(p.R / q.R - 1) < 0.03
            if same_rate and ((m in (2, 3) and abs(ratio / m - 1) < 0.08) or abs(np.log(ratio)) < 0.06):
                dup = True
            # the same rate to 0.5 % in a band within SPLIT_GAP_HZ of the other's: one singer whose band a
            # neighbor's rows interrupted (2554 and 2976 Hz @ 38.98/38.96 here: the 42.5 Hz singer owns the
            # rows between; the upper piece sang 47-93 s only and its on/off series did not co-vary enough to link)
            gap = max(p.f_lo - q.f_hi, q.f_lo - p.f_hi)
            if abs(p.R / q.R - 1) < 0.005 and gap <= SPLIT_GAP_HZ:
                dup = True; dropped.append((p, q))
            if rate_harmonic(p, q):
                dup = True; dropped.append((p, q))
        if not dup and dominant and not dominant_in_own_rows(p):
            nondom.append(p); dup = True
        if not dup:
            chosen.append(p)
        if len(chosen) >= n:
            break
    chosen = sorted(chosen, key=lambda p: p.f_c)                      # numbered 1..N by carrier
    num = {p.animal: i + 1 for i, p in enumerate(chosen)}
    cmap = plt.get_cmap("tab10")
    col = {p.animal: cmap(i % 10) for i, p in enumerate(chosen)}
    return chosen, num, col, dropped, D, parts, rate_harmonic, nondom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder", required=True); ap.add_argument("--site", required=True); ap.add_argument("--time", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--fmin", type=float, default=1500.0); ap.add_argument("--fmax", type=float, default=12000.0)
    ap.add_argument("--excerpt", type=float, default=1.5, help="excerpt length (s)")
    ap.add_argument("--k", type=int, default=4, help="periods each side in the period-synchronous average (2k+1 in all)")
    ap.add_argument("--no-dominant", action="store_true", help="do not skip animals whose rhythm is not the strongest in their own rows")
    args = ap.parse_args()
    t = pd.Timestamp(args.time, tz=W2.TZ)
    sig, sr, t0, _ = W2.read_window(args.recorder, args.site, t, 300.0)
    X = W2.analyse_signal(sig, sr, 300.0)
    f, dt, P, fr, excess, B, per_blob, A = (X[k] for k in ("f", "dt", "P", "fr", "excess", "B", "per_blob", "A"))
    t0 = t0.tz_convert(W2.TZ)
    print(f"{len(B)} peaks -> {len(X['parts'])} parts -> {len(A)} animals")
    chosen, num, col, dropped, D, parts, rate_harmonic, nondom = choose_animals(X, args.n, dominant=not args.no_dominant, fmax=args.fmax)
    for p, q in dropped:
        print(f"  dropped animal {p.animal} ({p.f_lo:.0f}-{p.f_hi:.0f} Hz @ {p.R:.2f}): rate x{p.R / q.R:.3f} of animal {q.animal} ({q.f_lo:.0f}-{q.f_hi:.0f} Hz @ {q.R:.2f})")
    for p in nondom:
        print(f"  skipped animal {p.animal} ({p.f_lo:.0f}-{p.f_hi:.0f} Hz @ {p.R:.2f}, {p.loud_db:.1f} dB): not the strongest rhythm in its own rows")
    quiet_w = np.percentile(10 * np.log10(P + 1e-20), 10, axis=1)       # the whole window's quiet level per analysis row
    Aidx = A.set_index("animal")

    # ---- panel A data: finer frames, mean-level columns, dB above each row's quiet level ----
    fA, tA, ZA = scipy.signal.stft(sig, fs=sr, window="hann", nperseg=A_NPERSEG, noverlap=A_NPERSEG // 2, boundary=None, padded=False)
    LA = 10 * np.log10(np.abs(ZA) ** 2 + 1e-20); LA -= np.percentile(LA, 10, axis=1, keepdims=True)
    kA = max(1, int(round(A_COL_S / (tA[1] - tA[0])))); nA = LA.shape[1] // kA
    DA = LA[:, :nA * kA].reshape(LA.shape[0], nA, kA).mean(axis=2); tAe = np.arange(nA + 1) * kA * (tA[1] - tA[0])
    selA = (fA >= args.fmin) & (fA <= args.fmax); dfA = fA[1] - fA[0]

    # ---- per-animal excerpts ----
    panels = []
    for p in chosen:
        d = per_blob[p.blob]; R = float(p.R)
        s0 = excerpt_start(d, f, p, dt, args.excerpt, 300.0)
        i0, i1 = int(s0 * sr), int((s0 + args.excerpt) * sr)
        fd, td, Pd = SD.fine_spec(sig[i0:i1], sr)
        Dd = SD.db_above_quiet(Pd, SD.quiet_on_grid(fd, f, quiet_w))
        lo, hi = p.f_c - PAD_HZ, p.f_c + PAD_HZ
        band = (fd >= lo) & (fd <= hi)
        panels.append(dict(p=p, R=R, s0=s0, td=td, fd=fd[band], raw=Dd[band], filt=sync_average(Dd, td[1] - td[0], R, args.k)[band], lo=lo, hi=hi,
                           rc=Aidx.loc[p.animal, "chirp_rate"] if p.animal in Aidx.index else np.nan))
    vmax_ex = max(6.0, float(np.percentile(np.concatenate([q["raw"].ravel() for q in panels]), 99.5)))
    vmax_map = max(15, W2.THRESH_DB * 2.5)

    per_row = 2
    nrow = int(np.ceil(len(panels) / per_row))
    H = [4.6, 0.5, 4.6, 0.45] + [2.4] * nrow
    fig = plt.figure(figsize=(18, 1.05 * sum(H) + 1.2))
    gs = fig.add_gridspec(len(H), per_row * 3, height_ratios=H, width_ratios=[1.15, 1, 1] * per_row, hspace=0.75, wspace=0.3)

    # (A) the window
    ax = fig.add_subplot(gs[0, :])
    vlo, vhi = np.percentile(DA[selA], 5), np.percentile(DA[selA], 99.7)
    ax.pcolormesh(tAe, np.concatenate([[fA[selA][0] - dfA / 2], fA[selA] + dfA / 2]), DA[selA], cmap="magma", vmin=vlo, vmax=vhi, shading="flat", rasterized=True)
    khz_log(ax, args.fmin, args.fmax)
    ax.set_xlim(0, 300); ax.set_xlabel("seconds into the window"); ax.set_ylabel("kHz")
    # the chosen animals' fundamental bands bracketed at the right edge, numbers pushed apart where bands overlap
    ys = np.log([p.f_c for p in chosen]); min_sep = 0.075
    for _ in range(50):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_sep:
                mid = (ys[i] + ys[i - 1]) / 2; ys[i - 1], ys[i] = mid - min_sep / 2, mid + min_sep / 2
    for p, y in zip(chosen, np.exp(ys)):
        ax.plot([301.5, 301.5], [p.f_lo - DF / 2, p.f_hi + DF / 2], color=col[p.animal], lw=3, clip_on=False, solid_capstyle="butt")
        ax.text(304, y, str(num[p.animal]), color=col[p.animal], fontsize=8, fontweight="bold", va="center", clip_on=False)
    skip_note = f"; {len(nondom)} loud animals whose rhythm is not the strongest in their own rows are left to the map" if nondom else ""
    title(ax, f"(A)  one 300 s window: {args.site} {args.recorder[-4:]}, {t0:%d %b %Y}, {t0:%H:%M:%S} to {(t0 + pd.Timedelta(seconds=300)):%H:%M:%S}. "
              f"{A_NPERSEG}-point frames ({sr / A_NPERSEG:.1f} Hz rows), {A_COL_S:.3f} s columns of mean level, dB above each row's quiet level over the window. "
              f"{len(A)} animals were found; the {len(chosen)} shown, numbered by carrier at the right, are the loudest (>= {MIN_LOUD_DB:.0f} dB above their rows' quiet level) "
              f"whose own rhythm dominates their rows{skip_note}.", 190)

    # (B) the modulation map
    fig.add_subplot(gs[1, :]).set_axis_off()
    ax = fig.add_subplot(gs[2, :])
    sel = (f >= args.fmin) & (f <= args.fmax)
    fe = np.concatenate([[f[sel][0] - DF / 2], f[sel] + DF / 2])
    re_ = np.concatenate([[fr[0] - (fr[1] - fr[0]) / 2], fr + (fr[1] - fr[0]) / 2])
    ax.pcolormesh(re_, fe, excess[sel], cmap="gray", vmin=0, vmax=vmax_map, shading="flat", rasterized=True)
    ax.set_xscale("log"); khz_log(ax, args.fmin, args.fmax)
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50, 100])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))

    def box(ax_, p, color, lw, ls):
        b = B.loc[p.blob]
        ax_.add_patch(plt.Rectangle((b.am_lo * 0.96, p.f_lo - DF / 2), b.am_hi * 1.04 - b.am_lo * 0.96, p.f_hi - p.f_lo + DF, fill=False, lw=lw, ls=ls, color=color))

    def outline_parts(ax_, only_animal=None):
        """Outlines on a map axis: chosen animals' fundamentals solid, absorbed harmonics dotted, unabsorbed rate multiples
        dashed, everything else thin gray. only_animal: color only that animal's parts, the rest gray. Returns the dashed count."""
        n_missed = 0
        for p in parts.itertuples():
            if p.animal in col and (only_animal is None or p.animal == only_animal):
                box(ax_, p, col[p.animal], 1.4 if p.am_harm_of < 0 else 0.9, "-" if p.am_harm_of < 0 else ":")
            else:
                missed = None
                for q in chosen:
                    if (only_animal is None or q.animal == only_animal) and rate_harmonic(p, q):
                        missed = q.animal
                if missed is not None:
                    box(ax_, p, col[missed], 1.0, "--"); n_missed += 1
                else:
                    box(ax_, p, "0.55", 0.5, "-")
        return n_missed

    n_missed = outline_parts(ax)
    ax.set_xlim(fr[0], fr[-1]); ax.set_xlabel("modulation rate (Hz)"); ax.set_ylabel("carrier (kHz)")
    n_harm = int(parts.am_harm_of.ge(0).sum())
    title(ax, f"(B)  the modulation map: excess modulation power (dB) per carrier row and rate. {len(B)} peaks above {W2.THRESH_DB:.0f} dB resolve into "
              f"{len(parts)} parts and {len(A)} animals; {n_harm} parts are rate harmonics absorbed into their fundamental. Solid outlines in color: the {len(chosen)} animals' "
              f"fundamentals; dotted: their absorbed harmonics; dashed: {n_missed} peaks at a multiple of an animal's rate in its rows that were not absorbed; gray: all other parts.", 190)

    # (C) per animal: map zoom, raw excerpt, rate-filtered excerpt
    axc = fig.add_subplot(gs[3, :]); axc.set_axis_off()
    bb = axc.get_position()
    fig.text(bb.x0, bb.y0 + bb.height * 0.1, textwrap.fill(
        f"(C)  each animal: its modulation spectrum, the map's excess (dB above each row's running-median baseline) averaged over its fundamental rows against rate, "
        f"with its rate (solid), the rate's multiples where absorbed harmonics sit (dotted), the {W2.THRESH_DB:.0f} dB peak threshold (black dotted) and, dashed in their colors, "
        f"the rates of other shown animals whose bands touch these rows; {args.excerpt:g} s of the window from inside its on-periods, "
        f"centered where its rate-locked amplitude over its band peaks, raw (dB above each row's quiet level over the window, one scale to {vmax_ex:.0f} dB; dotted lines = the band); "
        f"the same {args.excerpt:g} s rate-filtered at its refined rate: a period-synchronous average over {2 * args.k + 1} periods, each pixel the mean of itself and the same phase "
        f"point {args.k} periods before and after, so what repeats at that rate is kept and everything else is divided by {2 * args.k + 1} or smeared.", 220),
        fontsize=10, fontweight="bold", va="bottom")
    for i, q in enumerate(panels):
        p = q["p"]; r, c0 = 4 + i // per_row, (i % per_row) * 3
        c = col[p.animal]
        unit = "chirps" if (pd.notna(q["rc"]) and abs(q["rc"] / q["R"] - 1) < 0.1) else "syllables"
        extra = f"; chirps at {q['rc']:.1f} Hz" if pd.notna(q["rc"]) else ""
        # the modulation spectrum of the animal's rows: the map's excess averaged over its fundamental rows, against rate
        # (MBE, 2026-10-08: "a modulation graph like you showed ... instead of the zoomed in modulation map")
        ax = fig.add_subplot(gs[r, c0])
        rows = np.flatnonzero((f >= p.f_lo) & (f <= p.f_hi))
        ex = excess[rows].mean(axis=0)
        ax.plot(fr, ex, color=c, lw=1.0)
        ax.axhline(W2.THRESH_DB, color="k", ls=":", lw=0.8)
        for m in range(2, 7):                                            # the rate's multiples, where its absorbed harmonics sit
            if q["R"] * m <= fr[-1]:
                ax.axvline(q["R"] * m, color=c, lw=0.6, ls=":", alpha=0.7)
        ax.axvline(q["R"], color=c, lw=1.2, alpha=0.8)
        for o in chosen:                                                 # other shown animals whose bands touch these rows
            if o.animal != p.animal and o.f_lo <= p.f_hi and o.f_hi >= p.f_lo:
                ax.axvline(float(o.R), color=col[o.animal], lw=0.8, ls="--", alpha=0.8)
                ax.text(float(o.R), ax.get_ylim()[1] if False else ex.max() * 0.98, str(num[o.animal]), color=col[o.animal], fontsize=7, ha="center", va="top")
        rate_log(ax, fr[0], fr[-1]); ax.tick_params(labelsize=7)
        ax.set_ylim(min(-3, ex.min() - 1), max(ex.max() * 1.12, W2.THRESH_DB * 1.5))
        ax.grid(alpha=0.25, which="major")
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2.5)
        ax.set_ylabel("excess (dB)", fontsize=8)
        title(ax, f"{num[p.animal]}:  {p.f_c:.0f} Hz @ {q['R']:.2f} Hz, {1000 / q['R']:.1f} ms{extra}; {p.loud_db:.0f} dB, on {p.on_frac:.0%}",
              40, fs=8.5, bold=False, color=c)
        if r == len(H) - 1:
            ax.set_xlabel("modulation rate (Hz)", fontsize=8)
        for j, (img, what) in enumerate(((q["raw"], f"{args.excerpt:g} s from {q['s0']:.1f} s, raw"),
                                         (q["filt"], f"rate-filtered at {q['R']:.2f} Hz: {unit} at that rate pass"))):
            ax = fig.add_subplot(gs[r, c0 + 1 + j])
            ax.pcolormesh(q["td"], q["fd"], img, cmap="magma", vmin=0, vmax=vmax_ex, shading="auto", rasterized=True)
            logband(ax, q["lo"], q["hi"]); ax.set_yticklabels([])
            ax.axhline(p.f_lo - DF / 2, color="w", lw=0.5, ls=":"); ax.axhline(p.f_hi + DF / 2, color="w", lw=0.5, ls=":")
            ax.set_xlim(0, args.excerpt); ax.tick_params(labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(c); sp.set_linewidth(2.5)
            title(ax, what, 34, fs=8.5, bold=False, color=c)
            if r == len(H) - 1:
                ax.set_xlabel("s", fontsize=8)
    for j in range(len(panels), nrow * per_row):
        for k in range(3):
            fig.add_subplot(gs[4 + j // per_row, (j % per_row) * 3 + k]).set_axis_off()

    fig.suptitle(f"One window: {args.site} {args.recorder[-4:]}, {t0:%d %B %Y}, the 300 s centered on {t:%H:%M}", fontsize=12)
    fig.savefig(args.out + ".pdf", bbox_inches="tight"); fig.savefig(args.out + ".png", dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}.pdf / .png")
    for q in panels:
        p = q["p"]; print(f"  {num[p.animal]}: animal {p.animal}, {p.f_lo:.0f}-{p.f_hi:.0f} Hz, F_c {p.f_c:.0f}, R {q['R']:.2f}, {p.loud_db:.1f} dB, depth {p.depth:.2f}, on {p.on_frac:.2f}, excerpt from {q['s0']:.1f} s")


if __name__ == "__main__":
    main()
