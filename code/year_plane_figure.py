"""Figure 7: the year on the carrier-rate plane.

  (A) every singer-hour of the year (one point per hour of a track in a recorder-night), syllable rate against
      reference carrier, colored by the HRRR air temperature at the hour. The same axes and color meaning as
      Fig 5B, so one night's clusters become the year's lines.
  (B) the same points colored by streak for the streaks with >= MIN_N singer-nights, each labeled once with its
      paper number (paper_ids.py: types numbered 1..N by carrier at 25 degC) at its center of mass; the rest gray.
  (C) each streak as a fitted line (log carrier on log rate over its singer-hours) with its position at 25 degC:
      the streak slope, the population's coupling, written to streak_lines<suffix>.csv for Figure 8.

Points are drawn in random order so no temperature sits on top. House style (paper_page.PaperPage).

Usage:
  python analysis/callers/year_plane_figure.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v9 \
      --out analysis/callers/runs/figures_paper/F07_year_plane
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

MIN_N = 40
FLO, FHI = 1500.0, 20000.0
RLO, RHI = 10.0, 100.0
TLO, THI = 12.0, 32.0
# page geometry (inches from the page's top-left)
X0, X1 = 1.35, 7.25
ROWS = {"A": (1.15, 2.30), "B": (3.85, 2.30), "C": (6.55, 2.30)}


LINEAR = False   # --linear: the same panels on plain F against R axes (MBE 2026-10-14, exploratory: "replot fig7a in the F-R plane -- that is without log")


def style(ax):
    ax.set_xlim(RLO, RHI); ax.set_ylim(FLO, FHI)
    if LINEAR:
        ax.xaxis.set_major_locator(FixedLocator([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        ax.yaxis.set_major_locator(FixedLocator([2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]))
    else:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([t for t in (2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100) if t >= RLO]))
        ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000]))
    ax.xaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_xlabel("syllable rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(True, which="major", alpha=0.25)


def to_u(R, F):
    """Plot coordinates for label placement: log units, or on linear axes the unit square of the axes."""
    return (np.log(R), np.log(F)) if not LINEAR else ((R - RLO) / (RHI - RLO), (F - FLO) / (FHI - FLO))


def from_u(u, v):
    return (np.exp(u), np.exp(v)) if not LINEAR else (RLO + u * (RHI - RLO), FLO + v * (FHI - FLO))


def repel(xy, min_d, n_iter=200, step=0.3):
    """Push label positions (in log units) apart until no pair is closer than min_d."""
    xy = xy.copy()
    for _ in range(n_iter):
        moved = False
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                d = xy[j] - xy[i]; dist = np.hypot(*d)
                if dist < min_d:
                    push = (min_d - dist) / 2 * step
                    direction = d / dist if dist > 1e-9 else np.array([0.0, 1.0])
                    xy[i] -= push * direction; xy[j] += push * direction; moved = True
        if not moved:
            break
    return xy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trillers-only", action="store_true", help="drop the chirpers (tracks with a chirp rate in >= 50 %% of their windows) and the chirp-form streaks: Figure 7 proper, the chirpers go to Figure 7b")
    ap.add_argument("--number", default="7")
    ap.add_argument("--linear", action="store_true", help="plain (not log) axes; label spacing then in axes units")
    ap.add_argument("--overlay", default=None, help="CSV of published points (columns pulse_rate, carrier_kHz, species[, n_overlapping]) drawn on (A) in black with a fitted line per species (exploratory; MBE 2026-10-14: the Anaxipha figure)")
    ap.add_argument("--overlay-min-n", type=int, default=2, help="overlay species with fewer recordings than this are drawn as points only, no fitted line or label (the iNaturalist overlay has 100 taxa)")
    ap.add_argument("--rlo", type=float, default=None, help="lower rate limit of the axes (default 10 Hz)")
    ap.add_argument("--a-only", action="store_true", help="draw panel A alone, taller (the overlay figure); streak_lines is not rewritten")
    ap.add_argument("--title", default=None, help="legend title override")
    ap.add_argument("--legend", default="", help="legend text after the title")
    args = ap.parse_args()
    global LINEAR, RLO; LINEAR = args.linear
    if args.rlo is not None:
        RLO = args.rlo
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    P = pd.read_csv(os.path.join(args.root, "streak_hourly_points_all.csv")).drop(columns=["streak"]).merge(A[["an", "streak"]], on="an", how="left")
    P["streak"] = P.streak.fillna(-1).astype(int)
    P = P[P.R.between(RLO, RHI) & P.F.between(FLO, FHI)]
    if args.trillers_only:
        rc = P.groupby("an").rc_frac.mean() if "rc_frac" in P.columns else pd.Series(dtype=float)
        chirp_an = set(rc.index[rc >= 0.5]); n0 = len(P)
        P = P[~P.an.isin(chirp_an)]
        if "form" in S.columns:
            S = S[S.form != "chirp"]
        print(f"trillers only: dropped {n0 - len(P):,} chirper hours ({len(chirp_an):,} tracks) and the chirp-form streaks")
    big = [s for s in S.index if S.loc[s, "n"] >= args.min_n]
    ids = paper_ids.load(args.root, args.suffix)
    rng = np.random.default_rng(0)
    P = P.iloc[rng.permutation(len(P))]
    n_big_pts = int(P.streak.isin(big).sum())
    print(f"{len(P):,} singer-hours from {P.an.nunique():,} singer-nights, {P.recorder.nunique()} recorders, {P.night.nunique()} nights; "
          f"{len(big)} streaks with >= {args.min_n} singer-nights hold {n_big_pts:,} ({n_big_pts / len(P):.0%}); T {P['T'].min():.1f}-{P['T'].max():.1f} °C")

    page = PaperPage(number=args.number, title=args.title or ("A year of trillers on the carrier against syllable-rate plane" if args.trillers_only else "A year of singers on the carrier-rate plane"), legend=args.legend)
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]

    # (A) temperature
    top, h = (1.15, 4.6) if args.a_only else ROWS["A"]
    axA = fig.add_axes(rect(X0, X1, top, h))
    sc = axA.scatter(P.R, P.F, s=0.6, c=P["T"], cmap="turbo", norm=Normalize(TLO, THI), alpha=0.4, edgecolors="none", rasterized=True)
    style(axA)
    cax = axA.inset_axes([0.04, 0.92, 0.30, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("air temperature at the hour (°C)", fontsize=BODY_PT)
    cb.ax.tick_params(labelsize=BODY_PT); cb.solids.set_alpha(1)
    if args.overlay:
        O = pd.read_csv(args.overlay); w = O["n_overlapping"] if "n_overlapping" in O.columns else pd.Series(1.0, index=O.index)
        if "T" in O.columns:                                   # published recordings with their own temperature: the same color scale as the hours, black rims
            axA.scatter(O.pulse_rate, O.carrier_kHz * 1000, s=9, c=O["T"], cmap="turbo", norm=Normalize(TLO, THI), edgecolors="k", lw=0.35, zorder=5)
        else:
            axA.scatter(O.pulse_rate, O.carrier_kHz * 1000, s=3 + 3 * w, color="k", edgecolors="white", lw=0.3, zorder=5)
        ends = []
        for sp, g in O.groupby("species"):
            if len(g) < args.overlay_min_n:                     # too few recordings for a line: points only (the iNaturalist overlay)
                continue
            x, y = np.log(g.pulse_rate), np.log(g.carrier_kHz * 1000); b, a = np.polyfit(x, y, 1, w=np.sqrt(w[g.index]))
            xx = np.array([x.min(), x.max()]); axA.plot(np.exp(xx), np.exp(a + b * xx), color="k", lw=1.0, zorder=6); ends.append((sp, b, xx[-1], a + b * xx[-1]))
        # labels at the warm end, pushed apart in log units where lines end close together (scia / rosamacula / litarena)
        ends.sort(key=lambda e: e[3]); lab_y = [e[3] for e in ends]
        for i in range(1, len(lab_y)):
            if abs(ends[i][2] - ends[i - 1][2]) < 0.25 and lab_y[i] - lab_y[i - 1] < 0.06:
                lab_y[i] = lab_y[i - 1] + 0.06
        for (sp, b, xe, ye), ly in zip(ends, lab_y):
            nm = f"{sp.split()[0][0]}. {' '.join(sp.split()[1:])}" if " " in sp else f"A. {sp}"   # binomial given (iNaturalist) or an Anaxipha epithet (Walker & Funk)
            axA.annotate(nm if args.a_only else f"{sp} {b:.2f}", (np.exp(xe), np.exp(ye)), xytext=(np.exp(xe) * 1.06, np.exp(ly)), textcoords="data", fontsize=5.5, style="italic", va="center", zorder=7,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), arrowprops=dict(arrowstyle="-", color="0.4", lw=0.4) if abs(ly - ye) > 0.01 else None)
    if args.a_only:
        # type numbers at the warm end of each type's line, fitted here (the full figure writes them to streak_lines<suffix>.csv)
        for sid in big:
            g = P[P.streak == sid]; x, y = np.log(g.R), np.log(g.F); b, a = np.polyfit(x, y, 1); xh = np.percentile(x, 99)
            axA.annotate(str(ids.get(sid, sid)), (np.exp(xh), np.exp(a + b * xh)), xytext=(2, 1), textcoords="offset points", fontsize=4.5, color="0.3", zorder=6)
        page.finish(fig, args.out); print(f"wrote {args.out}.pdf / .png"); return
    page.label(axA, "A", "Singer-hours by temperature" + (" with the published points in black" if args.overlay else ""), y=1 + 0.05 / h)

    # (B) streak labels
    top, h = ROWS["B"]
    axB = fig.add_axes(rect(X0, X1, top, h))
    other = P[~P.streak.isin(big)]
    axB.scatter(other.R, other.F, s=0.5, color="0.82", alpha=0.5, edgecolors="none", rasterized=True)
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    order = {sid: i for i, sid in enumerate(np.random.default_rng(1).permutation(big))}
    cents = []
    for sid in big:
        g = P[P.streak == sid]
        axB.scatter(g.R, g.F, s=0.6, color=cols[order[sid] % 20], alpha=0.5, edgecolors="none", rasterized=True)
        cents.append([*to_u(np.exp(np.log(g.R).mean()), np.exp(np.log(g.F).mean()))])
    cents = np.array(cents)
    lab = repel(cents, min_d=0.085 if not LINEAR else 0.035)
    for sid, c, l in zip(big, cents, lab):
        axB.annotate(str(ids.get(sid, sid)), xy=from_u(*c), xytext=from_u(*l), fontsize=5, ha="center", va="center", color="k",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cols[order[sid] % 20], lw=0.6, alpha=0.9),
                     arrowprops=dict(arrowstyle="-", color="0.3", lw=0.4) if np.hypot(*(l - c)) > (0.02 if not LINEAR else 0.01) else None)
    style(axB)
    page.label(axB, "B", "Singer-hours by type", y=1 + 0.05 / h)
    # (C) each streak as a fitted line (log carrier on log rate over its singer-hours, 1st to 99th percentile of rate) with its
    # position at 25 C (rate from the type's rate-on-temperature fit): the streak slope, written to <root>/streak_lines<suffix>.csv
    top, h = ROWS["C"]
    axC = fig.add_axes(rect(X0, X1, top, h))
    axC.scatter(P.R[::4], P.F[::4], s=0.4, color="0.88", lw=0, rasterized=True)
    lines = []
    for sid in big:
        g = P[P.streak == sid]; x, y = np.log(g.R), np.log(g.F)
        b, a = np.polyfit(x, y, 1); bT, aT = np.polyfit(g["T"], x, 1); x25 = aT + 25 * bT
        xx = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 20); c = cols[order[sid] % 20]
        axC.plot(np.exp(xx), np.exp(a + b * xx), color=c, lw=1.2, zorder=3)
        axC.scatter([np.exp(x25)], [np.exp(a + b * x25)], s=16, color=c, edgecolors="k", lw=0.4, zorder=4)
        axC.annotate(str(ids.get(sid, sid)), (np.exp(x25), np.exp(a + b * x25)), fontsize=5, xytext=(3, 2), textcoords="offset points", zorder=5)
        lines.append(dict(streak=sid, paper_id=ids.get(sid, sid), slope=b, intercept=a, R25=np.exp(x25), F25=np.exp(a + b * x25), lnR_lo=xx[0], lnR_hi=xx[-1], n_hours=len(g), r=np.corrcoef(x, y)[0, 1]))
    style(axC)
    page.label(axC, "C", "Each streak as a fitted line, with its position at 25 °C", y=1 + 0.05 / h)
    L = pd.DataFrame(lines); L.to_csv(os.path.join(args.root, f"streak_lines{args.suffix}.csv"), index=False, float_format="%.5f")
    print(f"streak slopes: median {L.slope.median():.2f}, IQR {L.slope.quantile(.25):.2f}-{L.slope.quantile(.75):.2f}; wrote streak_lines{args.suffix}.csv")
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
