"""Each of the 25 trill types in its own panel: the type's singer-hours colored by air temperature, over the fitted
lines of all 25 types in black (the type's own line drawn like the rest). 7 rows x 4 columns on a letter page, shared log axes
(MBE 2026-10-15: "each of the 25 triller types in its own panel just showing its songs colored by temperature
overlaid on a plot of the lines for all 25 (in black) ... a 7 row, 4 column grid").

Data as Figure 7 (year_plane_figure.py): streak_hourly_points_all.csv with the streak of each singer-night from
animal_nights_streaks<suffix>.csv, chirpers dropped; lines from streak_lines<suffix>.csv (written by Figure 7C).

Usage:
  python analysis/callers/type_panels_figure.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --out analysis/callers/runs/figures_paper/F07s_type_panels
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

FLO, FHI = 1500.0, 20000.0
RLO, RHI = 10.0, 100.0
TLO, THI = 12.0, 32.0
NCOL, NROW = 4, 7
MIN_N = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--out", required=True)
    ap.add_argument("--number", default="7 supplement"); ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    P = pd.read_csv(os.path.join(args.root, "streak_hourly_points_all.csv")).drop(columns=["streak"]).merge(A[["an", "streak"]], on="an", how="left")
    P["streak"] = P.streak.fillna(-1).astype(int)
    P = P[P.R.between(RLO, RHI) & P.F.between(FLO, FHI)]
    rc = P.groupby("an").rc_frac.mean() if "rc_frac" in P.columns else pd.Series(dtype=float)
    P = P[~P.an.isin(set(rc.index[rc >= 0.5]))]
    if "form" in S.columns:
        S = S[S.form != "chirp"]
    L = pd.read_csv(os.path.join(args.root, f"streak_lines{args.suffix}.csv")).set_index("streak")
    ids = paper_ids.load(args.root, args.suffix)
    big = sorted([s for s in S.index if S.loc[s, "n"] >= args.min_n and s in L.index], key=lambda s: int(ids.get(s, 10 ** 6)))
    global NROW; NROW = max(NROW, int(np.ceil(len(big) / NCOL)))        # the catalog grew past 28 types on 2026-10-16
    print(f"{len(big)} trill types, {len(P):,} singer-hours")

    page = PaperPage(number=args.number, title="Each trill type on the plane of all the lines")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]

    X0, X1, TOP = 1.35, 7.3, 1.15
    GAPX, GAPY, HEAD = 0.14, 0.15, 0.11
    PW = (X1 - X0 - (NCOL - 1) * GAPX) / NCOL; PITCH = (Hc - (TOP - 1.0) - 0.15) / NROW; PH = PITCH - GAPY - HEAD   # rows fill the page's content height
    norm = Normalize(TLO, THI); sc = None
    for i, s in enumerate(big):
        rr, cc = divmod(i, NCOL); x0 = X0 + cc * (PW + GAPX); top = TOP + rr * PITCH + HEAD
        ax = fig.add_axes(rect(x0, x0 + PW, top, PH))
        for s2 in big:                                   # all 25 lines in black over their 1st-99th percentile of rate
            l = L.loc[s2]; xx = np.array([l.lnR_lo, l.lnR_hi])
            ax.plot(np.exp(xx), np.exp(l.intercept + l.slope * xx), color="k", lw=0.35, alpha=0.6, zorder=2)   # the focal line drawn like the rest (MBE: heavier, "it dominates")
        g = P[P.streak == s]
        sc = ax.scatter(g.R, g.F, s=0.5, c=g["T"], cmap="turbo", norm=norm, alpha=0.6, edgecolors="none", rasterized=True, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(RLO, RHI); ax.set_ylim(FLO, FHI)
        ax.xaxis.set_major_locator(FixedLocator([10, 20, 30, 50, 100])); ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_locator(FixedLocator([2000, 3000, 5000, 10000, 20000])); ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        ax.tick_params(labelsize=BODY_PT - 1, length=2, pad=1.5); ax.grid(True, which="major", alpha=0.2)
        if cc != 0:
            ax.set_yticklabels([])
        if rr != NROW - 1 and i + NCOL < len(big):
            ax.set_xticklabels([])
        if cc == 0:
            ax.set_ylabel("carrier (kHz)", fontsize=BODY_PT - 1, labelpad=1)
        if rr == NROW - 1 or i + NCOL >= len(big):
            ax.set_xlabel("syllable rate (Hz)", fontsize=BODY_PT - 1, labelpad=1)
        l = L.loc[s]
        ax.set_title(f"{ids.get(s, s)}   {l.F25 / 1000:.1f} kHz @ {l.R25:.0f} Hz   {len(g):,} h", fontsize=BODY_PT - 0.5, loc="left", pad=2)
    # colorbar in the empty slot(s) of the last row
    n_empty = NCOL * NROW - len(big)
    if n_empty and sc is not None:
        rr, cc = divmod(len(big), NCOL); x0 = X0 + cc * (PW + GAPX) + 0.1; x1 = X0 + (cc + n_empty) * (PW + GAPX) - GAPX - 0.1
        cax = fig.add_axes(rect(x0, x1, TOP + rr * PITCH + HEAD + 0.3, 0.08))
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("air temperature at the hour (°C)", fontsize=BODY_PT)
        cb.set_ticks([15, 20, 25, 30]); cb.ax.tick_params(labelsize=BODY_PT, length=2); cb.solids.set_alpha(1)
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
