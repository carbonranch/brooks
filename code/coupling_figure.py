"""Carrier-rate coupling within a singer, over the year (house style, paper_page.PaperPage).

  (A) six example tracks, one per chosen type (dashed red: the type's streak slope from Figure 7C): each window's reference carrier against its syllable rate, both as
      percent of the track's night median, colored by the HRRR air temperature at the window; the fitted slope (log
      carrier on log rate) is the track's coupling. The example is the track whose slope is CLOSEST TO ITS TYPE'S
      MEDIAN among tracks with >= EX_MIN_WIN windows and a rate range of EX_REL_RANGE of the type's rate at 25 degC:
      a typical track, not the cleanest. All six panels share axes.
  (B) every trill type with >= MIN_N singer-periods: the median coupling over its tracks with the interquartile
      range, against the type's carrier at 25 degC, labeled with the paper number; colored by the subfamily of the
      SINA candidate nearest the type's position (a candidate, not an identification). Dotted line at 1: carrier
      proportional to rate.

Tracks enter (B) with >= MIN_WIN windows and a rate range between REL_LO and REL_HI of the type's rate at 25 degC:
the lower bound removes tracks too flat to carry a slope, the upper bound tracks whose rate jumped.

Usage:
  python analysis/callers/coupling_figure.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 \
      --examples 4301 3601 5501 5201 4601 4801 --number 12 --out analysis/callers/runs/figures_paper/F12_coupling
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
import window_blobs2 as W2  # noqa: E402
from year_coupling import COL  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

MIN_N = 40
MIN_WIN = 12
REL_LO, REL_HI = 0.02, 0.30
EX_MIN_WIN = 40
EX_REL_RANGE = (0.10, 0.30)
TLO, THI = 12.0, 32.0
XLIM, YLIM = 16.0, 10.0          # percent about the track median
# page geometry (inches from the page's top-left)
A_TOP, A_H, A_X0, A_X1, A_GAP = 1.45, 1.55, 1.35, 7.15, 0.10
B_TOP, B_H = 3.70, 3.6


def load_track(an, roots):
    unit, tr = an.rsplit(":", 1)
    for r in roots:
        p = os.path.join(r, unit, "night_night_global_window_animals.csv")
        if os.path.exists(p):
            W = pd.read_csv(p, usecols=["t", "track", "R", "F_fix"])
            g = W[(W.track == int(tr)) & W.F_fix.notna() & (W.R > 0)].copy()
            g["t"] = pd.to_datetime(g.t, utc=True).dt.tz_convert(W2.TZ)
            Wx = pd.read_csv(os.path.join(r, "weather_hourly_hrrr.csv")); Wx["t"] = pd.to_datetime(Wx.t, utc=True).dt.tz_convert(W2.TZ)
            Wx = Wx.sort_values("t")
            g["T"] = np.interp(g.t.astype("int64"), Wx.t.astype("int64"), Wx.T_C)
            return g.sort_values("t"), unit
    raise FileNotFoundError(an)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--examples", type=int, nargs="+", default=[4301, 3601, 5501, 5201, 4601, 4801])
    ap.add_argument("--min-n", type=int, default=MIN_N); ap.add_argument("--number", default="8"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    Tk = pd.read_csv(os.path.join(root, f"year_coupling_tracks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"year_coupling{args.suffix}.csv")).set_index("streak")     # carries candidate and group
    C = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    if "form" in C.columns:
        C = C[C.form != "chirp"]
    ids = paper_ids.load(root, args.suffix)
    slp = os.path.join(root, f"streak_lines{args.suffix}.csv")                # Figure 7C's streak slopes
    SL = pd.read_csv(slp).set_index("streak") if os.path.exists(slp) else pd.DataFrame(columns=["slope"])
    J = A[["an", "streak", "T", "site", "night"]].merge(Tk, on="an")
    J = J.merge(C[["R25", "F25", "n"]], left_on="streak", right_index=True)
    J["rel"] = J.R_range / J.R25
    J["nwin"] = J["n_x"] if "n_x" in J.columns else J["n"]
    good = J[(J.nwin >= MIN_WIN) & J.rel.between(REL_LO, REL_HI) & J.slope.notna()]
    g = good.groupby("streak").slope.agg(n_tracks="size", within="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
    B = C[C.n >= args.min_n].join(g, how="inner")
    B["candidate"] = S.candidate.reindex(B.index); B["group"] = S.group.reindex(B.index).fillna("no candidate")
    print(f"{len(J):,} tracks in the cache; {len(good):,} with >= {MIN_WIN} windows and a rate range {REL_LO:.0%}-{REL_HI:.0%} of the type's R25")
    print(B[["F25", "n", "within", "q25", "q75", "n_tracks", "group"]].assign(paper=[ids.get(s, s) for s in B.index]).sort_values("F25").round(2).to_string())
    print("\nby group:"); print(B.groupby("group").within.agg(["median", "min", "max", "size"]).round(2).to_string())

    ex = []
    for s in args.examples:
        cand = good[(good.streak == s) & (good.nwin >= EX_MIN_WIN) & good.rel.between(*EX_REL_RANGE)]
        if cand.empty or s not in B.index:
            print(f"no example track for {s}"); continue
        med = B.loc[s, "within"]
        row = cand.assign(d=(cand.slope - med).abs()).sort_values(["d", "nwin"], ascending=[True, False]).iloc[0]
        ex.append((s, row))
        print(f"example {s} (type {ids.get(s, s)}): {row.an}, {int(row.nwin)} windows, rate range {row.R_range:.1f} Hz ({row.rel:.0%}), slope {row.slope:.2f} (type median {med:.2f}), r {row.r:.2f}")

    page = PaperPage(number=args.number, title="Carrier follows rate within a singer")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    ncol = len(ex); w = (A_X1 - A_X0 - (ncol - 1) * A_GAP) / ncol
    tnorm = Normalize(TLO, THI); tcmap = plt.get_cmap("turbo")
    axes = []
    for i, (s, row) in enumerate(ex):
        ax = fig.add_axes(rect(A_X0 + i * (w + A_GAP), A_X0 + i * (w + A_GAP) + w, A_TOP, A_H)); axes.append(ax)
        gt, unit = load_track(row.an, args.roots)
        x = 100 * (gt.R / gt.R.median() - 1); y = 100 * (gt.F_fix / gt.F_fix.median() - 1)
        sc = ax.scatter(x, y, c=gt["T"], cmap=tcmap, norm=tnorm, s=7, lw=0.2, edgecolors="k", zorder=2)
        lx, ly = np.log(gt.R), np.log(gt.F_fix); b, a = np.polyfit(lx, ly, 1)
        xx = np.linspace(x.min(), x.max(), 50); rr = gt.R.median() * (1 + xx / 100)
        ax.plot(xx, 100 * (np.exp(a + b * np.log(rr)) / gt.F_fix.median() - 1), color="k", lw=0.9, zorder=3)
        if s in SL.index:                                                  # the type's streak slope (Figure 7C), through the track's median: dashed
            bs = float(SL.loc[s, "slope"]); xs = np.linspace(-XLIM, XLIM, 20)
            ax.plot(xs, 100 * ((1 + xs / 100) ** bs - 1), color="tab:red", lw=0.8, ls="--", zorder=2)
        ax.plot([-XLIM, XLIM], [-XLIM, XLIM], color="0.5", lw=0.5, ls=":", zorder=1)
        ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
        ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-YLIM, YLIM)
        site = unit.split("_")[2]; night = pd.Timestamp(unit.split("_")[3]).strftime("%d %b")
        ax.set_title(f"type {ids.get(s, s)}, {gt.F_fix.median() / 1000:.1f} kHz @ {gt.R.median():.0f} Hz\nsinger {b:.2f}, streak {float(SL.loc[s, 'slope']) if s in SL.index else float('nan'):.2f}", fontsize=5.5, loc="left")
        if i == 0:
            ax.set_ylabel("carrier (% of night median)")
        else:
            ax.set_yticklabels([])
        if i == ncol // 2:
            ax.set_xlabel("rate (% of night median)")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    cax = fig.add_axes(rect(A_X1 + 0.05, A_X1 + 0.11, A_TOP, A_H))
    cb = fig.colorbar(sc, cax=cax); cb.set_label("°C", fontsize=BODY_PT); cb.ax.tick_params(labelsize=BODY_PT)
    page.label(axes[0], "A", "Six singers through a night", y=1 + 0.30 / A_H)

    # (B)
    ax = fig.add_axes(rect(A_X0, A_X1, B_TOP, B_H))
    # one color: the subfamily of the nearest SINA candidate is a name, and names come only in the last figure (MBE, 2026-10-13)
    ax.errorbar(B.F25, B.within, yerr=[B.within - B.q25, B.q75 - B.within], fmt="o", color="0.25", ms=3.5, elinewidth=0.7, capsize=1.5, alpha=0.9)
    for r in B.itertuples():
        ax.annotate(str(ids.get(r.Index, r.Index)), (r.F25, r.within), fontsize=5, xytext=(3, 2), textcoords="offset points", color="0.25")
    common = [s for s in B.index if s in SL.index]
    if common:
        ax.scatter(B.loc[common, "F25"], SL.loc[common, "slope"], s=22, marker="_", color="tab:red", lw=1.2, zorder=4, label="streak slope (Figure 7C)")
        ax.scatter([], [], s=12, color="0.25", label="within-singer coupling, median and IQR over tracks")
        ax.legend(fontsize=5, loc="upper right", frameon=False)
    ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlim(1800, 10000)
    ax.xaxis.set_major_locator(FixedLocator([2000, 3000, 4000, 5000, 6000, 8000, 10000])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_xlabel("type carrier at 25 °C (kHz)"); ax.set_ylabel("coupling within singers: % carrier per % rate (median, IQR over tracks)")
    ax.grid(alpha=0.3, which="major", lw=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    page.label(ax, "B", "Coupling per type", y=1 + 0.05 / B_H)
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
