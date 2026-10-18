"""Figure 5: coherent organization of individual songs across one night (House 8662, 20 Sep 2025), before linking.

Layout and wording follow MBE's Figure 5 of 2026-10-10 (MBE_Figures/MBE_F05_OneNightStreaks.pdf):
  (A) Singers from the 5 minute window of Figs 3D and 4 on the carrier-rate plane: the detector's animals at their
      lowest carrier component, the seven shown in the window figure numbered in its colors, their higher carrier
      components small, other singers gray, chirp-rate-only singers as squares at their chirp rate
  (B) Singers from all 5 minute windows in the night (the postfilter's kept rows), colored by HRRR air temperature at
      the window; the clusters of (C) to (G) lettered
  (C) to (G) one row per cluster: the strongest window-animal in each third of the cluster's time span, an excerpt of
      its band (dB above the window's quiet level, pulses resolved) beside its fold (two periods), headed by time,
      temperature, carrier and rate (night_streak_examples_figure.draw_row)

Usage:
    python analysis/callers/night_streaks_figure.py --unit-dir analysis/callers/runs/v3/y2025/BARLT_00008662_House_20250920_night \
        --window "2025-09-20 23:05" --weather analysis/callers/runs/v2/y2025/weather_hourly_hrrr.csv \
        --out analysis/callers/runs/figures_paper/F05_one_night_streaks
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
from window_figure import choose_animals  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402
import night_streak_examples_figure as NSE  # noqa: E402

FLO, FHI = 1500.0, 20000.0
RLO, RHI = 2.0, 100.0
# page geometry (inches from the page's top-left), after MBE's layout
PLANE_TOP, PLANE_H = 1.25, 2.20
A_X, B_X = (1.15, 4.20), (4.75, 7.15)
ROW0, ROW_PITCH, ROW_H, HEAD = 3.95, 1.07, 0.52, 0.28
CELL_X0, CELL_PITCH, SPEC_W, FOLD_W, FOLD_GAP = 1.15, 2.05, 0.85, 0.42, 0.2


def plane_axes(ax):
    ax.set_xscale("log"); ax.set_xlim(RLO, RHI)
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 15, 20, 30, 50, 70, 100])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_yscale("log"); ax.set_ylim(FLO, FHI)
    ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000])); ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_xlabel("syllable rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(True, which="major", alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-dir", required=True); ap.add_argument("--window", required=True)
    ap.add_argument("--weather", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ex = NSE.Examples(args.unit_dir, args.weather)
    rec, site = ex.rec, ex.site

    # the night's kept window-animals (all of them, for (B))
    Wd = pd.read_csv(os.path.join(args.unit_dir, "night_night_global_window_animals.csv"))
    Wd["t"] = pd.to_datetime(Wd.t, utc=True).dt.tz_convert(W2.TZ)
    Wd = Wd[Wd.keep.astype(bool)] if "keep" in Wd else Wd
    pts = Wd[Wd.R.notna() & (Wd.R > 0)].copy(); pts["T"] = [ex.temp_at(t) for t in pts.t]
    chirp_only = Wd[Wd.R.isna() & Wd.chirp_rate.notna()].copy(); chirp_only["T"] = [ex.temp_at(t) for t in chirp_only.t]

    # the window of Figs 3D and 4
    t = pd.Timestamp(args.window, tz=W2.TZ)
    X = ex.window(t)
    parts, B, A = X["parts"].copy(), X["B"], X["A"]
    parts["R"] = parts.blob.map(B.R_fine)
    chosen, num, col, *_ = choose_animals(X, 8)
    Aidx = A.set_index("animal")

    page = PaperPage(number=5, title="Coherent organization of individual songs across one night")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]

    # ---- (A) the window's singers ----
    ax = fig.add_axes(rect(*A_X, PLANE_TOP, PLANE_H))
    done = set()
    for a, g in parts[parts.am_harm_of < 0].groupby("animal"):
        g = g.sort_values("f_c")
        chirp = a in Aidx.index and pd.isna(Aidx.loc[a, "R"])
        mk = "s" if chirp else "o"
        c = col.get(a, "0.25")
        low = g.iloc[0]
        ax.scatter([low.R], [low.f_c], s=28 if a in col else 10, color=c, marker=mk, edgecolors="k" if a in col else "none", lw=0.5, zorder=4,
                   label=None if (a in col or ("other" in done)) else "other singers (lowest component)")
        if a not in col:
            done.add("other")
        if len(g) > 1:
            ax.scatter(g.R.iloc[1:], g.f_c.iloc[1:], s=6, color=c, marker=mk, lw=0, alpha=0.7, zorder=3,
                       label=None if "comp" in done else "numbered singers' higher carrier components (2F, 3F)")
            done.add("comp")
        if a in col:
            ax.text(low.R * 1.07, low.f_c, str(num[a]), color=c, fontsize=BODY_PT, fontweight="bold", va="center", zorder=5)
    plane_axes(ax); ax.legend(fontsize=5, loc="lower left", frameon=False)
    page.label(ax, "A", "Singers in one window", y=1 + 0.05 / PLANE_H)

    # ---- (B) every kept window-animal of the night, colored by temperature ----
    ax = fig.add_axes(rect(*B_X, PLANE_TOP, PLANE_H))
    allT = pd.concat([pts["T"], chirp_only["T"]])
    tnorm = Normalize(np.floor(allT.min()), np.ceil(allT.max())); tcmap = plt.get_cmap("turbo")
    sc = ax.scatter(pts.R, pts.F_c, c=pts["T"], cmap=tcmap, norm=tnorm, s=3, lw=0, zorder=3)
    ax.scatter(chirp_only.chirp_rate, chirp_only.F_c, c=chirp_only["T"], cmap=tcmap, norm=tnorm, s=5, marker="s", lw=0, zorder=3)
    plane_axes(ax)
    cax = fig.add_axes(rect(B_X[1] + 0.06, B_X[1] + 0.13, PLANE_TOP, PLANE_H))
    cb = fig.colorbar(sc, cax=cax); cb.set_label("air temperature at the window (°C, HRRR)", fontsize=BODY_PT); cb.ax.tick_params(labelsize=BODY_PT)
    # the clusters of the rows below, lettered at their low-rate end
    for i, (s, picks) in enumerate(ex.rows):
        L = chr(ord("C") + i)
        if "Rc_lo" in s:
            ax.text(s["Rc_lo"] * 0.85, np.sqrt(s["F_lo"] * s["F_hi"]), L, fontsize=8, fontweight="bold", ha="right", va="center", zorder=6)
        else:
            M = pts[NSE.in_streak(pts, s)]
            if len(M):
                m = M.sort_values("R").iloc[max(0, int(0.03 * len(M)))]
                ax.text(m.R * 0.88, m.F_c, L, fontsize=8, fontweight="bold", ha="right", va="center", zorder=6)
    page.label(ax, "B", "Singers in all windows of the night", y=1 + 0.05 / PLANE_H)

    # ---- (C) to (G) the clusters' examples ----
    for i, (s, picks) in enumerate(ex.rows):
        top = ROW0 + i * ROW_PITCH
        L = chr(ord("C") + i)
        fig.text((CELL_X0 - fx0) / Wc, 1 - (top - 1.0) / Hc, f"({L}) {s['name']}", fontsize=8, fontweight="bold", va="bottom", ha="left")
        cells = [(rect(CELL_X0 + j * CELL_PITCH, CELL_X0 + j * CELL_PITCH + SPEC_W, top + HEAD, ROW_H),
                  rect(CELL_X0 + j * CELL_PITCH + SPEC_W + FOLD_GAP, CELL_X0 + j * CELL_PITCH + SPEC_W + FOLD_GAP + FOLD_W, top + HEAD, ROW_H)) for j in range(len(picks))]
        NSE.draw_row(fig, ex, s, picks, cells, last_row=(i == len(ex.rows) - 1))

    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png; night points {len(pts)} + {len(chirp_only)} chirp-only; window animals {len(A)}, {len(chosen)} numbered")


if __name__ == "__main__":
    main()
