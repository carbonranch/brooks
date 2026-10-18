"""Figure 8A for a whole type: one panel per singer, its windows' carrier against syllable rate through the night, both as
percent of the night median, colored by air temperature, with the singer's fitted slope (black) and the type's streak
slope (dashed red). A grid of 8 rows x 6 columns on a letter page, panels ordered by the singer's median rate so the
page runs along the streak (MBE 2026-10-15: "expand figure [8A] to a grid for a type ... the temperature changes of
individual singers").

Singers: the type's firm trill tracks with >= --min-win windows, distinct recorder-nights, the widest within-night
temperature range first. Data as coupling_figure.load_track (per-window table of the night, HRRR temperature
interpolated to the window).

Usage:
  python analysis/callers/singer_paths_grid.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --types 5 16 \
      --out-dir analysis/callers/runs/figures_paper/exploratory/singer_paths
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coupling_figure as CF  # noqa: E402
import paper_ids  # noqa: E402
from paper_page import PaperPage, BODY_PT, LABEL_PT  # noqa: E402

XLIM, YLIM = 15.0, 10.0
TLO, THI = 12.0, 32.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--types", nargs="+", required=True); ap.add_argument("--rows", type=int, default=8); ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--min-win", type=int, default=30); ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv")); S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    Tk = pd.read_csv(os.path.join(root, f"year_coupling_tracks{args.suffix}.csv")).rename(columns={"n": "nwin"})
    SL = pd.read_csv(os.path.join(root, f"streak_lines{args.suffix}.csv")).set_index("streak")
    ids = paper_ids.load(root, args.suffix); inv = {str(v): k for k, v in ids.items()}
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["an", "T", "R"]).dropna(subset=["T"])
    dT = P.groupby("an").agg(dT=("T", lambda x: x.max() - x.min()), Rmed=("R", "median"))
    tnorm = Normalize(TLO, THI); tcmap = plt.get_cmap("turbo")
    for tname in args.types:
        sid = inv[str(tname)]; pid = ids.get(sid, sid)
        g = A[A.streak == sid]
        if "form" in g.columns and "form" in S.columns:
            g = g[g.form == S.loc[sid, "form"]]
        J = g[["an", "recorder", "night", "depth"]].merge(Tk, on="an").join(dT, on="an")
        J = J[J.nwin >= args.min_win].sort_values("dT", ascending=False)
        picks, used = [], set()
        for r in J.itertuples():
            if (r.recorder, r.night) in used:
                continue
            used.add((r.recorder, r.night)); picks.append(r)
            if len(picks) >= args.rows * args.cols:
                break
        picks = sorted(picks, key=lambda r: r.Rmed)
        bs = float(SL.loc[sid, "slope"]) if sid in SL.index else np.nan
        print(f"type {pid}: {len(picks)} singers of {len(J)} with >= {args.min_win} windows; temperature ranges {min(r.dT for r in picks):.0f}-{max(r.dT for r in picks):.0f} °C; streak slope {bs:.2f}")
        legend = (f"One panel per singer, {len(picks)} singers of the type with at least {args.min_win} five-minute windows in one night and the widest temperature range within it, "
                  "ordered by the singer's median rate from slow to fast, so the page runs along the streak. Each point is one window, its carrier against its syllable rate as percent "
                  "of the singer's night median, colored by the HRRR air temperature at the window; the black line is the singer's fitted slope of log carrier on log rate (its number in the "
                  "panel head), the dashed red line the type's streak slope from Figure 7C, the dotted gray line carrier in proportion to rate. Panel head: night, recorder, site, the singer's median "
                  "rate and carrier, its slope, and its temperature range.")
        page = PaperPage(number=f"S-{pid}-paths", title=f"Type {pid}: {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz at 25 °C, {len(picks)} singers through their nights.", legend=legend)
        Wc, Hc = 7.2, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
        fig = plt.figure(figsize=(Wc, Hc))

        def rect(x0, x1, top, h):
            return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
        X0, X1, TOP = fx0 + 0.45, fx0 + Wc - 0.42, 1.45
        GAPX, HEAD = 0.10, 0.20
        pw = (X1 - X0 - (args.cols - 1) * GAPX) / args.cols; pitch_h = (Hc - (TOP - 1.0) - 0.12) / args.rows; ph = pitch_h - HEAD - 0.06   # rows fill the page's content height (the legend's length sets it)     # three-line heads; eight rows end above the legend
        fig.text(0.02 / Wc, 1 - 0.02 / Hc, f"Type {pid} (streak {sid}), {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz at 25 °C: {len(picks)} singers through their nights", fontsize=LABEL_PT, fontweight="bold", va="top", ha="left")
        fig.text(0.02 / Wc, 1 - 0.20 / Hc, "each panel one singer: its windows' carrier against rate, percent of the night median, colored by air temperature\n"
                 "black = the singer's slope, dashed red = the type's streak slope; panels ordered by the singer's median rate, slow to fast", fontsize=BODY_PT, va="top", ha="left", linespacing=1.3)
        sc = None; slopes = []
        for i, r in enumerate(picks):
            rr, cc = divmod(i, args.cols); x0 = X0 + cc * (pw + GAPX); top = TOP + rr * pitch_h + HEAD
            ax = fig.add_axes(rect(x0, x0 + pw, top, ph))
            try:
                gt, unit = CF.load_track(r.an, args.roots)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "no windows", ha="center", va="center", fontsize=BODY_PT - 1, transform=ax.transAxes); continue
            x = 100 * (gt.R / gt.R.median() - 1); y = 100 * (gt.F_fix / gt.F_fix.median() - 1)
            sc = ax.scatter(x, y, c=gt["T"], cmap=tcmap, norm=tnorm, s=4, lw=0.15, edgecolors="k", zorder=2, rasterized=True)
            lx, ly = np.log(gt.R), np.log(gt.F_fix); b, a = np.polyfit(lx, ly, 1); slopes.append(b)
            xx = np.linspace(max(x.min(), -XLIM), min(x.max(), XLIM), 50); rrr = gt.R.median() * (1 + xx / 100)
            ax.plot(xx, 100 * (np.exp(a + b * np.log(rrr)) / gt.F_fix.median() - 1), color="k", lw=0.8, zorder=3)
            if np.isfinite(bs):
                xs = np.linspace(-XLIM, XLIM, 20); ax.plot(xs, 100 * ((1 + xs / 100) ** bs - 1), color="tab:red", lw=0.7, ls="--", zorder=2)
            ax.plot([-XLIM, XLIM], [-XLIM, XLIM], color="0.5", lw=0.4, ls=":", zorder=1)
            ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
            ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-YLIM, YLIM); ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, -5, 0, 5, 10])
            ax.tick_params(labelsize=BODY_PT - 1.5, length=1.5, pad=1.5, width=0.5)
            if cc > 0:
                ax.set_yticklabels([])
            if rr < args.rows - 1 and i + args.cols < len(picks):
                ax.set_xticklabels([])
            if cc == 0 and rr == args.rows // 2:
                ax.set_ylabel("carrier (% of night median)", fontsize=BODY_PT - 1)
            if rr == args.rows - 1 and cc == args.cols // 2:
                ax.set_xlabel("rate (% of night median)", fontsize=BODY_PT - 1)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            site = unit.split("_")[2][:6]; night = pd.Timestamp(unit.split("_")[3]).strftime("%y-%m-%d")
            ax.set_title(f"{night} {r.recorder[-4:]} {site}\n{gt.F_fix.median() / 1000:.2f} kHz @ {gt.R.median():.0f} Hz\nslope {b:.2f}   {gt['T'].min():.0f}–{gt['T'].max():.0f} °C", fontsize=4.5, loc="left", pad=1.5, linespacing=1.1)
        if sc is not None:
            cax = fig.add_axes(rect(X1 + 0.10, X1 + 0.16, TOP + HEAD, 2 * pitch_h - HEAD - 0.07))
            cb = fig.colorbar(sc, cax=cax); cb.set_label("°C", fontsize=BODY_PT); cb.ax.tick_params(labelsize=BODY_PT - 1)
        base = os.path.join(args.out_dir, f"type_{pid:02d}_paths" if isinstance(pid, int) else f"type_{pid}_paths")
        page.finish(fig, base); plt.close(fig)
        for ext in ("_content.pdf", "_content.png"):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        print(f"  slopes: median {np.median(slopes):.2f}, IQR {np.percentile(slopes, 25):.2f}-{np.percentile(slopes, 75):.2f}; wrote {base}.pdf / .png")


if __name__ == "__main__":
    main()
