"""Figure 8: a sample of the streak grids (the complete set is the supplement, streak_grid.py).

Three types side by side, each a reduced grid: rows = positions along the type's line (cold/slow at the top to
warm/fast at the bottom), columns = individual singers at that position from distinct recorder-nights, ordered
by temperature; every cell 0.5 s of the loudest 30 s of the singer's deepest window, on one shared log-frequency
axis per type zoomed to the band; color = dB above the row's quiet level over the window; outside the detected
band in grayscale (MBE, 2026-10-13; `--veil` for the old white veil). Above each grid, where its rows sit on the year plane (Fig 7). Cells are chosen as in the
supplement (streak_grid.choose_grid: deep tracks, pulse SNR >= 6 dB where enough, distinct recorder-nights).

Usage:
  python analysis/callers/grid_sample_figure.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v9 \
      --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --streaks 5301 6901 4051 \
      --out analysis/callers/runs/figures_paper/F08_grid_sample
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import streak_gallery as G  # noqa: E402
import specdisplay as SD  # noqa: E402
from streak_grid import choose_grid, HF_SPLIT, C_LO, C_HI, MASK_ALPHA  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

ROWS_N, COLS_N, DUR = 6, 3, 0.5
# page geometry (inches from the page's top-left)
COL_X0, COL_W, COL_GAP = 1.30, 1.90, 0.22
INSET_TOP, INSET_H = 1.55, 0.90
GRID_TOP, CELL_H, CELL_GAP, CELL_WGAP = 2.75, 0.76, 0.16, 0.06


def cells_for(sid, groups, unit_roots, form):
    out = []
    for i, grp in enumerate(groups):
        for j, (_, r) in enumerate(grp.iterrows()):
            w = G.window_of(r.an, unit_roots, form)
            if w is None or not np.isfinite(w["R"]):
                out.append((i, j, r, None)); continue
            try:
                x = G.analyse(w); out.append((i, j, r, dict(w, **x)))
            except Exception:  # noqa: BLE001
                out.append((i, j, r, None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--streaks", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--veil", action="store_true", help="the old look: rows outside the detected band under a white veil (default since 2026-10-13: grayscale)")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    C = pd.read_csv(os.path.join(root, f"type_catalog{args.suffix}.csv")).set_index("type")
    P_all = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["R", "F", "an"]).drop_duplicates().merge(A[["an", "streak"]], on="an", how="left")
    sub = P_all.sample(min(len(P_all), 30000), random_state=0)
    ids = paper_ids.load(root, args.suffix)
    pp = os.path.join(root, f"pulse_shapes{args.suffix}.csv")
    clear = None
    if os.path.exists(pp):
        Pq = pd.read_csv(pp); clear = set(Pq[(Pq.status == "ok") & (Pq.snr_db >= 6)].an)

    page = PaperPage(number=10, title="Coherence of three streaks across the year")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]

    for k, sid in enumerate(args.streaks):
        g = A[A.streak == sid]
        if "form" in A.columns and "form" in S.columns:
            g = g[g.form == S.loc[sid, "form"]]
        c = C_LO if S.loc[sid, "F25"] < HF_SPLIT else C_HI
        groups = choose_grid(g, c, ROWS_N, COLS_N, clear)
        tform = S.loc[sid, "form"] if "form" in S.columns else None
        cells = cells_for(sid, groups, args.unit_roots, tform)
        ok = [cc for cc in cells if cc[3] is not None]
        ylo = 0.8 * min(cc[3]["F_lo"] for cc in ok); yhi = 1.25 * max(cc[3]["F_hi"] for cc in ok)
        x0 = COL_X0 + k * (COL_W + COL_GAP); x1 = x0 + COL_W
        name = C.candidate.get(sid, np.nan) if "candidate" in C else np.nan
        st = C.candidate_status.get(sid, "") if "candidate_status" in C else ""
        who = "chirper" if tform == "chirp" else "triller"          # no species names before the names figure (MBE, 2026-10-13)
        label_id = ids.get(sid, sid)
        if tform == "chirp":                                         # a chirp type is named on the chirp plane: the chirp type most of its singer-nights fall in
            cdir = os.path.join(root, "chirp")
            if os.path.exists(os.path.join(cdir, "animal_nights_streaks_c2.csv")):
                Ac = pd.read_csv(os.path.join(cdir, "animal_nights_streaks_c2.csv"))[["an", "streak"]]; cid = pd.read_csv(os.path.join(cdir, "paper_ids_chirp.csv")).set_index("streak").paper_id
                m = g.merge(Ac, on="an", suffixes=("", "_c")).streak_c.map(cid).value_counts()
                if len(m):
                    label_id = m.index[0]
        # inset: the rows on the year plane
        axi = fig.add_axes(rect(x0 + 0.45, x1 - 0.1, INSET_TOP, INSET_H))
        axi.scatter(sub.R, sub.F, s=0.2, color="0.85", lw=0, rasterized=True)
        gs_ = P_all[P_all.streak == sid]
        axi.scatter(gs_.R, gs_.F, s=0.3, color="tab:orange", lw=0, alpha=0.6, rasterized=True)
        for i, grp in enumerate(groups):           # the row's position: a small open ring on the streak, its number just above (MBE, 2026-10-13)
            axi.scatter([grp.R.median()], [grp.F.median()], s=14, facecolors="none", edgecolors="k", lw=0.5, zorder=5)
            axi.annotate(str(i + 1), (grp.R.median(), grp.F.median()), xytext=(0, 3.2), textcoords="offset points", fontsize=4.5, ha="center", va="bottom", zorder=6)
        axi.set_xscale("log"); axi.set_yscale("log"); axi.set_xlim(10, 100); axi.set_ylim(1500, 20000)
        axi.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100])); axi.xaxis.set_minor_locator(FixedLocator([]))
        axi.yaxis.set_major_locator(FixedLocator([2000, 5000, 10000, 20000])); axi.yaxis.set_minor_locator(FixedLocator([]))
        axi.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axi.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axi.tick_params(labelsize=5, length=1.5, pad=1); axi.set_xlabel("rate (Hz)", fontsize=5, labelpad=1); axi.set_ylabel("carrier (kHz)", fontsize=5, labelpad=1)
        fig.text((x0 - fx0) / Wc, 1 - (INSET_TOP - 1.0 - 0.08) / Hc, f"({chr(65 + k)}) Type {label_id}\n{S.loc[sid, 'F25'] / 1000:.1f} kHz at {S.loc[sid, 'R25']:.0f} Hz, {who}",
                 fontsize=8, fontweight="bold", va="bottom", ha="left")
        # the grid
        cw = (COL_W - (COLS_N - 1) * CELL_WGAP) / COLS_N
        axes = {}
        for (i, j, r, w) in cells:
            top = GRID_TOP + i * (CELL_H + CELL_GAP)
            ax = fig.add_axes(rect(x0 + j * (cw + CELL_WGAP), x0 + j * (cw + CELL_WGAP) + cw, top, CELL_H)); axes[(i, j)] = ax
            if w is None:
                ax.text(0.5, 0.5, "no window", ha="center", va="center", fontsize=5, transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); continue
            ex, f, dt = w["ex_db"], w["f"], w["dt"]
            n_half = int(round(DUR / dt))
            rows_b = (f >= w["F_lo"]) & (f <= w["F_hi"])
            bp = ex[rows_b].sum(axis=0) if rows_b.any() else ex.sum(axis=0)
            kk = int(np.argmax(np.convolve(bp, np.ones(n_half), "valid"))) if ex.shape[1] > n_half else 0
            sr = w["sr"]; a0 = kk * 128; a1 = a0 + n_half * 128
            ff, tf, Pf = SD.fine_spec(w["sig_ex"][a0:a1], sr)
            Df = SD.db_above_quiet(Pf, SD.quiet_on_grid(ff, f, w["quiet"]))
            sel = (ff >= ylo) & (ff <= yhi)
            if not args.veil:                       # experiment (MBE, 2026-10-13): rows outside the detected band in grayscale instead of under a veil
                inb = (ff[sel] >= w["F_lo"]) & (ff[sel] <= w["F_hi"])
                Dm_in = np.ma.masked_array(Df[sel], mask=np.broadcast_to(~inb[:, None], Df[sel].shape))
                Dm_out = np.ma.masked_array(Df[sel], mask=np.broadcast_to(inb[:, None], Df[sel].shape))
                ax.pcolormesh(tf, ff[sel], Dm_out, cmap="gray", vmin=0, vmax=30, shading="auto", rasterized=True)
                ax.pcolormesh(tf, ff[sel], Dm_in, cmap="magma", vmin=0, vmax=30, shading="auto", rasterized=True)
            else:
                ax.pcolormesh(tf, ff[sel], Df[sel], cmap="magma", vmin=0, vmax=30, shading="auto", rasterized=True)
                ax.axhspan(ylo, w["F_lo"], color="white", alpha=MASK_ALPHA, lw=0); ax.axhspan(w["F_hi"], yhi, color="white", alpha=MASK_ALPHA, lw=0)
            ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(0, DUR)
            ax.set_title(f"{w['t']:%d %b} · {r['T']:.0f} °C · {w['R']:.0f} Hz", fontsize=4, loc="left", pad=1.5)
            ticks = [t for t in [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000] if ylo <= t <= yhi]
            if len(ticks) < 2:
                ticks = list(np.round(np.geomspace(ylo * 1.02, yhi * 0.98, 3), -2))
            ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if j > 0:
                ax.set_yticklabels([])
            if i < len(groups) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.25, 0.5]); ax.set_xticklabels(["0", "0.25", "0.5"])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
        for i, grp in enumerate(groups):
            if (i, 0) in axes:
                axes[(i, 0)].set_ylabel(f"{i + 1}: {grp.R.median():.0f} Hz, {grp.F.median() / 1000:.2f} kHz", fontsize=4.5, labelpad=1)
        if (len(groups) - 1, 0) in axes:
            axes[(len(groups) - 1, 1)].set_xlabel("s", fontsize=5, labelpad=1) if (len(groups) - 1, 1) in axes else None
        print(f"streak {sid}: {len(ok)} cells", flush=True)
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
