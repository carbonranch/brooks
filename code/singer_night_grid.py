"""A grid of individual singers through their nights: for one type, rows are singers (tracks) whose night spanned a
wide temperature range, columns are moments through that night in time order, each cell the singer's song at that
moment, so the reader sees one animal's carrier, rate and syllable change with temperature (MBE 2026-10-15: "a version
of the grid except showing across a streak the temperature changes of individual singers").

Rows: the type's firm trill tracks with >= --cols + 1 hours, depth in the upper 60 %, the largest temperature range
within the night first, distinct recorder-nights, ordered top to bottom by the singer's median rate (slow to fast).
Columns: --cols windows at even steps through the track's span, the deepest window with a syllable rate within
+/- 20 min of each target time; the window's air temperature is interpolated from the type's hourly table.
Cells as in streak_grid (0.5 s of the loudest stretch of the window's loudest 30 s, band in color, rest in grayscale,
one shared log-frequency axis). Top right: each singer's hourly path on the plane over the year's points.

Usage:
  python analysis/callers/singer_night_grid.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --unit-roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --types 5 16 \
      --out-dir analysis/callers/runs/figures_paper/exploratory/singer_nights
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ids  # noqa: E402
import specdisplay as SD  # noqa: E402
import streak_gallery as G  # noqa: E402
import streak_grid as SG  # noqa: E402
from paper_page import PaperPage, BODY_PT, LABEL_PT  # noqa: E402

TZ = G.TZ


def track_windows(an: str, unit_roots: list[str]) -> pd.DataFrame | None:
    unit, track = an.split(":"); track = int(track)
    for root in unit_roots:
        p = os.path.join(root, unit, "night_night_global_window_animals.csv")
        if os.path.exists(p):
            Wd = pd.read_csv(p)
            g = Wd[(Wd.track == track) & Wd.F_fix.notna() & Wd.R.notna() & (Wd.R > 0)]
            if "keep" in g.columns:
                g = g[g.keep == True]  # noqa: E712
            if "chirp_rate" in g.columns:
                g = g[g.chirp_rate.isna()] if g.chirp_rate.isna().any() else g
            g = g.assign(t=pd.to_datetime(g.t, utc=True)).sort_values("t")
            parts = unit.split("_")
            return g.assign(recorder=parts[0] + "_" + parts[1], site=parts[2])
    return None


def pick_windows(g: pd.DataFrame, hours: pd.DataFrame, cols: int) -> list[dict]:
    """cols windows at even steps through the track's span, the deepest with a syllable rate within +/- 20 min of each target."""
    t = g.t.astype("int64").to_numpy() / 1e9; th = hours.hour.astype("int64").to_numpy() / 1e9
    T = np.interp(t, th, hours["T"].to_numpy())
    g = g.assign(T=T)
    targets = np.linspace(t.min(), t.max(), cols)
    out, used = [], set()
    for tq in targets:
        near = g[(np.abs(t - tq) <= 20 * 60) & ~g.index.isin(used)]
        if near.empty:
            near = g[~g.index.isin(used)].assign(d=np.abs(g.t.astype("int64").to_numpy()[~g.index.isin(used)] / 1e9 - tq)).nsmallest(1, "d")
        if near.empty:
            continue
        b = near.sort_values("depth", ascending=False).iloc[0]; used.add(b.name)
        out.append(dict(recorder=b.recorder, site=b.site, t=b.t.tz_convert(TZ), R=float(b.R), F_fix=float(b.F_fix), F_lo=float(b.F_lo), F_hi=float(b.F_hi), depth=float(b.depth),
                        Rc=np.nan, ref_lo=float(b.ref_lo) if pd.notna(b.get("ref_lo", np.nan)) else float(b.F_lo), ref_hi=float(b.ref_hi) if pd.notna(b.get("ref_hi", np.nan)) else float(b.F_hi), T=float(b["T"])))
    return sorted(out, key=lambda w: w["t"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--types", nargs="+", required=True, help="paper numbers (or streak ids with --streak-ids)"); ap.add_argument("--streak-ids", action="store_true")
    ap.add_argument("--rows", type=int, default=8); ap.add_argument("--cols", type=int, default=5); ap.add_argument("--dur", type=float, default=0.5)
    ap.add_argument("--min-hours", type=int, default=6); ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); SD.paper_fonts()
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv")); S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    ids = paper_ids.load(args.root, args.suffix); inv = {str(v): k for k, v in ids.items()}
    P = pd.read_csv(os.path.join(args.root, "streak_hourly_points_all.csv"), usecols=["hour", "R", "F", "T", "an"]).dropna(subset=["T"])
    P["hour"] = pd.to_datetime(P.hour, utc=True)
    P_all = P.merge(A[["an", "streak"]], on="an", how="left")
    for tname in args.types:
        sid = int(tname) if args.streak_ids else inv[str(tname)]; pid = ids.get(sid, sid)
        g = A[(A.streak == sid)]
        if "form" in g.columns and "form" in S.columns:
            g = g[g.form == S.loc[sid, "form"]]
        H = P[P.an.isin(g.an)]
        st = H.groupby("an").agg(n=("T", "size"), dT=("T", lambda x: x.max() - x.min()), Rmed=("R", "median")).join(g.set_index("an")[["depth", "recorder", "night", "T"]])
        st = st[(st.n >= args.min_hours) & (st.depth >= st.depth.quantile(0.4))].sort_values("dT", ascending=False)
        picks, used = [], set()
        for an, r in st.iterrows():
            if (r.recorder, r.night) in used:
                continue
            used.add((r.recorder, r.night)); picks.append(an)
            if len(picks) >= args.rows:
                break
        picks = sorted(picks, key=lambda a: st.loc[a, "Rmed"])
        print(f"type {pid}: {len(picks)} singers, temperature ranges {[round(st.loc[a, 'dT'], 1) for a in picks]}")
        cells = []
        for i, an in enumerate(picks):
            gw = track_windows(an, args.unit_roots)
            if gw is None or gw.empty:
                continue
            ws = pick_windows(gw, H[H.an == an].sort_values("hour"), args.cols)
            for j, w in enumerate(ws):
                try:
                    x = G.analyse(dict(w)); cells.append((i, j, an, dict(w, **x), ""))
                except SystemExit:
                    cells.append((i, j, an, None, "no audio"))
                except Exception as e:  # noqa: BLE001
                    cells.append((i, j, an, None, type(e).__name__))
        ok = [c for c in cells if c[3] is not None]
        if not ok:
            print(f"  no cells"); continue
        ylo = 0.8 * min(c[3]["F_lo"] for c in ok); yhi = 1.25 * max(c[3]["F_hi"] for c in ok)
        nrow, ncol = len(picks), args.cols
        legend = (f"Rows are individual singers of the type, the eight with the widest temperature range within one night (top to bottom by the singer's median rate); "
                  f"columns are moments through that singer's night in time order, {args.cols} windows at even steps through its span. Each cell is {args.dur:g} s of the loudest 30 s "
                  "of that window on one shared log-frequency axis, the detected band in color (dB above the row's quiet level, 0 to 30) and the rest in grayscale. "
                  "The cell head gives the local time, the air temperature, the syllable rate and the carrier. Top right: each singer's hourly path on the plane, in the row's color, over the year's points in gray.")
        label = f"Type {pid} (streak {sid})"
        page = PaperPage(number=f"S8-{pid}", title=f"{label}: {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz at 25 °C, eight singers through their nights.", legend=legend)
        Wc, Hc = 7.2, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
        fig = plt.figure(figsize=(Wc, Hc))

        def rect(x0, x1, top, h):
            return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
        X0, X1 = fx0 + 0.62, fx0 + Wc - 0.03
        GRID_TOP, HEAD_H, GAP = 1.95, 0.10, 0.04
        pitch_h = (1.0 + Hc - GRID_TOP - 0.22) / nrow; cell_h = pitch_h - HEAD_H - GAP
        pitch_w = (X1 - X0) / ncol; cell_w = pitch_w - 0.07
        fig.text(0.02 / Wc, 1 - 0.02 / Hc, f"{label}, {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz: singers through their nights", fontsize=LABEL_PT, fontweight="bold", va="top", ha="left")
        fig.text(0.02 / Wc, 1 - 0.20 / Hc, "rows: individual singers, the widest within-night temperature ranges, slow to fast\ncolumns: moments through the singer's night, in time order",
                 fontsize=BODY_PT, va="top", ha="left", linespacing=1.3)
        ticks = [t for t in [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000] if ylo <= t <= yhi]
        cols10 = plt.get_cmap("tab10")
        first_axes = {}
        for (i, j, an, w, err) in cells:
            top = GRID_TOP + i * pitch_h + HEAD_H
            ax = fig.add_axes(rect(X0 + j * pitch_w, X0 + j * pitch_w + cell_w, top, cell_h))
            if j == 0:
                first_axes[i] = ax
            ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(0, args.dur)
            ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
            ax.set_xticks([0, args.dur / 2, args.dur]); ax.tick_params(labelsize=BODY_PT - 1, length=1.5, pad=1.5, width=0.5)
            if j > 0:
                ax.set_yticklabels([])
            if i < nrow - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(["0", f"{args.dur / 2:g}", f"{args.dur:g}" if j == ncol - 1 else ""])
                if j == 0:
                    ax.set_xlabel("s", fontsize=BODY_PT - 1, labelpad=1)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_edgecolor(cols10(i % 10))
            if w is None:
                ax.text(0.5, 0.5, err, ha="center", va="center", fontsize=BODY_PT - 1, transform=ax.transAxes, color="0.4"); continue
            ff, tf, Df, _, _ = SG.cell_image(w, args.dur, 0)
            sel = (ff >= ylo) & (ff <= yhi); inb = (ff[sel] >= w["F_lo"]) & (ff[sel] <= w["F_hi"])
            ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(inb[:, None], Df[sel].shape)), cmap="gray", vmin=0, vmax=30, shading="auto", rasterized=True)
            ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(~inb[:, None], Df[sel].shape)), cmap="magma", vmin=0, vmax=30, shading="auto", rasterized=True)
            ax.set_title(f"{w['t']:%H:%M}  {w['T']:.1f} °C  {w['R']:.1f} Hz  {w['F_fix'] / 1000:.2f} kHz", fontsize=4.5, loc="left", pad=1.2)
        for i, an in enumerate(picks):
            if i in first_axes:
                h = H[H.an == an]; night = str(st.loc[an, "night"])[2:]
                first_axes[i].set_ylabel(f"{night}\n{st.loc[an, 'recorder'][-4:]} {an.split('_')[2][:6]}\n{h['T'].min():.0f}–{h['T'].max():.0f} °C", fontsize=BODY_PT - 1.5, labelpad=2)
        # inset: the singers' hourly paths on the plane
        Hp = H[H.an.isin(picks)]; sub = P_all[P_all.streak == sid]
        xlo, xhi = Hp.R.min() / 1.3, Hp.R.max() * 1.3; ylo_, yhi_ = Hp.F.min() / 1.2, Hp.F.max() * 1.2
        axi = fig.add_axes(rect(fx0 + 5.1, fx0 + Wc - 0.03, 1.0 + 0.22, 0.66))   # clear of the head line
        axi.scatter(sub.R, sub.F, s=0.6, color="0.85", lw=0, rasterized=True)
        for i, an in enumerate(picks):
            h = H[H.an == an].sort_values("hour"); axi.plot(h.R, h.F, color=cols10(i % 10), lw=0.8, alpha=0.9); axi.scatter(h.R.iloc[[0]], h.F.iloc[[0]], s=6, color=cols10(i % 10), zorder=5)
        axi.set_xscale("log"); axi.set_yscale("log"); axi.set_xlim(xlo, xhi); axi.set_ylim(ylo_, yhi_)
        xt = [t for t in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100] if xlo <= t <= xhi]; yt = [t for t in ticks if ylo_ <= t <= yhi_]
        axi.xaxis.set_major_locator(FixedLocator(xt)); axi.yaxis.set_major_locator(FixedLocator(yt)); axi.xaxis.set_minor_locator(FixedLocator([])); axi.yaxis.set_minor_locator(FixedLocator([]))
        axi.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axi.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axi.tick_params(labelsize=BODY_PT - 1.5, length=1.5, pad=1); axi.set_title("the singers' paths on the streak (dot = first hour)", fontsize=BODY_PT, pad=2)
        axi.set_xlabel("Hz", fontsize=BODY_PT - 1, labelpad=0); axi.set_ylabel("kHz", fontsize=BODY_PT - 1, labelpad=0)
        base = os.path.join(args.out_dir, f"type_{pid:02d}_singers" if isinstance(pid, int) else f"type_{pid}_singers")
        page.finish(fig, base); plt.close(fig)
        for ext in ("_content.pdf", "_content.png"):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        print(f"  wrote {base}.pdf / .png ({len(ok)} cells)")


if __name__ == "__main__":
    main()
