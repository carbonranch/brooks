"""Figure 2: recorder locations by area, each in its own 200 m square of the drone orthomosaic, with the recording
on/off record of every recorder at that location under it, and a locator of the whole ranch below.

Layout follows MBE's Figure 2 of 2026-10-10 (MBE_Figures/MBE_F02_sites.pdf): a 3 x 4 grid of the eleven areas in
alphabetical order (A to K), each a 200 m square of the May 2024 orthomosaic centered on the area's recorder (or
recorder centroid), every recorder a yellow dot labeled by its number, a 50 m bar; under the square one strip per
recorder across the 13 months (Nov 2024 to Nov 2025), blue on the days it recorded (>= 0.5 h of audio), white where
it did not, month initials along the bottom. Below the grid the whole ranch (the orthomosaic cropped to the
boundary) with the eleven squares in red and lettered, a 1 km bar and a north arrow.

Positions come from data/deployments.parquet, scoped to the (area, recorder) deployment that overlaps the period
(the recorder's own GPS is not used). Hours per day come from the audio catalog (wavfiles.parquet).

Usage:
  python analysis/callers/site_panels_figure.py --out analysis/callers/runs/figures_paper/F02_sites
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_page import PaperPage, BODY_PT  # noqa: E402
from site_map_figure import PERIOD, MOSAIC, BOUNDARY, recorders_in_period, positions, ranch_boundary, mosaic_crop  # noqa: E402

HALF_M = 100.0                  # half-width of each square (200 m square)
SCALE_M = 50.0                  # scale bar in each square
ON_HOURS = 0.5                  # a day counts as recorded at or above this many hours of audio
ON_COLOR = "#2b6cb0"
DOT_COLOR = "#ffd54a"
BOX_COLOR = "#e02020"
NCOL, NROW = 4, 3
# page geometry (inches from the page's top-left), after MBE's layout
COL_X0, COL_PITCH, SQ = 1.13, 1.76, 1.20
ROW_Y0, ROW_PITCH = 1.17, 2.00
STRIP_GAP, STRIP_IN = 0.04, 0.055
LOC_X0, LOC_W, LOC_TOP = 2.50, 3.90, 7.10


def short(rec: str) -> str:
    return rec.replace("BARLT_0000", "").replace("BARLT_000", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog", default="wavfiles.parquet")
    ap.add_argument("--deployments", default="data/deployments.parquet")
    ap.add_argument("--mosaic", default=MOSAIC)
    ap.add_argument("--boundary", default=BOUNDARY if os.path.exists(BOUNDARY) else None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    H = recorders_in_period(args.catalog)
    pairs = H.groupby(["recorder_group", "recorder"]).hours.sum().reset_index()
    P = positions(args.deployments, pairs)
    areas = sorted(P.area.unique(), key=str.lower)
    assert len(areas) <= NCOL * NROW, f"{len(areas)} areas do not fit a {NROW} x {NCOL} grid"
    days = pd.date_range(PERIOD[0], PERIOD[1], freq="D")
    M = H.pivot_table(index="recorder", columns="date", values="hours", aggfunc="sum").reindex(columns=days).fillna(0.0)
    edges = mdates.date2num(np.append(days.values, days.values[-1] + np.timedelta64(1, "D")))
    months = pd.date_range(days[0], days[-1], freq="MS")
    print(f"{len(P)} recorders in {len(areas)} areas; {H.hours.sum():,.0f} h over {len(days)} days")

    page = PaperPage(number=2, title="Recorder locations and uptime")
    Wc, Hc = 7.0, page.size[1]                                    # a canvas wider than the margins, so strip labels left of the first column are not clipped
    fx0 = 1.0 + (6.5 - Wc) / 2                                    # the canvas's left edge on the page (PaperPage centers it)
    fig = plt.figure(figsize=(Wc, Hc))

    def axes_in(x_page, top_page, w, h):
        """axes from page inches (x from the left edge, top from the top edge)."""
        return fig.add_axes([(x_page - fx0) / Wc, 1 - (top_page - 1.0 + h) / Hc, w / Wc, h / Hc])

    halo = [pe.withStroke(linewidth=1.6, foreground="black", alpha=0.9)]
    squares = {}
    for k, area in enumerate(areas):
        r, c = divmod(k, NCOL)
        letter = chr(ord("A") + k)
        x_page, top = COL_X0 + c * COL_PITCH, ROW_Y0 + r * ROW_PITCH
        g = P[P.area == area]
        cx, cy = g.x.mean(), g.y.mean()
        squares[letter] = (cx, cy)
        ax = axes_in(x_page, top, SQ, SQ)
        got = mosaic_crop(args.mosaic, cx - HALF_M, cy - HALF_M, cx + HALF_M, cy + HALF_M, None, max_px=700)
        if got is not None:
            img, ext = got; ax.imshow(img, extent=[ext[0], ext[1], ext[2], ext[3]], interpolation="bilinear", zorder=0)
        ax.set_facecolor("#e8e8e4")
        ax.scatter(g.x, g.y, s=14, color=DOT_COLOR, edgecolor="black", linewidth=0.5, zorder=3)
        for _, p in g.iterrows():
            dx, dy = p.x - cx, p.y - cy
            n = np.hypot(dx, dy)
            ux, uy = (dx / n, dy / n) if n > 1 else (1.0, 0.0)
            lx, ly = p.x + 9 * ux, p.y + 9 * uy
            lx = float(np.clip(lx, cx - HALF_M + 14, cx + HALF_M - 14)); ly = float(np.clip(ly, cy - HALF_M + 8, cy + HALF_M - 8))
            ax.text(lx, ly, short(p.recorder), fontsize=5, color="white", fontweight="bold", path_effects=halo, zorder=4,
                    ha="left" if ux >= 0 else "right", va="center")
        sx0, sy0 = cx - HALF_M + 10, cy - HALF_M + 10
        ax.plot([sx0, sx0 + SCALE_M], [sy0, sy0], color="white", lw=1.8, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=3, foreground="black")], zorder=5)
        ax.text(sx0 + SCALE_M / 2, sy0 + 5, f"{SCALE_M:.0f} m", ha="center", va="bottom", color="white", fontsize=4.5, fontweight="bold", path_effects=halo, zorder=5)
        ax.set_xlim(cx - HALF_M, cx + HALF_M); ax.set_ylim(cy - HALF_M, cy + HALF_M)
        ax.set_xticks([]); ax.set_yticks([])
        page.label(ax, letter, area, y=1 + 0.05 / SQ)
        # strips
        recs = sorted(g.recorder)
        hs = len(recs) * STRIP_IN
        axs = axes_in(x_page, top + SQ + STRIP_GAP, SQ, hs)
        on = (M.reindex(index=recs).values >= ON_HOURS).astype(float)
        axs.pcolormesh(edges, np.arange(len(recs) + 1), on, cmap=ListedColormap(["white", ON_COLOR]), vmin=0, vmax=1, shading="flat", rasterized=True)
        for yy in range(1, len(recs)):
            axs.axhline(yy, color="white", lw=0.5)
        axs.set_ylim(len(recs), 0); axs.set_yticks(np.arange(len(recs)) + 0.5); axs.set_yticklabels([short(x) for x in recs], fontsize=4.5)
        axs.tick_params(axis="y", length=0, pad=1.5)
        axs.set_xticks(mdates.date2num(months.values)); axs.set_xticklabels([m.strftime("%b")[0] for m in months])
        axs.tick_params(axis="x", labelsize=3.8, length=1.5, pad=1)
        axs.set_xlim(edges[0], edges[-1])
        for sp in list(ax.spines.values()) + list(axs.spines.values()):
            sp.set_linewidth(0.5)

    # the locator: the ranch with the squares
    if args.boundary:
        boundary = ranch_boundary(args.boundary)
        bx0, by0, bx1, by1 = boundary.bounds
        pad = 60.0
        xmin, xmax, ymin, ymax = bx0 - pad, bx1 + pad, by0 - pad, by1 + pad
        loc_h = LOC_W * (ymax - ymin) / (xmax - xmin)
        ax = axes_in(LOC_X0, LOC_TOP, LOC_W, loc_h); ax.set_axis_off()
        got = mosaic_crop(args.mosaic, xmin, ymin, xmax, ymax, boundary, max_px=2400)
        if got is not None:
            img, ext = got; ax.imshow(img, extent=[ext[0], ext[1], ext[2], ext[3]], interpolation="bilinear", zorder=0)
        for poly in getattr(boundary, "geoms", [boundary]):
            bx, by = poly.exterior.xy; ax.plot(bx, by, color="white", lw=0.5, alpha=0.8, zorder=1)
        for L, (cx, cy) in squares.items():
            ax.add_patch(plt.Rectangle((cx - HALF_M, cy - HALF_M), 2 * HALF_M, 2 * HALF_M, fill=False, ec=BOX_COLOR, lw=0.8, zorder=3))
            ax.text(cx + HALF_M + 25, cy, L, ha="left", va="center", fontsize=9, fontweight="bold", color="black", zorder=4)
        ax.set_aspect("equal"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        sx0, sy0 = xmin + 40, ymin + 40
        ax.plot([sx0, sx0 + 1000], [sy0, sy0], color="0.2", lw=1.5, solid_capstyle="butt", zorder=5)
        ax.text(sx0 + 500, sy0 - 30, "1 km", ha="center", va="top", fontsize=BODY_PT, color="0.2")
        nx, ny = xmax - 120, ymax - 320
        ax.annotate("N", (nx, ny + 220), (nx, ny), ha="center", va="center", fontsize=BODY_PT, fontweight="bold", color="0.2",
                    arrowprops=dict(arrowstyle="-|>", color="0.2", lw=0.8))
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
