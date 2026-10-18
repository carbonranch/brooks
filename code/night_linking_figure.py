"""Figure 6: linking the singers of one night into tracks (House 8662, 20 Sep 2025, the night of Figs 3-5).

  (A) Syllable rate of every kept window-animal against time of night, with the common drift the linker reads from
      the data drawn through a few reference rates: singers cool together and their rates fall along it
  (B) The common drift itself, as percent change in rate, with the HRRR air temperature
  (C) The same window-animals with the common drift removed (rate referred to 19:00): a singer's detrended rate is
      nearly constant all night; the tracks of two hours or more colored, the rest gray
  (D) Carrier of the same window-animals against time, the same tracks in the same colors; the track Fig 3 follows
      outlined in black
  (E) That track in detail: its windows' rate (with the common drift it rides) and carrier, the windows it skipped,
      and the other kept singers in its band that were not joined to it
  (F) The night's longest track the same way
  (G) Per window: singers found, kept by the postfilter, and in a track of two hours or more
  (H) The night's tracks by length

Usage:
    python analysis/callers/night_linking_figure.py --unit-dir analysis/callers/runs/v3/y2025/BARLT_00008662_House_20250920_night \
        --weather analysis/callers/runs/v2/y2025/weather_hourly_hrrr.csv --track 51 33 --out analysis/callers/runs/figures_paper/F06_night_linking
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs2 as W2  # noqa: E402
import night_global as NG  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402

MIN_H = 2.0                    # a track counts as long from this span
RLO, RHI = 10.0, 90.0
FLO, FHI = 1500.0, 20000.0
# page geometry (inches from the page's top-left)
X0, X1 = 1.35, 7.35
ROWS = {"A": (1.15, 0.92), "B": (2.37, 0.33), "C": (3.12, 0.92), "D": (4.34, 0.92), "E": (5.85, 0.68), "F": (6.98, 0.68), "GH": (8.10, 0.60)}
XL, XR = (1.35, 4.25), (4.75, 7.35)
T_COLOR = "tab:blue"
TRACK_CMAP = "tab20"


def log_axis(ax, lo, hi, ticks, fmt=lambda v, p: f"{v:g}", which="y"):
    a = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_yscale if which == "y" else ax.set_xscale)("log"); (ax.set_ylim if which == "y" else ax.set_xlim)(lo, hi)
    a.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi])); a.set_minor_locator(FixedLocator([])); a.set_major_formatter(FuncFormatter(fmt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-dir", required=True); ap.add_argument("--weather", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--track", type=int, nargs="+", required=True, help="tracks to show in detail; the first is the one Fig 3 follows")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tag = "night" if args.unit_dir.rstrip("/").endswith("_night") else "day"
    A = pd.read_csv(os.path.join(args.unit_dir, f"night_{tag}_global_window_animals.csv")); A["t"] = pd.to_datetime(A.t, utc=True).dt.tz_convert(W2.TZ)
    T = pd.read_csv(os.path.join(args.unit_dir, f"night_{tag}_global_tracks.csv"))
    D = pd.read_csv(os.path.join(args.unit_dir, f"night_{tag}_global_drift.csv"), index_col=0); D.index = pd.to_datetime(D.index, utc=True).tz_convert(W2.TZ)
    delta = D.iloc[:, 0]
    Wx = pd.read_csv(args.weather); Wx["t"] = pd.to_datetime(Wx.t, utc=True).dt.tz_convert(W2.TZ); Wx = Wx.dropna(subset=["T_C"]).sort_values("t")
    t0, t1 = delta.index[0], delta.index[-1] + pd.Timedelta(minutes=5)
    wn = Wx[(Wx.t >= t0 - pd.Timedelta(hours=1)) & (Wx.t <= t1 + pd.Timedelta(hours=1))]
    hhmm = mdates.DateFormatter("%H:%M", tz=W2.TZ)

    kept = A[A.keep.astype(bool)]
    withR = kept[kept.R.notna()].copy()
    withR["R_detr"] = np.exp(withR.y_detr)                         # the rate referred to the night's start (delta(19:00) ~ 0)
    long_tracks = T[T.span_h >= MIN_H].sort_values("span_h", ascending=False)
    long_ids = list(long_tracks.track)
    cmap = plt.get_cmap(TRACK_CMAP)
    color_of = {tr: cmap(i % 20) for i, tr in enumerate(long_ids)}
    n_long = len(long_ids)
    print(f"{len(A)} window-animals, {len(kept)} kept, {len(withR)} with a rate; {len(T)} tracks, {n_long} >= {MIN_H:g} h; drift end {100 * (np.exp(delta.iloc[-1]) - 1):+.1f} %")

    page = PaperPage(number=6, title="Linking singers across one night")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]

    def time_axis(ax, label=False):
        ax.set_xlim(t0, t1); ax.xaxis.set_major_formatter(hhmm); ax.xaxis.set_major_locator(mdates.HourLocator(interval=2 if ax.get_position().width < 0.5 else 1))
        if label:
            ax.set_xlabel("time of night")

    def draw_tracks(ax, col, xcol="t"):
        """gray points for every kept window-animal with a rate; the long tracks as colored points joined window to window."""
        ax.scatter(withR.t, withR[col], s=1.2, color="0.6", lw=0, zorder=2)
        for tr in long_ids:
            g = withR[withR.track == tr].sort_values("t")
            ax.plot(g.t, g[col], "-", color=color_of[tr], lw=0.6, alpha=0.9, zorder=3)
            ax.scatter(g.t, g[col], s=3, color=color_of[tr], lw=0, zorder=4)
        for tr in args.track:
            g = withR[withR.track == tr].sort_values("t")
            ax.plot(g.t, g[col], "-", color="k", lw=1.4, zorder=5); ax.plot(g.t, g[col], "-", color=color_of.get(tr, "w"), lw=0.6, zorder=6)

    # ---- (A) rate against time, with the common drift through reference rates ----
    top, h = ROWS["A"]
    ax = fig.add_axes(rect(X0, X1, top, h))
    ax.scatter(withR.t, withR.R, s=1.2, color="0.45", lw=0, zorder=2)
    for R_ref in (15.0, 20.0, 30.0, 45.0, 65.0):
        ax.plot(delta.index, R_ref * np.exp(delta.values), "-", color="tab:red", lw=0.8, alpha=0.9, zorder=3)
    ax.text(delta.index[2], 65.0 * 1.06, "the common drift, through reference rates", color="tab:red", fontsize=BODY_PT, va="bottom")
    log_axis(ax, RLO, RHI, (10, 15, 20, 30, 40, 50, 60, 80)); ax.set_ylabel("syllable rate (Hz)")
    time_axis(ax)
    page.label(ax, "A", "Syllable rate through the night", y=1 + 0.05 / h)

    # ---- (B) the common drift and the temperature ----
    top, h = ROWS["B"]
    ax = fig.add_axes(rect(X0, X1, top, h))
    ax.plot(delta.index, 100 * (np.exp(delta.values) - 1), "-", color="tab:red", lw=1.0)
    ax.set_ylabel("rate change (%)"); ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylim(min(-30, 100 * (np.exp(delta.min()) - 1) - 2), 3)
    axT = ax.twinx(); axT.plot(wn.t, wn.T_C, "-", color=T_COLOR, lw=0.8); axT.set_ylabel("°C", color=T_COLOR); axT.tick_params(axis="y", colors=T_COLOR)
    time_axis(ax)
    page.label(ax, "B", "Common drift and temperature", y=1 + 0.05 / h)

    # ---- (C) detrended rate against time with the long tracks ----
    top, h = ROWS["C"]
    ax = fig.add_axes(rect(X0, X1, top, h))
    draw_tracks(ax, "R_detr")
    log_axis(ax, RLO, RHI, (10, 15, 20, 30, 40, 50, 60, 80)); ax.set_ylabel("rate, drift removed (Hz)")
    time_axis(ax)
    page.label(ax, "C", "Drift removed, tracks colored", y=1 + 0.05 / h)

    # ---- (D) carrier against time with the same tracks ----
    top, h = ROWS["D"]
    ax = fig.add_axes(rect(X0, X1, top, h))
    draw_tracks(ax, "F_c")
    log_axis(ax, FLO, FHI, (1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000), lambda v, p: f"{v / 1000:g}"); ax.set_ylabel("carrier (kHz)")
    time_axis(ax, label=True)
    page.label(ax, "D", "Carrier through the night", y=1 + 0.05 / h)

    # ---- (E), (F) one track in detail each ----
    def track_detail(letter, key, track_id, note):
        top, h = ROWS[key]
        tr = T.set_index("track").loc[track_id]
        g = withR[withR.track == track_id].sort_values("t")
        f_lo, f_hi = g.F_lo.min() - 150, g.F_hi.max() + 150
        near = withR[(withR.track != track_id) & (withR.F_c >= f_lo) & (withR.F_c <= f_hi)]            # other kept singers whose carrier falls in the track's band (+/- F_PAD)
        ta, tb = g.t.min() - pd.Timedelta(minutes=30), g.t.max() + pd.Timedelta(minutes=30)
        near = near[(near.t >= ta) & (near.t <= tb)]
        y_mean = float(np.log(g.R).sub(delta.reindex(g.t).values).mean())
        allw = delta.index[(delta.index >= g.t.min()) & (delta.index <= g.t.max())]
        skipped = [w for w in allw if w not in set(g.t)]
        ax = fig.add_axes(rect(XL[0], XL[1], top, h))
        ax.scatter(near.t, near.R, s=6, facecolors="none", edgecolors="0.5", lw=0.5, zorder=2)
        ax.plot(delta.index, np.exp(y_mean + delta.values), "-", color="tab:red", lw=0.7, alpha=0.8, zorder=3)
        ax.plot(g.t, g.R, "-", color="k", lw=0.6, zorder=4); ax.scatter(g.t, g.R, s=7, color="k", lw=0, zorder=5)
        ax.set_ylim(g.R.min() * 0.85, g.R.max() * 1.15)
        for w in skipped:
            ax.plot([w, w], [g.R.max() * 1.10, g.R.max() * 1.15], color="0.4", lw=0.5, zorder=1)
        ax.set_xlim(ta, tb); ax.xaxis.set_major_formatter(hhmm); ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.set_ylabel("syllable rate (Hz)")
        page.label(ax, letter, f"Track {track_id}", y=1 + 0.05 / h)
        ax = fig.add_axes(rect(XR[0], XR[1], top, h))
        ax.scatter(near.t, near.F_c, s=6, facecolors="none", edgecolors="0.5", lw=0.5, zorder=2)
        ax.plot(g.t, g.F_c, "-", color="k", lw=0.6, zorder=4); ax.scatter(g.t, g.F_c, s=7, color="k", lw=0, zorder=5)
        ax.fill_between(g.t, g.F_lo, g.F_hi, color="k", alpha=0.08, lw=0, zorder=1)
        ax.set_xlim(ta, tb); ax.xaxis.set_major_formatter(hhmm); ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.set_ylabel("carrier (Hz)"); ax.set_ylim(f_lo - 200, f_hi + 200)
        if letter == "E":
            ax.text(0.02, 0.96, "hollow: other kept singers in the band, not joined; shaded: the detected band", transform=ax.transAxes, fontsize=5, va="top", color="0.3",
                    bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8))
        return ax

    axE = track_detail("E", "E", args.track[0], "; its rate rides the common drift (red)")
    if len(args.track) > 1:
        track_detail("F", "F", args.track[1], "")

    # ---- (G) per window counts ----
    top, h = ROWS["GH"]
    ax = fig.add_axes(rect(XL[0], XL[1], top, h))
    per = pd.DataFrame({"found": A.groupby("t").size(), "kept": kept.groupby("t").size(),
                        "in long track": withR[withR.track.isin(long_ids)].groupby("t").size()}).reindex(delta.index).fillna(0)
    ax.plot(per.index, per.found, color="0.6", lw=0.8, label="found"); ax.plot(per.index, per.kept, color="0.25", lw=0.8, label="kept by the postfilter")
    ax.plot(per.index, per["in long track"], color="tab:green", lw=0.8, label=f"in a track of {MIN_H:g} h or more")
    ax.legend(fontsize=5, loc="upper right", frameon=False); ax.set_ylabel("singers per window"); ax.set_ylim(0, None)
    time_axis(ax, label=True)
    page.label(ax, "G", "Singers per window", y=1 + 0.05 / h)

    # ---- (H) tracks by length ----
    ax = fig.add_axes(rect(XR[0], XR[1], top, h))
    edges = [0, 0.5, 1, 2, 3, 4, 6, 8, 12]
    cnt, _ = np.histogram(T.span_h, bins=edges)
    ax.bar(range(len(cnt)), cnt, color=["0.6"] * 3 + ["tab:green"] * (len(cnt) - 3), width=0.8)
    ax.set_xticks(range(len(cnt))); ax.set_xticklabels([f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])], fontsize=5)
    ax.set_xlabel("track span (h)"); ax.set_ylabel("tracks")
    for i, c in enumerate(cnt):
        ax.text(i, c + 1, str(c), ha="center", va="bottom", fontsize=5)
    page.label(ax, "H", "Tracks by span", y=1 + 0.05 / h)

    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
