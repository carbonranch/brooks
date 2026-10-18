"""One site by hour of day across a year: carrier x rate of the window-level
detections that belong to long-lived tracks (>= MIN_WIN windows), one panel
per local hour, coloured by grid temperature; and an hour x month heat map of
long-lived animals per window per recorder.

Usage:
    python analysis/callers/hour_of_day_plots.py analysis/callers/runs/y2025 --site Back9 --year 2025
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TZ = "America/Chicago"
MIN_WIN = 24
MAX_POINTS = 6000     # per panel, for the pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--site", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--by", choices=["month", "week"], default="month", help="column resolution of the heat map")
    ap.add_argument("--no-panels", action="store_true", help="skip the 24-panel figure")
    args = ap.parse_args()
    Wx = pd.read_csv(os.path.join(args.root, "weather_hourly_hrrr.csv"))
    Wx["t"] = pd.to_datetime(Wx.t, utc=True)
    Wx = Wx.dropna(subset=["T_C"]).sort_values("t").drop_duplicates("t")
    wt, wT = Wx.t.astype("int64").to_numpy(), Wx.T_C.to_numpy()
    rows, nwin = [], []
    for d in sorted(glob.glob(os.path.join(args.root, f"BARLT_*_{args.site}_{args.year}*_*"))):
        m = re.match(r"(BARLT_\d+)_(\w+)_(\d{8})_(night|day)$", os.path.basename(d))
        if not m:
            continue
        tag = m.group(4)
        p = os.path.join(d, f"night_{tag}_window_animals.csv")
        if not os.path.exists(p):
            continue
        WA = pd.read_csv(p, usecols=["t", "track", "R", "F_fix", "depth"])
        if WA.empty:
            continue
        WA["t"] = pd.to_datetime(WA.t, utc=True)
        # every window that had audio: count from the per-window animal files
        wins = glob.glob(os.path.join(d, f"*_{tag}_animals.csv"))
        for w in wins:
            hm = re.search(r"_(\d{8})_(\d{4})_", os.path.basename(w))
            nwin.append(dict(recorder=m.group(1), date=m.group(3), hour=int(hm.group(2)[:2])))
        cnt = WA.groupby("track").size()
        WA = WA[WA.track.isin(cnt[cnt >= MIN_WIN].index) & WA.F_fix.notna()]
        WA["recorder"], WA["date"] = m.group(1), pd.Timestamp(m.group(3))
        rows.append(WA)
    A = pd.concat(rows, ignore_index=True)
    A["T"] = np.interp(A.t.astype("int64").to_numpy(), wt, wT)
    A["tl"] = A.t.dt.tz_convert(TZ)
    A["hour"] = A.tl.dt.hour
    A["month"] = A.tl.dt.month
    A["week"] = A.tl.dt.isocalendar().week.astype(int)
    NW = pd.DataFrame(nwin)
    NW["month"] = pd.to_datetime(NW.date).dt.month
    NW["week"] = pd.to_datetime(NW.date).dt.isocalendar().week.astype(int)
    key = args.by
    print(f"{args.site} {args.year}: {len(A)} window-animals in long tracks, {A.recorder.nunique()} recorders, {len(NW)} windows with audio")

    # ---- 24 panels
    out = os.path.join(args.root, f"{args.site}_{args.year}_by_hour")
    order = list(range(12, 24)) + list(range(0, 12))          # noon -> noon so the night is contiguous
    if not args.no_panels:
      vmin, vmax = np.nanpercentile(A["T"], [2, 98])
      fig, axes = plt.subplots(4, 6, figsize=(30, 18), sharex=True, sharey=True)
      sc = None
      for h, ax in zip(order, axes.ravel()):
          g = A[A.hour == h]
          n_an = g.groupby(["recorder", "date", "track"]).ngroups
          if len(g) > MAX_POINTS:
              g = g.sample(MAX_POINTS, random_state=0)
          ax.scatter(A.R, A.F_fix, s=0.3, color="0.9", edgecolors="none", zorder=1, rasterized=True)
          if len(g):
              sc = ax.scatter(g.R, g.F_fix, s=1.2 + 5 * g.depth, c=g["T"], cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.7, edgecolors="none", zorder=2, rasterized=True)
          nw_h = (NW.hour == h).sum()
          ax.set_title(f"{h:02d}:00–{h:02d}:59  {n_an} animals ({len(A[A.hour == h])} window-detections) in {nw_h} windows"
                       + (f"; T {A[A.hour == h]['T'].quantile(.1):.0f}–{A[A.hour == h]['T'].quantile(.9):.0f} °C" if len(A[A.hour == h]) else ""), fontsize=9, loc="left")
          ax.set_xscale("log")
          ax.set_yscale("log")
          ax.set_xlim(7, 110)
          ax.set_ylim(1500, 23000)
          ax.grid(alpha=0.25, which="both", lw=0.4)
      for ax in axes[-1, :]:
          ax.set_xlabel("rate (Hz)")
      for ax in axes[:, 0]:
          ax.set_ylabel("carrier (Hz)")
      fig.suptitle(f"{args.site}, {args.year}, by hour of day (local): window-level detections of long-lived animals in carrier × rate; colour = HRRR temperature; grey = all hours", fontsize=14)
      fig.tight_layout(rect=(0, 0, 0.97, 0.965))
      cax = fig.add_axes([0.975, 0.2, 0.007, 0.6])
      fig.colorbar(sc, cax=cax, label="temperature (°C)")
      fig.savefig(out + ".pdf", dpi=120)
      fig.savefig(out + ".png", dpi=80)

    # ---- heat map hour x month: long-lived animals per window (per recorder)
    per = A.groupby([key, "hour", "recorder", "date", "t"]).track.nunique().rename("n").reset_index()
    # windows with no animals count as zero: scale by coverage
    cov = NW.groupby([key, "hour"]).size().unstack(key).reindex(index=order)
    tot = per.groupby([key, "hour"]).n.sum().unstack(key).reindex(index=order)
    H0 = (tot / cov).fillna(0)
    H0 = H0[[c for c in H0.columns if cov[c].sum() >= 24]]   # drop columns with almost no audio
    fig2, ax2 = plt.subplots(figsize=(12 if key == "month" else 20, 8))
    im = ax2.imshow(H0.values, aspect="auto", cmap="magma", origin="upper")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([f"{h:02d}:00" for h in order])
    ax2.set_xticks(range(len(H0.columns)))
    if key == "month":
        ax2.set_xticklabels(["JFMAMJJASOND"[m - 1] for m in H0.columns])
        ax2.set_xlabel(f"month of {args.year}")
    else:
        lab = [pd.Timestamp.fromisocalendar(args.year, int(w), 1).strftime("%d %b") if (i % 2 == 0) else "" for i, w in enumerate(H0.columns)]
        ax2.set_xticklabels(lab, rotation=90, fontsize=8)
        ax2.set_xlabel(f"ISO week of {args.year} (Monday's date)")
    ax2.set_ylabel("hour of day (local), noon to noon")
    ax2.set_title(f"{args.site} {args.year}: long-lived animals per 300 s window (mean over all windows with audio, per recorder)", loc="left", fontsize=11)
    fig2.colorbar(im, ax=ax2, label="animals per window")
    fig2.tight_layout()
    fig2.savefig(out + f"_heatmap_{key}.png", dpi=120)
    fig2.savefig(out + f"_heatmap_{key}.pdf")
    H0.round(2).to_csv(out + f"_heatmap_{key}.csv")
    print(f"wrote {out}_heatmap_{key}.png / .pdf / .csv")
    pd.set_option("display.width", 200)
    print(H0.round(1).to_string())


if __name__ == "__main__":
    main()
