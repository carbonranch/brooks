"""Exploratory: the type mix by week at a few sites, one panel per site on one color key (MBE, 2026-10-13: House,
Back9, NorthSide). Singers tracked at once per recorded night-hour by week, stacked by type; trill types from the v3
year tables, chirp types from the chirp plane (<year>/chirp/), both at >= 40 singer-nights; the most active types over
the chosen sites are colored and named by paper number and 25 C position, the rest are gray. Weeks with under a
quarter of the site's usual recording are hatched.

Usage:
  python analysis/callers/explore_type_mix_sites.py --sites House Back9 NorthSide \
      --out analysis/callers/runs/figures_paper/explore/type_mix_three_sites
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ids  # noqa: E402

TZ = "America/Chicago"
OFF = 100000  # chirp-plane streak ids are offset so they never collide with the trill catalog's


def load(R: str, min_n: int):
    A = pd.read_csv(f"{R}/animal_nights_streaks_v9.csv"); S = pd.read_csv(f"{R}/streaks_coarse_v9.csv", index_col=0); S = S[S.n >= min_n]
    P = pd.read_csv(f"{R}/streak_hourly_points_all.csv").drop(columns=["streak"]).merge(A[["an", "streak"]], on="an", how="inner")
    Ac = pd.read_csv(f"{R}/chirp/animal_nights_streaks_c2.csv"); Sc = pd.read_csv(f"{R}/chirp/streaks_coarse_c2.csv", index_col=0); Sc = Sc[Sc.n >= min_n]
    Pc = pd.read_csv(f"{R}/chirp/streak_hourly_points_all.csv").drop(columns=["streak"], errors="ignore").merge(Ac[["an", "streak"]], on="an", how="inner")
    cid = pd.read_csv(f"{R}/chirp/paper_ids_chirp.csv").set_index("streak").paper_id
    Pc["streak"] = Pc.streak + OFF; Sc.index = Sc.index + OFF
    ids = paper_ids.load(R, "_v9"); ids.update({int(k) + OFF: v for k, v in cid.items()})
    P = pd.concat([P[P.streak.isin(S.index)], Pc[Pc.streak.isin(Sc.index)]], ignore_index=True)
    P["hour"] = pd.to_datetime(P.hour, utc=True).dt.tz_convert(TZ); P["week"] = P.hour.dt.tz_localize(None).dt.to_period("W-SAT")
    return P, pd.concat([S, Sc]), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="analysis/callers/runs/v3/year_nov24_nov25"); ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--first", default="2024-11-01"); ap.add_argument("--last", default="2025-11-30"); ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--min-n", type=int, default=40); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P, S, ids = load(args.root, args.min_n)
    U = pd.read_csv(f"{args.root}/recorded_windows.csv", parse_dates=["night"]); U["hours"] = U.windows * 300 / 3600; U["week"] = U.night.dt.to_period("W-SAT")
    first, last = pd.Timestamp(args.first), pd.Timestamp(args.last); U = U[(U.night >= first) & (U.night <= last)]
    weeks = pd.period_range(first, last, freq="W-SAT"); x = np.array([w.start_time + pd.Timedelta(days=3.5) for w in weeks])
    sub = P[P.site.isin(args.sites)]; top = list(sub.groupby("streak").size().sort_values(ascending=False).index[:args.top])
    cmap = plt.get_cmap("tab20"); col = {s: cmap(i % 20) for i, s in enumerate(top)}
    lab = {s: f"{ids.get(s, s)}  {S.loc[s, 'F25'] / 1000:.1f} kHz @ {S.loc[s, 'R25']:.0f} Hz" + (" chirps" if s >= OFF else "") for s in top}
    fig, axes = plt.subplots(len(args.sites), 1, figsize=(13, 3.6 * len(args.sites) + 0.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, site in zip(axes, args.sites):
        rec = U[U.site == site].groupby("week").hours.sum().reindex(weeks).fillna(0)
        H = P[P.site == site].groupby(["streak", "week"]).size().unstack("week").reindex(columns=weeks).fillna(0)
        per = H.div(rec.replace(0, np.nan), axis=1)
        stack = [per.loc[s].fillna(0).to_numpy() if s in per.index else np.zeros(len(weeks)) for s in top] + [per.loc[~per.index.isin(top)].sum(axis=0).fillna(0).to_numpy()]
        ax.stackplot(x, *stack, colors=[col[s] for s in top] + [(0.82, 0.82, 0.82, 1)], linewidth=0)
        thin = rec < 0.25 * rec[rec > 0].median()
        for w, t in zip(weeks, thin):
            if t:
                ax.axvspan(w.start_time, w.end_time, color="white", alpha=0.6, lw=0, hatch="///", zorder=5)
        ax.set_ylabel("singers tracked at once"); ax.grid(alpha=.25)
        n_rec = U[U.site == site].recorder.nunique()
        ax.set_title(f"{site}: {n_rec} recorders, {rec.sum():,.0f} recorded night-hours, {int(H.values.sum()):,} singer-hours", loc="left", fontsize=10, fontweight="bold")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[s]) for s in top] + [plt.Rectangle((0, 0), 1, 1, color=(0.82, 0.82, 0.82))],
                   labels=[lab[s] for s in top] + ["other types"], fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator()); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle(f"Type mix by week by site (v3; the {args.top} most active types over these sites named; hatched = weeks with under a quarter of the site's usual recording)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(args.out + ".png", dpi=110); print("wrote", args.out)


if __name__ == "__main__":
    main()
