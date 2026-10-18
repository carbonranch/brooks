"""The mix of call types over the year: singer-hours per streak per week.

MBE (2026-09-26): be conservative about counts of individuals; the primary
output is how the mix of call types changes over time. A streak (one line in
the temperature-corrected carrier x rate plane, built by build_streaks ->
split_streaks_by_T -> merge_streaks) is the call type. Here every singer-hour
of the year is assigned its singer-period's streak and summed per streak per
week; the week's total is divided by the number of recorder-periods recorded
that week (unit directories in the run roots), so weeks with fewer recorders
are not read as quieter.

Writes, into <root>:
  type_mix<suffix>.csv          streak x week: singer-hours, recorder-periods, hours per recorder-period
  type_mix_streaks<suffix>.csv  per streak: F25, R25, total hours, first/last/peak week,
                                chirp share (fraction of its singer-hours that carried a chirp rate),
                                median chirp rate
  type_mix<suffix>.png/.pdf     top: singer-hours per recorder-period stacked by streak, with the
                                weekly mean nightly minimum temperature (dashed); middle: each
                                week's composition, faded where there is little activity (a share
                                plot alone made the winter look like a wall, MBE 2026-10-03);
                                bottom: streak x week heat map (hours per recorder-period, log
                                colour), streaks ordered by carrier

Usage:
    python analysis/callers/type_mix.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --period night
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TZ = "America/Chicago"
TOP_N = 14


def recorder_periods_by_week(roots, period, first, last) -> pd.Series:
    dates = []
    for r in roots:
        for d in glob.glob(os.path.join(r, f"BARLT_*_{period}")):
            m = re.search(r"_(\d{8})_(night|day)$", os.path.basename(d))
            if m:
                dates.append(pd.Timestamp(m.group(1)))
    s = pd.Series(1, index=pd.DatetimeIndex(dates))
    s = s[(s.index >= first) & (s.index <= last)]
    return s.groupby(s.index.to_period("W-SAT")).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--suffix", default="_v6")
    ap.add_argument("--roots", nargs="+", required=True, help="run roots whose unit directories count the recorder-periods per week")
    ap.add_argument("--period", default="night", choices=["night", "day"])
    ap.add_argument("--first", default="2024-11-01")
    ap.add_argument("--last", default="2025-11-30")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--out", default=None)
    ap.add_argument("--weather", default=None, help="hourly HRRR table; default <root>/weather_hourly_hrrr.csv")
    args = ap.parse_args()
    root = args.root
    if args.period == "night":
        A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    else:
        A = pd.read_csv(os.path.join(root, f"animal_days_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[S.harmonic_of < 0] if "harmonic_of" in S.columns else S
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"))
    if args.period == "day":
        # the hourly points file holds nights only; day hours come from the year_figs table the day assignment used
        raise SystemExit("day periods: not yet wired (needs the day hours table); run with --period night")
    P = P.drop(columns=["streak"], errors="ignore").merge(A[["an", "streak"]], on="an", how="inner")
    P["hour"] = pd.to_datetime(P.hour, utc=True).dt.tz_convert(TZ)
    P["week"] = P.hour.dt.tz_localize(None).dt.to_period("W-SAT")
    P = P[P.streak.isin(S.index)]
    if "rc_frac" not in P.columns:
        P["rc_frac"] = 0.0; P["Rc"] = np.nan
    rp = recorder_periods_by_week(args.roots, args.period, pd.Timestamp(args.first), pd.Timestamp(args.last))
    weeks = pd.period_range(pd.Timestamp(args.first), pd.Timestamp(args.last), freq="W-SAT")
    H = P.groupby(["streak", "week"]).size().unstack("week").reindex(columns=weeks).fillna(0)
    H.columns.name = "week"          # reindex drops the axis name
    rp = rp.reindex(weeks).fillna(0)
    per_rp = H.div(rp.replace(0, np.nan), axis=1)
    long = H.stack().rename("hours").reset_index()
    long["recorder_periods"] = long.week.map(rp)
    long["hours_per_rp"] = long.hours / long.recorder_periods.replace(0, np.nan)
    long["week"] = long.week.dt.start_time.dt.strftime("%Y-%m-%d")
    out = args.out or os.path.join(root, f"type_mix{args.suffix}")
    long.to_csv(out + ".csv", index=False, float_format="%.3f")
    # per streak
    rows = []
    for sid, g in P.groupby("streak"):
        wk = g.groupby("week").size()
        rows.append(dict(streak=sid, F25=S.loc[sid, "F25"], R25=S.loc[sid, "R25"], hours=len(g), singer_periods=g.an.nunique(),
                         first_week=str(wk.index.min().start_time.date()), peak_week=str(wk.idxmax().start_time.date()), last_week=str(wk.index.max().start_time.date()),
                         n_weeks=int((wk > 0).sum()), chirp_share=round(float(g.rc_frac.mean()), 3), Rc_med=round(float(g.Rc.median()), 2) if g.Rc.notna().any() else np.nan,
                         T_med=round(float(g["T"].median()), 1)))
    ST = pd.DataFrame(rows).sort_values("hours", ascending=False)
    ST.to_csv(out + "_streaks.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(f"{len(P):,} singer-hours in {len(ST)} streaks over {len(weeks)} weeks; recorder-{args.period}s per week median {rp.median():.0f}")
    print(ST.head(40).to_string(index=False))

    # weekly mean of the nightly minimum temperature (HRRR), for the activity panel
    wx = None
    wpath = args.weather or os.path.join(root, "weather_hourly_hrrr.csv")
    if os.path.exists(wpath):
        W = pd.read_csv(wpath)
        W["t"] = pd.to_datetime(W.t, utc=True).dt.tz_convert(TZ)
        W = W[(W.t.dt.hour >= 19) | (W.t.dt.hour < 7)] if args.period == "night" else W[(W.t.dt.hour >= 7) & (W.t.dt.hour < 19)]
        W["night"] = (W.t - pd.Timedelta(hours=12)).dt.tz_localize(None).dt.normalize()
        tn = W.groupby("night").T_C.min() if args.period == "night" else W.groupby("night").T_C.max()
        wx = tn.groupby(tn.index.to_period("W-SAT")).mean().reindex(weeks)

    # figure: (1) absolute activity stacked by streak, so winter tapers instead of hitting a wall;
    # (2) composition as a strip, faded where the week has too little activity to have one;
    # (3) streak x week heat map
    top = list(ST.streak.head(args.top))
    x = np.array([w.start_time + pd.Timedelta(days=3.5) for w in weeks])     # week centres
    cmap = plt.get_cmap("tab20")
    labels = [f"{s}: {S.loc[s, 'F25'] / 1000:.2f} kHz @ {S.loc[s, 'R25']:.0f} Hz" for s in top] + ["other streaks"]
    colors = [cmap(i % 20) for i in range(len(top))] + ["0.8"]
    abs_stack = [per_rp.loc[s].fillna(0).to_numpy() for s in top] + [per_rp.loc[~per_rp.index.isin(top)].sum(axis=0).fillna(0).to_numpy()]
    tot = per_rp.sum(axis=0).fillna(0)
    share = H.div(H.sum(axis=0).replace(0, np.nan), axis=1)
    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(13, 12.5), gridspec_kw=dict(height_ratios=[1.15, 0.45, 1.6]))
    ax1.stackplot(x, *abs_stack, labels=labels, colors=colors, linewidth=0)
    ax1.set_ylabel(f"singer-hours per recorder-{args.period} (stacked by streak)")
    ax1.set_xlim(x[0], x[-1]); ax1.set_ylim(0, None)
    ax1.set_title(f"Call-type mix by week, {args.period}s, {pd.Timestamp(args.first):%b %Y} – {pd.Timestamp(args.last):%b %Y}: "
                  f"{len(P):,} singer-hours (singers tracked >= 2 h), {len(ST)} streaks; streak = one line in the temperature-corrected carrier x rate plane",
                  loc="left", fontsize=9)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.16, 1.0), fontsize=7, frameon=False, ncol=2)   # the winter is empty: use it
    if wx is not None:
        ax1b = ax1.twinx()
        ax1b.plot(x, wx.to_numpy(), color="k", lw=1.2, ls="--")
        ax1b.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax1b.set_ylabel(f"weekly mean of the nightly {'minimum' if args.period == 'night' else 'maximum'} air temperature (°C, dashed)", fontsize=8)
    # composition strip: alpha rises with the week's activity (full at >= FULL h per recorder-period)
    FULL = 5.0      # singer-hours per recorder-period at which the composition is drawn at full strength
    alpha = np.clip(tot.to_numpy() / FULL, 0, 1)
    edges = np.array([w.start_time for w in weeks] + [weeks[-1].end_time])
    sh = [share.loc[s].fillna(0).to_numpy() for s in top] + [share.loc[~share.index.isin(top)].sum(axis=0).fillna(0).to_numpy()]
    bottom = np.zeros(len(weeks))
    for vals, col in zip(sh, colors):
        rgba = np.tile(mcolors.to_rgba(col), (len(weeks), 1)); rgba[:, 3] = alpha
        ax3.bar(edges[:-1], vals, width=np.diff(edges), bottom=bottom, align="edge", color=rgba, linewidth=0)
        bottom += vals
    ax3.set_xlim(x[0], x[-1]); ax3.set_ylim(0, 1); ax3.set_yticks([0, 0.5, 1])
    ax3.set_ylabel("share of the\nweek's hours", fontsize=8)
    ax3.set_title(f"composition of each week's singer-hours; faded in proportion to activity below {FULL:.1f} singer-hours per recorder-{args.period} (white = none)",
                  loc="left", fontsize=8.5)
    # heat map ordered by carrier
    order = S.loc[S.index.isin(per_rp.index)].sort_values("F25", ascending=False).index
    M = per_rp.loc[order].to_numpy()
    im = ax2.imshow(np.where(M > 0, M, np.nan), aspect="auto", cmap="viridis", norm=mcolors.LogNorm(vmin=max(np.nanmin(M[M > 0]), 1e-3), vmax=np.nanmax(M)),
                    extent=[0, len(weeks), len(order), 0])
    ax2.set_yticks(np.arange(len(order)) + 0.5)
    ax2.set_yticklabels([f"{s}  {S.loc[s, 'F25'] / 1000:.2f} kHz @ {S.loc[s, 'R25']:.0f} Hz" for s in order], fontsize=6.5)
    ticks = [i for i, w in enumerate(weeks) if w.start_time.day <= 7]
    ax2.set_xticks(ticks); ax2.set_xticklabels([weeks[i].start_time.strftime("%b %y") for i in ticks], fontsize=8)
    ax2.set_xlabel("week"); ax2.set_title(f"singer-hours per recorder-{args.period}, by streak (rows ordered by carrier) and week; grey = none", loc="left", fontsize=9)
    ax2.set_facecolor("0.92")
    cb = fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.01); cb.set_label(f"singer-hours per recorder-{args.period}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf"); plt.close(fig)
    print("wrote", out + ".csv/.png")


if __name__ == "__main__":
    main()
