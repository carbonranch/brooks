"""Year-scale views of the call types (streaks) for the paper, from the tables the
streak chain and year_hours_figure wrote. Counts are singer-hours or occupancy
(fraction of recorder-periods), never individuals.

  A. Line stability across seasons   year_line_stability.{csv,png}
     For each large streak, the monthly median of its singer-periods' position
     at 25 C, as % deviation of carrier (F25) and rate (R25) from the streak's
     year median, with the within-month IQR. A species that sings in spring
     and autumn must sit on the same line both times.
  B. Thermal limits per type         year_thermal_limits.{csv,png}
     The hourly HRRR temperature of every singer-hour of a streak: 5th / 50th /
     95th percentiles and the coldest hour, plus the nightly minimum of the
     nights it sang. Streaks ordered by their cold limit.
  C. Phenogram                       year_phenogram.png
     Singer-hours per recorder-night by week, rows ordered by peak week.
  D. Day and night per type          year_day_night.{csv,png}
     Day share of singer-periods per streak and the day/night temperature.
  E. Where each type sings           year_site_occupancy.{csv,png}
     Fraction of each site's recorder-nights with a singer-period of the
     streak (>= 2 h).

Usage:
    python analysis/callers/year_types.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 \
        --day-hours analysis/callers/runs/v2/year_figs/nov24_nov25_day_hours.csv
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
HF_SPLIT, C_LO, C_HI = 10000.0, 0.55, 0.25


def label(S, s):
    return f"{s}  {S.loc[s, 'F25'] / 1000:.2f} kHz @ {S.loc[s, 'R25']:.0f} Hz"


def unit_table(roots):
    rows = []
    for r in roots:
        for d in glob.glob(os.path.join(r, "BARLT_*")):
            m = re.match(r"(BARLT_\d+)_(.+)_(\d{8})_(night|day)$", os.path.basename(d))
            if m:
                rows.append(dict(recorder=m.group(1), site=m.group(2), night=pd.Timestamp(m.group(3)), period=m.group(4)))
    return pd.DataFrame(rows)


def line_stability(A, S, root, suffix, top_n=16):
    """Two temperature-free tests that a streak is the same line all year.
    across: offset perpendicular to the coupling line (log F - c log R), in %, which
    temperature cannot move if the coupling is right; along_res: position along the line
    minus the streak's own fit of position on HRRR temperature over the whole year, in %.
    (Bringing F and R to 25 C with the pooled 1.54 %/degC first and comparing R25 by month
    was tried and shows the correction slope, not the species: grid temperature attenuates
    the slope, so warm months came out 10 % high and cold months 10 % low in both axes.)"""
    big = S.sort_values("n", ascending=False).index[:top_n]
    A = A[A.streak.isin(big)].copy()
    A["month"] = pd.to_datetime(A.night).dt.to_period("M")
    rows = []; fits = {}
    for sid, g in A.groupby("streak"):
        c = C_LO if S.loc[sid, "F25"] < HF_SPLIT else C_HI
        lr, lf = np.log(g.R.to_numpy()), np.log(g.F.to_numpy())
        lr0, lf0 = np.median(lr), np.median(lf)
        along = ((lr - lr0) + c * (lf - lf0)) / np.hypot(1, c)
        across = ((lf - lf0) - c * (lr - lr0)) / np.hypot(1, c)
        T = g["T"].to_numpy()
        ok = np.isfinite(T)
        b, a = np.polyfit(T[ok], along[ok], 1)
        res = along - (a + b * T)
        fits[sid] = dict(slope_pct_per_C=100 * b, n=len(g), T_range=float(np.nanpercentile(T, 95) - np.nanpercentile(T, 5)))
        g = g.assign(across=100 * across, along_res=100 * res)
        for m, gm in g.groupby("month"):
            if len(gm) < 8:
                continue
            rows.append(dict(streak=sid, month=str(m), n=len(gm), across_med=gm.across.median(), across_q25=gm.across.quantile(.25), across_q75=gm.across.quantile(.75),
                             along_med=gm.along_res.median(), along_q25=gm.along_res.quantile(.25), along_q75=gm.along_res.quantile(.75), T_med=gm["T"].median()))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(root, f"year_line_stability{suffix}.csv"), index=False, float_format="%.3f")
    months = sorted(D.month.unique())
    xi = {m: i for i, m in enumerate(months)}
    ncol = 4; nrow = int(np.ceil(len(big) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, sid in zip(axes.ravel(), big):
        g = D[D.streak == sid]
        x = np.array([xi[m] for m in g.month])
        ax.axhline(0, color="k", lw=0.5)
        ax.axhspan(-3, 3, color="0.93", zorder=0)
        ax.errorbar(x - 0.12, g.across_med, yerr=[g.across_med - g.across_q25, g.across_q75 - g.across_med], fmt="o", ms=4, color="tab:blue", lw=1, capsize=2,
                    label="across the line (log F − c·log R)")
        ax.errorbar(x + 0.12, g.along_med, yerr=[g.along_med - g.along_q25, g.along_q75 - g.along_med], fmt="s", ms=4, color="tab:orange", lw=1, capsize=2,
                    label="along the line, minus the year's fit on T")
        for xx, n in zip(x, g.n):
            ax.text(xx, -14.5, str(n), ha="center", fontsize=5.5, color="0.4")
        ax.set_title(label(S, sid) + f"   (n {int(S.loc[sid, 'n'])}; {fits[sid]['slope_pct_per_C']:.2f} %/°C)", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(big):]:
        ax.set_axis_off()
    axes[0, 0].set_ylim(-15, 15)
    axes[0, 0].set_xticks(range(len(months))); axes[0, 0].set_xticklabels([pd.Period(m).strftime("%b\n%y") for m in months], fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("% offset", fontsize=7.5)
    axes[0, 0].legend(fontsize=6.5, loc="upper right", frameon=False)
    fig.suptitle("Is a streak the same line all year? Monthly median of each singer-period's offset across the coupling line (blue; temperature cannot move it) "
                 "and along it after the streak's own year-wide fit on temperature (orange); bars = within-month IQR, n below, gray band ±3 %", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(root, f"year_line_stability{suffix}.png"), dpi=140); fig.savefig(os.path.join(root, f"year_line_stability{suffix}.pdf")); plt.close(fig)
    summ = D.groupby("streak").agg(months=("month", "size"), max_abs_across=("across_med", lambda s: s.abs().max()), max_abs_along=("along_med", lambda s: s.abs().max()),
                                   T_range=("T_med", lambda s: s.max() - s.min()))
    summ["slope_pct_per_C"] = [fits[s]["slope_pct_per_C"] for s in summ.index]
    summ.to_csv(os.path.join(root, f"year_line_stability_summary{suffix}.csv"), float_format="%.3f")
    print("\nA. line stability: largest monthly median offset (%), and the streak's along-line slope on HRRR T (%/°C):")
    print(summ.round(2).to_string())
    return D


def thermal_limits(A, S, P, root, suffix, wx):
    rows = []
    nightly_min = wx
    for sid in S.index:
        g = P[P.streak == sid]
        if len(g) < 20:
            continue
        an = A[A.streak == sid]
        nm = pd.to_datetime(an.night).map(nightly_min)
        rows.append(dict(streak=sid, F25=S.loc[sid, "F25"], R25=S.loc[sid, "R25"], hours=len(g), nights=an.night.nunique(),
                         T_min=g["T"].min(), T_p05=g["T"].quantile(.05), T_p50=g["T"].median(), T_p95=g["T"].quantile(.95), T_max=g["T"].max(),
                         Tnightmin_p05=nm.quantile(.05), Tnightmin_min=nm.min(), Tnightmin_med=nm.median()))
    D = pd.DataFrame(rows).sort_values("T_p05")
    D.to_csv(os.path.join(root, f"year_thermal_limits{suffix}.csv"), index=False, float_format="%.2f")
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(D) + 1.6))
    y = np.arange(len(D))
    ax.barh(y, D.T_p95 - D.T_p05, left=D.T_p05, height=0.6, color="tab:orange", alpha=0.75, label="5–95 % of singer-hours")
    ax.scatter(D.T_p50, y, color="k", s=10, zorder=3, label="median")
    ax.scatter(D.T_min, y, marker="|", color="tab:blue", s=60, zorder=3, label="coldest hour heard")
    ax.set_yticks(y); ax.set_yticklabels([label(S, s) + f"  ({int(h)} h)" for s, h in zip(D.streak, D.hours)], fontsize=6.5)
    ax.set_xlabel("HRRR air temperature at the hour (°C)")
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Thermal range of each call type: HRRR temperature of its night singer-hours over the year,\nstreaks ordered by their cold limit (5th percentile)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"year_thermal_limits{suffix}.png"), dpi=140); fig.savefig(os.path.join(root, f"year_thermal_limits{suffix}.pdf")); plt.close(fig)
    print("\nB. thermal limits (°C): cold limit = 5th percentile of singer-hour temperature")
    print(D[["streak", "F25", "R25", "hours", "T_min", "T_p05", "T_p50", "T_p95", "Tnightmin_min", "Tnightmin_p05"]].round(1).to_string(index=False))
    return D


def phenogram(S, P, rp_week, weeks, root, suffix):
    P = P.copy()
    P["week"] = P.hour.dt.tz_localize(None).dt.to_period("W-SAT")
    H = P.groupby(["streak", "week"]).size().unstack("week").reindex(columns=weeks).fillna(0)
    per = H.div(rp_week.reindex(weeks).replace(0, np.nan), axis=1)
    peak = per.idxmax(axis=1)
    order = sorted(per.index, key=lambda s: (peak[s].start_time, -per.loc[s].sum()))
    M = per.loc[order].to_numpy()
    fig, ax = plt.subplots(figsize=(13, 0.22 * len(order) + 1.8))
    im = ax.imshow(np.where(M > 0, M, np.nan), aspect="auto", cmap="viridis", norm=mcolors.LogNorm(vmin=max(np.nanmin(M[M > 0]), 1e-3), vmax=np.nanmax(M)),
                   extent=[0, len(weeks), len(order), 0])
    for i, s in enumerate(order):
        w = per.loc[s]; nz = np.flatnonzero(w.to_numpy() > 0)
        ax.plot([nz.min(), nz.max() + 1], [i + 0.5, i + 0.5], color="w", lw=0.4, alpha=0.6)
        ax.plot([list(weeks).index(peak[s]) + 0.5], [i + 0.5], marker="v", color="w", ms=3)
    ax.set_yticks(np.arange(len(order)) + 0.5); ax.set_yticklabels([label(S, s) for s in order], fontsize=6.5)
    ticks = [i for i, w in enumerate(weeks) if w.start_time.day <= 7]
    ax.set_xticks(ticks); ax.set_xticklabels([weeks[i].start_time.strftime("%b %y") for i in ticks], fontsize=8)
    ax.set_facecolor("0.92")
    ax.set_title("Phenogram: singer-hours per recorder-night by week, streaks ordered by peak week (white marker); gray = none", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("singer-hours per recorder-night", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"year_phenogram{suffix}.png"), dpi=140); fig.savefig(os.path.join(root, f"year_phenogram{suffix}.pdf")); plt.close(fig)


def day_night(A, Dd, S, root, suffix, min_n=40):
    rows = []
    for sid in S.index:
        gn, gd = A[A.streak == sid], Dd[Dd.streak == sid]
        if len(gn) + len(gd) < min_n:
            continue
        rows.append(dict(streak=sid, F25=S.loc[sid, "F25"], R25=S.loc[sid, "R25"], n_night=len(gn), n_day=len(gd),
                         day_share=len(gd) / (len(gn) + len(gd)), T_night=gn["T"].median(), T_day=gd["T"].median() if len(gd) else np.nan,
                         day_months=",".join(str(m) for m in sorted(pd.to_datetime(gd.night).dt.month.unique())) if len(gd) else ""))
    D = pd.DataFrame(rows).sort_values("day_share", ascending=False)
    D.to_csv(os.path.join(root, f"year_day_night{suffix}.csv"), index=False, float_format="%.3f")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    sz = np.clip(np.sqrt(D.n_night + D.n_day) * 1.2, 8, 80)
    sc = ax1.scatter(D.F25 / 1000, D.day_share, s=sz, c=D.R25, cmap="plasma", norm=mcolors.LogNorm(vmin=8, vmax=100), edgecolors="k", linewidths=0.4)
    for r in D.itertuples():
        if r.day_share > 0.2 or r.n_night + r.n_day > 1500:
            ax1.annotate(str(r.streak), (r.F25 / 1000, r.day_share), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax1.set_xscale("log"); ax1.set_xlabel("carrier at 25 °C (kHz)"); ax1.set_ylabel("day share of singer-periods (≥ 2 h)")
    ax1.set_title("Which types sing by day: share of each streak's singer-periods that are daytime (size = periods, color = rate)", fontsize=8.5, loc="left")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.02); cb.set_label("rate at 25 °C (Hz)", fontsize=8)
    ax1.grid(alpha=0.3, which="both", lw=0.4)
    ok = D.T_day.notna()
    ax2.scatter(D.T_night[ok], D.T_day[ok], s=sz[ok], c="tab:orange", edgecolors="k", linewidths=0.4)
    lim = [5, 35]; ax2.plot(lim, lim, color="k", lw=0.6); ax2.set_xlim(lim); ax2.set_ylim(lim)
    for r in D[ok].itertuples():
        ax2.annotate(str(r.streak), (r.T_night, r.T_day), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel("median temperature of night singer-periods (°C)"); ax2.set_ylabel("median temperature of day singer-periods (°C)")
    ax2.set_title("Day singing happens warmer: per streak, median HRRR temperature of its day vs night periods", fontsize=8.5, loc="left")
    ax2.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"year_day_night{suffix}.png"), dpi=140); fig.savefig(os.path.join(root, f"year_day_night{suffix}.pdf")); plt.close(fig)
    print("\nD. day and night per streak:")
    print(D.round(2).to_string(index=False))
    return D


def site_occupancy(A, S, U, root, suffix):
    nights = U[U.period == "night"]
    rn_site = nights.groupby("site").size()
    A = A.copy(); A["night"] = pd.to_datetime(A.night)
    occ = A.groupby(["streak", "site"]).apply(lambda g: g[["recorder", "night"]].drop_duplicates().shape[0]).unstack("site").fillna(0)
    occ = occ.div(rn_site.reindex(occ.columns), axis=1)
    order = S.loc[S.index.isin(occ.index)].sort_values("F25", ascending=False).index
    sites = list(rn_site.sort_values(ascending=False).index)
    occ = occ.loc[order, [s for s in sites if s in occ.columns]]
    occ.to_csv(os.path.join(root, f"year_site_occupancy{suffix}.csv"), float_format="%.4f")
    fig, ax = plt.subplots(figsize=(0.75 * len(occ.columns) + 4, 0.22 * len(order) + 1.6))
    M = occ.to_numpy()
    im = ax.imshow(np.where(M > 0, M, np.nan), aspect="auto", cmap="magma", norm=mcolors.LogNorm(vmin=1e-3, vmax=max(0.05, np.nanmax(M))))
    ax.set_yticks(range(len(order))); ax.set_yticklabels([label(S, s) for s in order], fontsize=6.5)
    ax.set_xticks(range(len(occ.columns))); ax.set_xticklabels([f"{c}\n({rn_site[c]} rec-nights)" for c in occ.columns], fontsize=7)
    ax.set_facecolor("0.92")
    ax.set_title("Where each type sings: fraction of a site's recorder-nights with a singer-period (≥ 2 h) of the streak; gray = never", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("occupancy (fraction of recorder-nights)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"year_site_occupancy{suffix}.png"), dpi=140); fig.savefig(os.path.join(root, f"year_site_occupancy{suffix}.pdf")); plt.close(fig)
    print("\nE. site occupancy written")
    return occ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--suffix", default="_v6")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--day-hours", default=None)
    ap.add_argument("--first", default="2024-11-01")
    ap.add_argument("--last", default="2025-11-30")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[S.harmonic_of < 0] if "harmonic_of" in S.columns else S
    A = A[A.streak.isin(S.index)]
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv")).drop(columns=["streak"], errors="ignore").merge(A[["an", "streak"]], on="an")
    P["hour"] = pd.to_datetime(P.hour, utc=True).dt.tz_convert(TZ)
    W = pd.read_csv(os.path.join(root, "weather_hourly_hrrr.csv"))
    W["t"] = pd.to_datetime(W.t, utc=True).dt.tz_convert(TZ)
    Wn = W[(W.t.dt.hour >= 19) | (W.t.dt.hour < 7)].copy()
    Wn["night"] = (Wn.t - pd.Timedelta(hours=12)).dt.tz_localize(None).dt.normalize()
    nightly_min = Wn.groupby("night").T_C.min()
    U = unit_table(args.roots)
    weeks = pd.period_range(pd.Timestamp(args.first), pd.Timestamp(args.last), freq="W-SAT")
    un = U[U.period == "night"]
    rp_week = un.groupby(un.night.dt.to_period("W-SAT")).size()

    line_stability(A, S, root, args.suffix)
    thermal_limits(A, S, P, root, args.suffix, nightly_min)
    phenogram(S, P, rp_week, weeks, root, args.suffix)
    Dd = pd.read_csv(os.path.join(root, f"animal_days_streaks{args.suffix}.csv")) if os.path.exists(os.path.join(root, f"animal_days_streaks{args.suffix}.csv")) else pd.DataFrame(columns=A.columns)
    day_night(A, Dd, S, root, args.suffix)
    site_occupancy(A, S, U, root, args.suffix)


if __name__ == "__main__":
    main()
