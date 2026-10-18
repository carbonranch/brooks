"""Is temperature the controlling variable for rate and carrier, and is there a
date (season / growth) effect left once temperature is accounted for?

Inputs: the month's per-night window-animal files and hourly weather
(weather_hourly.csv, Open-Meteo ERA5 at the site). Every window of every
long-lived night-animal gets the temperature at its time.

1. Within-animal sensitivity: demean log R, log F and T within each
   animal-night, pool, regress -> %/degC for rate and carrier, the same slope
   the streaks in the (F, R) scatter should show if temperature drives both.
2. Nightly activity vs temperature: long-lived animals per recorder-night
   against the night's minimum temperature.
3. Temperature-corrected types: each animal-night's median (F, R) corrected
   to T_REF with the pooled slopes, then clustered; the correction should
   collapse the streaks into points.
4. Season: within each corrected type, animal-night medians regressed on
   temperature AND date together -> the date coefficient (%/day) is the
   effect that is not weather.

Usage:
    python analysis/callers/temperature_effects.py analysis/callers/runs/month
"""
from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

MIN_WINDOWS = 24
T_REF = 22.0
F_TOL, R_TOL = 0.04, 0.03
TZ = os.environ.get("BROOKS_TZ", "America/Chicago")


def load_windows(root: str) -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "BARLT_*"))):
        m = re.match(r"(BARLT_\d+)_(\w+)_(\d{8})$", os.path.basename(d))
        p = os.path.join(d, "night_night_window_animals.csv")
        if not m or not os.path.exists(p):
            continue
        W = pd.read_csv(p)
        n = W.groupby("track").t.transform("size")
        W = W[(n >= MIN_WINDOWS) & W.F_fix.notna()].copy()
        W["recorder"], W["site"], W["night"] = m.group(1), m.group(2), pd.Timestamp(m.group(3))
        W["an"] = W.recorder + "_" + W.night.dt.strftime("%m%d") + "_" + W.track.astype(str)
        rows.append(W)
    A = pd.concat(rows, ignore_index=True)
    A["t"] = pd.to_datetime(A.t, utc=True).dt.tz_convert(TZ)
    return A


def ols(X: np.ndarray, y: np.ndarray):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, res, rank, _ = np.linalg.lstsq(X1, y, rcond=None)
    dof = max(len(y) - X1.shape[1], 1)
    s2 = ((y - X1 @ beta) ** 2).sum() / dof
    cov = s2 * np.linalg.pinv(X1.T @ X1)
    return beta, np.sqrt(np.diag(cov))


def main() -> None:
    root = sys.argv[1]
    A = load_windows(root)
    Wx = pd.read_csv(os.path.join(root, "weather_hourly.csv"))
    Wx["t"] = pd.to_datetime(Wx.t, utc=True).dt.tz_convert(TZ)
    Wx = Wx.dropna(subset=["t"]).sort_values("t")
    A["T"] = np.interp(A.t.astype("int64"), Wx.t.astype("int64"), Wx.T_C)
    A["lR"], A["lF"] = np.log(A.R), np.log(A.F_fix)
    print(f"{len(A)} windows of {A.an.nunique()} long-lived animal-nights, {A.night.nunique()} nights, T {A['T'].min():.1f}-{A['T'].max():.1f} C")

    # 1. within-animal sensitivity, pooled
    g = A.groupby("an")
    dT = A["T"] - g["T"].transform("mean")
    dR = A.lR - g.lR.transform("mean")
    dF = A.lF - g.lF.transform("mean")
    ok = np.abs(dF) < 0.06
    bR, seR = ols(dT.to_numpy()[:, None], dR.to_numpy())
    bF, seF = ols(dT.to_numpy()[ok, None], dF.to_numpy()[ok])
    r2R = np.corrcoef(dT, dR)[0, 1] ** 2
    r2F = np.corrcoef(dT[ok], dF[ok])[0, 1] ** 2
    print(f"\nwithin animal-night, pooled over {len(A)} windows:")
    print(f"  rate:    {100 * bR[1]:+.2f} +/- {100 * seR[1]:.2f} %/degC  (R^2 {r2R:.2f}); Q10 = {np.exp(10 * bR[1]):.2f}")
    print(f"  carrier: {100 * bF[1]:+.2f} +/- {100 * seF[1]:.2f} %/degC  (R^2 {r2F:.2f}); ratio carrier/rate {bF[1] / bR[1]:.2f}")
    # per animal-night slopes for the distribution
    per = []
    for an, gg in g:
        if gg["T"].max() - gg["T"].min() < 1.5:
            continue
        b, _ = ols(gg["T"].to_numpy()[:, None], gg.lR.to_numpy())
        c, _ = ols(gg["T"].to_numpy()[:, None], gg.lF.to_numpy())
        per.append(dict(an=an, bR=100 * b[1], bF=100 * c[1], dT=gg["T"].max() - gg["T"].min(),
                        R=gg.R.median(), F=gg.F_fix.median(), site=gg.site.iloc[0], night=gg.night.iloc[0]))
    per = pd.DataFrame(per)
    print(f"  per animal-night (>= 1.5 C range, n={len(per)}): rate slope median {per.bR.median():+.2f} %/C "
          f"(IQR {per.bR.quantile(.25):+.2f}..{per.bR.quantile(.75):+.2f}), {(per.bR > 0).mean():.0%} positive; "
          f"carrier slope median {per.bF.median():+.2f} %/C, {(per.bF > 0).mean():.0%} positive")

    # 2. activity vs night temperature
    nightT = Wx[(Wx.t.dt.hour >= 19) | (Wx.t.dt.hour < 7)].copy()
    nightT["night"] = (nightT.t - pd.Timedelta(hours=12)).dt.tz_localize(None).dt.normalize()
    nT = nightT.groupby("night").agg(T_min=("T_C", "min"), T_mean=("T_C", "mean"), rain=("precip_mm", "sum"))
    pn = pd.read_csv(os.path.join(root, "month_per_night.csv"), parse_dates=["night"])
    act = pn.groupby("night").n_long.sum().to_frame().join(nT)
    rho = act[["n_long", "T_min"]].corr(method="spearman").iloc[0, 1]
    print(f"\nlong-lived animals per night (all recorders) vs night T_min: Spearman rho {rho:+.2f} over {len(act)} nights")

    # 3. temperature-corrected types
    M = g.agg(F=("F_fix", "median"), R=("R", "median"), T=("T", "mean"), day=("t", lambda s: s.min().dayofyear),
              site=("site", "first"), recorder=("recorder", "first"), night=("night", "first"), n=("t", "size"),
              depth=("depth", "median")).reset_index()
    M["R_ref"] = M.R * np.exp(-bR[1] * (M["T"] - T_REF))
    M["F_ref"] = M.F * np.exp(-bF[1] * (M["T"] - T_REF))
    for tag, fcol, rcol in (("raw", "F", "R"), ("corrected", "F_ref", "R_ref")):
        X = np.column_stack([np.log(M[fcol]) / F_TOL, np.log(M[rcol]) / R_TOL])
        M["type_" + tag] = sch.fcluster(sch.linkage(X, "average"), t=1.0, criterion="distance")
        ty = M.groupby("type_" + tag).size()
        print(f"types ({tag}): {len(ty)} clusters, {(ty >= 10).sum()} with >= 10 animal-nights, "
              f"largest {ty.max()}, animal-nights in clusters >= 10: {ty[ty >= 10].sum()} of {len(M)}")

    # 4. season within corrected types: log R and log F on T and day
    rows = []
    for ty, gg in M.groupby("type_corrected"):
        if len(gg) < 12 or gg.day.max() - gg.day.min() < 14 or gg["T"].max() - gg["T"].min() < 3:
            continue
        X = np.column_stack([gg["T"], gg.day])
        b, se = ols(X, np.log(gg.R))
        c, sc = ols(X, np.log(gg.F))
        rows.append(dict(type=ty, n=len(gg), F=gg.F_ref.median(), R=gg.R_ref.median(), n_sites=gg.site.nunique(),
                         days=gg.day.max() - gg.day.min(), dT=gg["T"].max() - gg["T"].min(),
                         R_perC=100 * b[1], R_perC_se=100 * se[1], R_perday=100 * b[2], R_perday_se=100 * se[2],
                         F_perC=100 * c[1], F_perC_se=100 * sc[1], F_perday=100 * c[2], F_perday_se=100 * sc[2]))
    S = pd.DataFrame(rows).sort_values("n", ascending=False)
    pd.set_option("display.width", 250)
    print(f"\nseason with temperature controlled, per corrected type (>= 12 animal-nights, >= 14 days, >= 3 C):")
    print(S.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if len(S):
        zR, zF = S.R_perday / S.R_perday_se, S.F_perday / S.F_perday_se
        print(f"date term: rate {np.median(S.R_perday):+.2f} %/day median, {(np.abs(zR) > 3).sum()} of {len(S)} types |z|>3; "
              f"carrier {np.median(S.F_perday):+.2f} %/day median, {(np.abs(zF) > 3).sum()} of {len(S)} |z|>3")
    M.to_csv(os.path.join(root, "temperature_animal_nights.csv"), index=False, float_format="%.4f")
    S.to_csv(os.path.join(root, "temperature_season_by_type.csv"), index=False, float_format="%.4f")

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    ax = axes[0, 0]
    sel = np.random.default_rng(0).choice(len(A), min(20000, len(A)), replace=False)
    ax.scatter(dT.iloc[sel], 100 * dR.iloc[sel], s=3, alpha=0.2, color="k")
    xx = np.array([dT.min(), dT.max()])
    ax.plot(xx, 100 * bR[1] * xx, "r-", lw=2)
    ax.set_xlabel("temperature - animal-night mean (degC)")
    ax.set_ylabel("rate - animal-night mean (%)")
    ax.set_title(f"within animal-night: rate {100 * bR[1]:+.2f} %/degC, R^2 {r2R:.2f}, Q10 {np.exp(10 * bR[1]):.2f}")
    ax.set_ylim(-15, 15)
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.scatter(dT.iloc[sel], 100 * dF.iloc[sel], s=3, alpha=0.2, color="k")
    ax.plot(xx, 100 * bF[1] * xx, "r-", lw=2)
    ax.set_xlabel("temperature - animal-night mean (degC)")
    ax.set_ylabel("carrier - animal-night mean (%)")
    ax.set_title(f"within animal-night: carrier {100 * bF[1]:+.2f} %/degC, R^2 {r2F:.2f}")
    ax.set_ylim(-6, 6)
    ax.grid(alpha=0.3)
    ax = axes[0, 2]
    ax.scatter(act.T_min, act.n_long, c=act.rain, cmap="Blues", edgecolors="k", s=50)
    for d_, r_ in act.iterrows():
        ax.annotate(f"{d_:%m-%d}", (r_.T_min, r_.n_long), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("night minimum temperature (degC)")
    ax.set_ylabel("long-lived animals, all recorders")
    ax.set_title(f"activity vs night temperature (Spearman {rho:+.2f}); colour = rain mm")
    ax.grid(alpha=0.3)
    for k, (tag, fcol, rcol, ttl) in enumerate((("raw", "F", "R", "as measured"), ("corrected", "F_ref", "R_ref", f"corrected to {T_REF:.0f} degC"))):
        ax = axes[1, k]
        sc_ = ax.scatter(M[rcol], M[fcol], c=M["T"], cmap="coolwarm", s=8, alpha=0.7, edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rate (Hz)")
        ax.set_ylabel("carrier (Hz)")
        ty = M.groupby("type_" + tag).size()
        ax.set_title(f"animal-night medians, {ttl}; colour = mean T; {(ty >= 10).sum()} types with >= 10")
        ax.grid(alpha=0.3, which="both")
        plt.colorbar(sc_, ax=ax, fraction=0.03, label="degC")
    ax = axes[1, 2]
    if len(S):
        ax.errorbar(S.F, S.R_perday, yerr=S.R_perday_se, fmt="o", color="b", ecolor="b", alpha=0.7, label="rate %/day")
        ax.errorbar(S.F * 1.03, S.F_perday, yerr=S.F_perday_se, fmt="s", color="r", ecolor="r", alpha=0.7, label="carrier %/day")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("type carrier (Hz)")
        ax.set_ylabel("date coefficient with temperature controlled (%/day)")
        ax.set_title("season effect per type (animal-night medians on T and day)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(root, "temperature_effects.png")
    fig.savefig(out, dpi=100)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
