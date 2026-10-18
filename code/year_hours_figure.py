"""Year-scale streak figures from singer-hours, coloured by hourly temperature.

Reads every unit directory (<root>/<recorder>_<site>_<yyyymmdd>_<night|day>)
of one or more run roots within a date range, keeps tracks with >= MIN_WIN
windows (2 h), attaches the HRRR hourly temperature (weather_hourly_hrrr.csv
in each root, concatenated), aggregates each track to one point per hour,
and draws:

  A. the whole range in one panel (all recorders, all months);
  B. a 12-panel month-by-month grid, all recorders combined;
  C. one 12-panel figure per recorder (into <out>_recorders/).

Usage:
    python analysis/callers/year_hours_figure.py analysis/callers/runs/y2024 analysis/callers/runs/y2025 \
        --first 2024-11-01 --last 2025-11-30 --out analysis/callers/runs/year_figs/nov24_nov25
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

from streak_story_figure import style

TZ = "America/Chicago"
MIN_WIN = 24
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_hours(roots, first, last, min_win=MIN_WIN, periods=("night", "day"), suffix="") -> pd.DataFrame:
    """suffix selects the linker's window-animals file: '' for the greedy night_animals output,
    '_global' for night_global's (night_<period>_global_window_animals.csv)."""
    wx = []
    for root in roots:
        p = os.path.join(root, "weather_hourly_hrrr.csv")
        if os.path.exists(p):
            W = pd.read_csv(p)
            W["t"] = pd.to_datetime(W.t, utc=True)
            wx.append(W.dropna(subset=["T_C"]))
    Wx = pd.concat(wx).sort_values("t").drop_duplicates("t")
    wt, wT = Wx.t.astype("int64").to_numpy(), Wx.T_C.to_numpy()
    lo, hi = pd.Timestamp(first), pd.Timestamp(last)
    rows = []
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*_????????_*"))):
            m = re.match(r"([A-Za-z0-9]+_\d+)_(\w+?)_(\d{8})_(night|day)$", os.path.basename(d))
            if not m or m.group(4) not in periods:
                continue
            date = pd.Timestamp(m.group(3))
            if date < lo or date > hi:
                continue
            f = os.path.join(d, f"night_{m.group(4)}{suffix}_window_animals.csv")
            if not os.path.exists(f):
                continue
            W = pd.read_csv(f)
            if W.empty or "F_fix" not in W:
                continue
            n = W.groupby("track").size()
            W = W[W.track.isin(n[n >= min_win].index) & W.F_fix.notna() & (W.R > 0)]
            if W.empty:
                continue
            W["t"] = pd.to_datetime(W.t, utc=True)
            W["recorder"], W["site"], W["date"], W["period"] = m.group(1), m.group(2), date, m.group(4)
            W["tid"] = os.path.basename(d) + ":" + W.track.astype(str)
            if "chirp_rate" not in W.columns:      # greedy output of the old detector: no chirp channel
                W["chirp_rate"] = np.nan
            rows.append(W[["t", "R", "F_fix", "depth", "chirp_rate", "recorder", "site", "date", "period", "tid"]])
    A = pd.concat(rows, ignore_index=True)
    A["T"] = np.interp(A.t.astype("int64").to_numpy(), wt, wT)
    A["hour"] = A.t.dt.floor("h")
    # Rc: the hour's median chirp rate (v2 chirpers carry R = syllable rate and chirp_rate);
    # rc_frac: the share of the hour's windows that had one -- the chirper variable for the type mix
    H = A.groupby(["tid", "hour"]).agg(R=("R", "median"), F=("F_fix", "median"), T=("T", "median"), depth=("depth", "median"),
                                       Rc=("chirp_rate", "median"), rc_frac=("chirp_rate", lambda s: float(s.notna().mean())),
                                       recorder=("recorder", "first"), site=("site", "first"), period=("period", "first")).reset_index()
    H["month"] = H.hour.dt.tz_convert(TZ).dt.month
    H["ym"] = H.hour.dt.tz_convert(TZ).dt.strftime("%Y-%m")
    return drop_artefacts(H)


def drop_artefacts(H: pd.DataFrame, path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artefacts.csv")) -> pd.DataFrame:
    """Remove singer-hours matching known non-biological sources listed in
    artefacts.csv (recorder, rate band, carrier band, date range, and an
    upper temperature bound, since the machines run when it is cold)."""
    if not os.path.exists(path):
        return H
    A = pd.read_csv(path)
    keep = np.ones(len(H), bool)
    hl = H.hour.dt.tz_convert(TZ).dt.tz_localize(None)
    for a in A.itertuples():
        m = ((H.recorder == a.recorder) & H.R.between(a.R_lo, a.R_hi) & H.F.between(a.F_lo, a.F_hi)
             & (hl >= pd.Timestamp(a.first)) & (hl < pd.Timestamp(a.last) + pd.Timedelta(days=1)) & (H["T"] <= a.T_max))
        keep &= ~m.to_numpy()
    n = int((~keep).sum())
    if n:
        print(f"dropped {n:,} singer-hours matching {len(A)} artefact rules (artefacts.csv)")
    return H[keep].copy()


def uptime_table(inventories, periods) -> pd.DataFrame:
    """Fraction of the month's hours (for the chosen periods) with a
    recording, per recorder x month, from inventory_<year>_files.csv (one row
    per hour-file: rec, date, hour, night)."""
    rows = []
    for p in inventories:
        if os.path.exists(p):
            rows.append(pd.read_csv(p, usecols=["rec", "date", "hour", "night"]))
    if not rows:
        return pd.DataFrame(columns=["recorder", "ym", "uptime"])
    F = pd.concat(rows, ignore_index=True)
    F["date"] = pd.to_datetime(F.date)
    want_night = "night" in periods
    want_day = "day" in periods
    F = F[(F.night & want_night) | (~F.night & want_day)]
    F["ym"] = F.date.dt.strftime("%Y-%m")
    have = F.groupby(["rec", "ym"]).size().rename("h").reset_index()
    frac = (12 if (want_night != want_day) else 24) / 24.0
    have["hours_in_month"] = have.ym.map(lambda s: pd.Period(s).days_in_month * 24 * frac)
    have["uptime"] = (have.h / have.hours_in_month).clip(upper=1.0)
    return have.rename(columns={"rec": "recorder"})[["recorder", "ym", "uptime"]]


def panel(ax, H, tmin, tmax, dot, alpha, cmap="jet"):
    H = H.sort_values("T")
    sc = ax.scatter(H.R, H.F, s=dot, c=H["T"], cmap=cmap, vmin=tmin, vmax=tmax, alpha=alpha, edgecolors="none")
    style(ax, "", "")
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dot", type=float, default=1.2)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--tmin", type=float, default=0)
    ap.add_argument("--tmax", type=float, default=35)
    ap.add_argument("--periods", default="night,day")
    ap.add_argument("--suffix", default="", help="window-animals file variant: '' (greedy) or _global (night_global)")
    ap.add_argument("--inventories", nargs="*", default=["analysis/callers/runs/inventory_2024_files.csv", "analysis/callers/runs/inventory_2025_files.csv"])
    args = ap.parse_args()
    periods = tuple(args.periods.split(","))
    H = load_hours(args.roots, args.first, args.last, periods=periods, suffix=args.suffix)
    U = uptime_table(args.inventories, periods)
    up = {(r.recorder, r.ym): r.uptime for r in U.itertuples()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    H.to_csv(args.out + "_hours.csv", index=False, float_format="%.3f")
    span = f"{pd.Timestamp(args.first):%b %Y} – {pd.Timestamp(args.last):%b %Y}"
    print(f"{len(H):,} singer-hours, {H.tid.nunique():,} singer-periods >= 2 h, {H.recorder.nunique()} recorders, {H.ym.nunique()} months")

    # A. whole range
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    sc = panel(ax, H, args.tmin, args.tmax, args.dot, args.alpha)
    ax.set_xlabel("syllable rate (Hz)"); ax.set_ylabel("carrier frequency (kHz)")
    ax.set_title(f"{len(H):,} singer-hours, {span}, {H.recorder.nunique()} recorders, {'+'.join(periods)};\n"
                 f"one point per singer per hour (singers tracked ≥ 2 h), coloured by the hour's HRRR air temperature", loc="left", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.solids.set_alpha(1.0); cb.set_label("HRRR air temperature at the hour (°C)")
    fig.tight_layout(); fig.savefig(args.out + "_all.pdf"); fig.savefig(args.out + "_all.png", dpi=150); plt.close(fig)

    # B. month grid, all recorders
    yms = sorted(H.ym.unique())
    def month_grid(Hs, title, path, rec=None):
        ncol = 4
        nrow = int(np.ceil(len(yms) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False, sharex=True, sharey=True)
        for ax, ym in zip(axes.ravel(), yms):
            g = Hs[Hs.ym == ym]
            ax.scatter(H.R, H.F, s=0.3, color="0.88", alpha=0.5, edgecolors="none")
            if len(g):
                panel(ax, g, args.tmin, args.tmax, args.dot * 1.5, min(1.0, args.alpha * 2))
            else:
                style(ax, "", "")
            y, mth = ym.split("-")
            if rec is None:
                ups = [up.get((r, ym)) for r in H.recorder.unique()]
                ups = [u for u in ups if u is not None]
                up_txt = f"mean uptime {np.mean(ups):.0%} of {len(ups)} rec." if ups else "uptime n/a"
                ax.set_title(f"{MONTHS[int(mth) - 1]} {y}: {len(g):,} singer-hours, {g.tid.nunique():,} singer-periods\n{g.recorder.nunique()} recorders; {up_txt}", fontsize=7.5, loc="left")
            else:
                u = up.get((rec, ym))
                up_txt = f"uptime {u:.0%}" if u is not None else "uptime 0 %"
                ax.set_title(f"{MONTHS[int(mth) - 1]} {y}: {len(g):,} singer-hours, {g.tid.nunique():,} singer-periods\n{up_txt}", fontsize=7.5, loc="left")
                if u is None or u < 0.05:
                    ax.set_facecolor("0.96")
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(yms):]:
            ax.set_axis_off()
        for ax in axes[-1, :]:
            ax.set_xlabel("syllable rate (Hz)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("carrier (kHz)", fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.94, 0.97))
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(args.tmin, args.tmax), cmap="jet")
        cax = fig.add_axes([0.955, 0.15, 0.01, 0.7]); cb = fig.colorbar(sm, cax=cax); cb.set_label("HRRR air temperature at the hour (°C)")
        fig.savefig(path + ".pdf"); fig.savefig(path + ".png", dpi=130); plt.close(fig)
    month_grid(H, f"singer-hours by month, all recorders (grey: whole range), {span}", args.out + "_by_month")

    # C. per recorder
    rdir = args.out + "_recorders"
    os.makedirs(rdir, exist_ok=True)
    for rec, g in H.groupby("recorder"):
        month_grid(g, f"{g.site.iloc[0]} {rec}: singer-hours by month (grey: all recorders, whole range), {span}, {'+'.join(periods)}; "
                      f"uptime = share of the month's {'+'.join(periods)} hours recorded (shaded panel: < 5 %)",
                   os.path.join(rdir, f"{g.site.iloc[0]}_{rec}_by_month"), rec=rec)
    print(f"wrote {args.out}_all, _by_month, and {H.recorder.nunique()} per-recorder grids in {rdir}")


if __name__ == "__main__":
    main()
