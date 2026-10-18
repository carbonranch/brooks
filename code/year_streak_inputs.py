"""Build the inputs the streak-definition chain expects, for a date range
spanning several run roots (the year), from the singer-hours table written
by year_hours_figure.py.

Writes into <out_root>:
  temperature_animal_nights.csv   one row per singer-period (>= 2 h track):
                                  an, recorder, site, night, F, R, T, depth, type_corrected
  temperature_season_by_type.csv  a stub (build_streaks joins it; values NaN)
  streak_hourly_points_all.csv    one row per singer-hour with 'an' = the
                                  singer-period id and streak = -1, so
                                  split_streaks_by_T / merge_streaks /
                                  streak_overlay_figure read it unchanged
  weather_hourly_hrrr.csv         the concatenated HRRR table

Then run, e.g.:
  build_streaks.py <out_root> --bR 0.0154 --bF 0.0046 --coupling 0.55 --coupling-hf 0.25 --along-tol 0.6 --perp-tol 0.06 --suffix _k55b
  split_streaks_by_T.py <out_root> --in _k55b --out _v4 --coupling 0.55 --slope 2.4 --sep-sd 2.5 --dbic 50 --max-k 2
  merge_streaks.py <out_root> --in _v4 --out _v6 --coupling 0.55
  streak_overlay_figure.py <out_root> --suffix _v6 --coupling 0.55

Usage:
    python analysis/callers/year_streak_inputs.py analysis/callers/runs/year_figs/nov24_nov25_night_hours.csv analysis/callers/runs/year_nov24_nov25 --roots analysis/callers/runs/y2024 analysis/callers/runs/y2025
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

TZ = "America/Chicago"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hours_csv")
    ap.add_argument("out_root")
    ap.add_argument("--roots", nargs="+", required=True, help="run roots holding weather_hourly_hrrr.csv")
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    H = pd.read_csv(args.hours_csv)
    H["hour"] = pd.to_datetime(H.hour, utc=True)
    # the period's date is in the tid (…_<yyyymmdd>_<night|day>:<track>)
    H["night"] = pd.to_datetime(H.tid.str.extract(r"_(\d{8})_(?:night|day):")[0], format="%Y%m%d")
    H["period"] = H.tid.str.extract(r"_(night|day):")[0]
    if "Rc" not in H.columns:          # hours tables written before the chirp columns existed
        H["Rc"] = float("nan"); H["rc_frac"] = 0.0
    A = H.groupby("tid").agg(recorder=("recorder", "first"), site=("site", "first"), night=("night", "first"), period=("period", "first"),
                             F=("F", "median"), R=("R", "median"), T=("T", "mean"), depth=("depth", "median"), n_hours=("hour", "size"),
                             Rc=("Rc", "median"), rc_frac=("rc_frac", "mean")).reset_index()
    A = A.rename(columns={"tid": "an"})
    A["type_corrected"] = -1
    A.to_csv(os.path.join(args.out_root, "temperature_animal_nights.csv"), index=False, float_format="%.4f")
    pd.DataFrame(columns=["type", "R_perC", "F_perC"]).to_csv(os.path.join(args.out_root, "temperature_season_by_type.csv"), index=False)
    P = H.rename(columns={"tid": "an"})[["hour", "R", "F", "depth", "T", "Rc", "rc_frac", "an", "site", "recorder", "night"]].copy()
    P["n"] = 1
    P["streak"] = -1
    P["hour"] = P.hour.dt.tz_convert(TZ)
    P.to_csv(os.path.join(args.out_root, "streak_hourly_points_all.csv"), index=False, float_format="%.4f")
    wx = []
    for r in args.roots:
        p = os.path.join(r, "weather_hourly_hrrr.csv")
        if os.path.exists(p):
            wx.append(pd.read_csv(p))
    W = pd.concat(wx, ignore_index=True)
    W["t"] = pd.to_datetime(W.t, utc=True)
    W = W.sort_values("t").drop_duplicates("t")
    W["t"] = W.t.dt.tz_convert(TZ)
    W.to_csv(os.path.join(args.out_root, "weather_hourly_hrrr.csv"), index=False, float_format="%.2f", date_format="%Y-%m-%d %H:%M:%S%z")
    W.to_csv(os.path.join(args.out_root, "weather_hourly.csv"), index=False, float_format="%.2f", date_format="%Y-%m-%d %H:%M:%S%z")
    print(f"{len(A):,} singer-periods ({(A.period == 'night').sum():,} night, {(A.period == 'day').sum():,} day), {len(P):,} singer-hours, "
          f"{A.recorder.nunique()} recorders, {A.night.min():%Y-%m-%d}..{A.night.max():%Y-%m-%d}; wrote {args.out_root}")


if __name__ == "__main__":
    main()
