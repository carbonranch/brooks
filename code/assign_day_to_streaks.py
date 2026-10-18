"""Assign the DAY singer-periods of the year to the night-defined streaks.

The v6 streak definition was built on night singer-periods only. Rather than
redefine the units, each day singer-period (from year_hours_figure's
*_day_hours.csv: one row per singer per hour) is reduced to its median rate,
carrier and temperature, brought to 25 C with the pooled within-night slopes,
and placed in the ridge frame (coupling 0.55 below 10 kHz, 0.25 above). It
joins the nearest streak if it lies within PERP_TOL across the line and
within the streak's own along-line extent (+/- ALONG_PAD); otherwise -1.

Writes animal_days_streaks<suffix>.csv (an, recorder, site, night(date),
period=day, F, R, T, F25, R25, streak, across, along) and prints per-streak
day counts and months beside the night counts.

Usage:
    python analysis/callers/assign_day_to_streaks.py analysis/callers/runs/year_nov24_nov25 \
        --day-hours analysis/callers/runs/year_figs/nov24_nov25_day_hours.csv --suffix _v6
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

BR, BF = 0.0154, 0.0046          # pooled within-night slopes (Sept 2024, HRRR), per degC
PERP_TOL = 0.06
ALONG_PAD = 0.05
HF_SPLIT = 10000.0


def coupling(F):
    return np.where(F >= HF_SPLIT, 0.25, 0.55)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--day-hours", required=True)
    ap.add_argument("--suffix", default="_v6")
    ap.add_argument("--audio-form", default=None, help="track_form_audio table for the day singer-periods: a day singer joins a streak only if its form matches the streak's form; mixed and unplaceable stay unassigned")
    args = ap.parse_args()
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[S.harmonic_of < 0]
    H = pd.read_csv(args.day_hours)
    D = H.groupby("tid").agg(R=("R", "median"), F=("F", "median"), T=("T", "median"), n_hours=("hour", "size"),
                             recorder=("recorder", "first"), site=("site", "first"), hour=("hour", "first")).reset_index()
    D = D[D.n_hours >= 2].rename(columns={"tid": "an"})
    D["night"] = pd.to_datetime(D.hour, utc=True).dt.tz_convert("America/Chicago").dt.strftime("%Y-%m-%d")
    D["period"] = "day"
    D["R25"] = D.R * np.exp(BR * (25 - D["T"]))
    D["F25"] = D.F * np.exp(BF * (25 - D["T"]))
    # streak frames from their night members
    A = A[A.streak.isin(S.index)]
    frames = {}
    for sid, g in A.groupby("streak"):
        c = float(coupling(np.array([S.loc[sid, "F25"]]))[0])
        lr0, lf0 = np.log(S.loc[sid, "R25"]), np.log(S.loc[sid, "F25"])
        along = ((np.log(g.R25) - lr0) + c * (np.log(g.F25) - lf0)) / np.sqrt(1 + c * c)
        frames[sid] = (c, lr0, lf0, along.quantile(0.02) - ALONG_PAD, along.quantile(0.98) + ALONG_PAD)
    lr, lf = np.log(D.R25.to_numpy()), np.log(D.F25.to_numpy())
    best = np.full(len(D), -1); best_x = np.full(len(D), np.inf); best_al = np.full(len(D), np.nan)
    for sid, (c, lr0, lf0, a_lo, a_hi) in frames.items():
        across = ((lf - lf0) - c * (lr - lr0)) / np.sqrt(1 + c * c)
        along = ((lr - lr0) + c * (lf - lf0)) / np.sqrt(1 + c * c)
        ok = (np.abs(across) <= PERP_TOL) & (along >= a_lo) & (along <= a_hi) & (np.abs(across) < best_x)
        best[ok] = sid; best_x[ok] = np.abs(across[ok]); best_al[ok] = along[ok]
    D["streak"] = best; D["across"] = np.where(best >= 0, best_x, np.nan); D["along"] = best_al
    if args.audio_form and "form" in S.columns:
        Fa = pd.read_csv(args.audio_form); Fa = Fa[Fa.status == "ok"].set_index("an")
        D["form"] = D.an.map(Fa.form_audio).fillna("unmeasured"); D["harm2"] = D.an.map(Fa.harm2).fillna(False).astype(bool)
        sform = D.streak.map(S.form)
        keep = (D.form == sform) & ~D.harm2
        n0 = int((D.streak >= 0).sum()); D.loc[~keep, "streak"] = -1
        print(f"audio form: {int((D.streak >= 0).sum()):,} of {n0:,} day singer-periods keep a streak (form matches); forms {D.form.value_counts().to_dict()}")
    out = os.path.join(args.root, f"animal_days_streaks{args.suffix}.csv")
    cols = ["an", "recorder", "site", "night", "period", "F", "R", "T", "F25", "R25", "n_hours", "streak", "across", "along"] + (["form", "harm2"] if "form" in D.columns else [])
    D[cols].to_csv(out, index=False, float_format="%.4f")
    D["month"] = pd.to_datetime(D.night).dt.month
    An = A.copy(); An["month"] = pd.to_datetime(An.night).dt.month
    rows = []
    for sid in S.index:
        gd, gn = D[D.streak == sid], An[An.streak == sid]
        rows.append(dict(streak=sid, F25=S.loc[sid, "F25"], R25=S.loc[sid, "R25"], n_night=len(gn), n_day=len(gd),
                         day_share=len(gd) / max(1, len(gd) + len(gn)), day_months=",".join(str(m) for m in sorted(gd.month.unique())),
                         day_T=gd["T"].median() if len(gd) else np.nan, night_T=gn["T"].median()))
    R = pd.DataFrame(rows).sort_values("n_night", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(R.round(2).to_string(index=False))
    R.to_csv(os.path.join(args.root, f"streak_day_night{args.suffix}.csv"), index=False, float_format="%.3f")
    print(f"\n{len(D):,} day singer-periods >= 2 h; {(D.streak >= 0).sum():,} assigned ({(D.streak >= 0).mean():.0%}); wrote {out}")


if __name__ == "__main__":
    main()
