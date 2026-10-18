"""How often is a track's fixed-window carrier (its reference band) a harmonic of its
own fundamental? Per night singer-period: the ratio of F_fix to the median
fundamental centroid F_c of the track's windows (F_c is the centroid of each
window-animal's lowest band). A ratio near 2 or 3 means the linker chose a
harmonic as the reference band (night_animals.reference_carrier picks the band
present in most windows, which in a dense chorus can be the harmonic). Summary
per streak: share of singer-periods with ratio >= 1.8.

Usage:
    python analysis/callers/year_harmonic_ref.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --jobs 12
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd


def _unit(args):
    unit, tracks, roots = args
    for r in roots:
        p = os.path.join(r, unit, "night_night_global_window_animals.csv")
        if os.path.exists(p):
            break
    else:
        return []
    try:
        W = pd.read_csv(p, usecols=["track", "F_c", "F_fix", "F_lo", "F_hi"])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for t in tracks:
        g = W[(W.track == t) & W.F_fix.notna()]
        if g.empty:
            continue
        out.append(dict(an=f"{unit}:{t}", F_fix=g.F_fix.median(), F_c=g.F_c.median(), F_lo=g.F_lo.median(), n=len(g),
                        frac_fund_below=float((g.F_hi < 0.7 * g.F_fix).mean())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    A["unit"] = A.an.str.split(":").str[0]; A["track"] = A.an.str.split(":").str[1].astype(int)
    jobs = [(u, g.track.tolist(), args.roots) for u, g in A.groupby("unit")]
    with Pool(args.jobs) as pool:
        parts = pool.map(_unit, jobs, chunksize=16)
    T = pd.DataFrame([r for p in parts for r in p]).merge(A[["an", "streak"]], on="an")
    T["ratio"] = T.F_fix / T.F_c
    T["harm_ref"] = T.ratio >= 1.8
    T.to_csv(os.path.join(args.root, f"year_harmonic_ref_tracks{args.suffix}.csv"), index=False, float_format="%.3f")
    g = T.groupby("streak").agg(n=("an", "size"), harm_ref_share=("harm_ref", "mean"), ratio_med=("ratio", "median"), F_c_med=("F_c", "median"), F_fix_med=("F_fix", "median"))
    g = g.join(S[["F25", "R25", "harmonic_of"]], how="left").sort_values("harm_ref_share", ascending=False)
    g.to_csv(os.path.join(args.root, f"year_harmonic_ref{args.suffix}.csv"), float_format="%.3f")
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print(f"{len(T):,} singer-periods; {T.harm_ref.mean():.1%} have a reference band >= 1.8x their fundamental centroid")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
