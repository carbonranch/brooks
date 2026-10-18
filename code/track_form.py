"""Song form per singer-period, from the detector's own windows.

For every singer-period (a global track through one recorder-night, the rows of
temperature_animal_nights.csv) this reads its windows in the unit's window-animals table and
measures:
  rc        share of windows with a chirp rate (the chirp channel's bout rhythm)
  Rc        median chirp rate over those windows
  bw_rel    median relative width of the detected band, (F_hi - F_lo) / F_c: a hooked tick or a
            tree-cricket trill is 5-15 % wide, a broadband katydid burst 30 % and more
  R_scatter robust sd of log syllable rate across the track's windows
and classifies:
  form      chirp (rc >= CHIRP), trill (rc < TRILL), mixed (between)
  band      narrow (bw_rel < BROAD) or broad
  placeable False when R_scatter > R_SCATTER: a syllable rate that wanders by more than 20 % within
            a night is not a rate (chirpers whose within-chirp rate the detector cannot hold)
The streak builder clusters firm classes separately (streaks_by_form.py), so a streak is defined by
carrier, rate AND song form (MBE, 2026-10-06: "to the extent we can reliably do it, we should
define streaks by type as well as carrier and rate").

Usage:
  python analysis/callers/track_form.py analysis/callers/runs/v2/year_nov24_nov25 --roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --jobs 8
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

CHIRP, TRILL = 0.5, 0.1
BROAD = 0.25
R_SCATTER = 0.2


def _unit(args):
    unit, tracks, roots = args
    for r in roots:
        p = os.path.join(r, unit, "night_night_global_window_animals.csv")
        if os.path.exists(p):
            break
    else:
        return []
    try:
        W = pd.read_csv(p, usecols=["track", "R", "F_c", "F_lo", "F_hi", "chirp_rate", "depth"])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for t in tracks:
        g = W[(W.track == t)]
        if g.empty:
            continue
        gr = g[g.R > 0]
        lr = np.log(gr.R.to_numpy()) if len(gr) else np.array([])
        scatter = float(np.median(np.abs(lr - np.median(lr))) * 1.4826) if len(lr) >= 3 else 0.0
        bw = ((g.F_hi - g.F_lo) / g.F_c).median()
        rc = float(g.chirp_rate.notna().mean())
        out.append(dict(an=f"{unit}:{t}", n_win=len(g), rc=rc, Rc=float(g.chirp_rate.median()) if g.chirp_rate.notna().any() else np.nan,
                        bw_rel=float(bw), R_scatter=scatter))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    A = pd.read_csv(os.path.join(args.root, "temperature_animal_nights.csv"), usecols=["an"])
    A["unit"] = A.an.str.rsplit(":", n=1).str[0]; A["track"] = A.an.str.rsplit(":", n=1).str[1].astype(int)
    jobs = [(u, g.track.tolist(), args.roots) for u, g in A.groupby("unit")]
    with Pool(args.jobs) as pool:
        parts = pool.map(_unit, jobs, chunksize=16)
    F = pd.DataFrame([r for p in parts for r in p])
    F["form"] = np.where(F.rc >= CHIRP, "chirp", np.where(F.rc < TRILL, "trill", "mixed"))
    F["band"] = np.where(F.bw_rel >= BROAD, "broad", "narrow")
    F["placeable"] = F.R_scatter <= R_SCATTER
    F.to_csv(os.path.join(args.root, "track_form.csv"), index=False, float_format="%.4f")
    print(f"{len(F):,} singer-periods; form {F.form.value_counts().to_dict()}; band {F.band.value_counts().to_dict()}; unplaceable {int((~F.placeable).sum())}")
    print("relative bandwidth deciles:", np.round(F.bw_rel.quantile([.1, .25, .5, .75, .9, .95, .99]).to_numpy(), 3).tolist())
    print("bandwidth by form (median):", F.groupby("form").bw_rel.median().round(3).to_dict())
    print(pd.crosstab(F.form, F.band).to_string())


if __name__ == "__main__":
    main()
