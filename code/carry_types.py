"""Carry one run's type partition over to another run of the same tracks: the v3 types with v4 carriers.

The v4 relink (METHOD 19 AV) changed only the fixed-window carrier; its tracks and singer-nights are the same as v3's
(26,116 in common), and the per-singer-night carriers moved by a median of 0.0 % (IQR -0.07 to +0.15 %). Re-clustering
those positions nevertheless re-partitioned the catalog (type 10 cut 59/40 into two along-line segments, 14 and 19
merged, 23 split): average linkage at this density is unstable to perturbations far below the type spacing. So the
v3 partition is carried over by singer-night id and the v4 measurements are attached to it:

  animal_nights_streaks<out>.csv  = v3 assignments (streak, form, band, ...) with v4 F, R, T, F25, R25, depth
  streaks_coarse<out>.csv         = v3 rows with F25, R25, their IQRs and the F/R ranges recomputed from the v4 values
  paper_ids<out>.csv              = v3 paper numbers (so types keep their names)
  animal_days_streaks<out>.csv    = the same for the day table where both runs have it

Usage:
  python analysis/callers/carry_types.py --from analysis/callers/runs/v3/year_nov24_nov25 --to analysis/callers/runs/v4/year_nov24_nov25 \
      --suffix _v9 --out _v3types
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

B_R, B_F = 0.0154, 0.0046      # build_streaks' per-degree corrections to 25 C


def carry(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    """src = the partitioned table (v3), dst = the same singer-periods re-measured (v4): src's assignment columns, dst's measurements."""
    meas = [c for c in ("F", "R", "T", "depth", "Rc", "rc", "bw_rel", "R_scatter", "contrast", "harm2") if c in dst.columns]
    out = src.drop(columns=[c for c in meas if c in src.columns]).merge(dst[["an"] + meas], on="an", how="inner")
    if "T" in out.columns:
        out["R25"] = out.R * np.exp(B_R * (25 - out["T"])); out["F25"] = out.F * np.exp(B_F * (25 - out["T"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True); ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--suffix", default="_v9"); ap.add_argument("--out", default="_v3types")
    args = ap.parse_args()
    A3 = pd.read_csv(os.path.join(args.src, f"animal_nights_streaks{args.suffix}.csv")); A4 = pd.read_csv(os.path.join(args.dst, f"animal_nights_streaks{args.suffix}.csv"))
    A = carry(A3, A4)
    dF = 100 * (A.F.to_numpy() / A3.set_index("an").F.reindex(A.an).to_numpy() - 1)
    print(f"nights: {len(A3):,} in {args.src}, {len(A4):,} in {args.dst}, {len(A):,} carried; carrier change median {np.nanmedian(dF):+.2f} %, mean |change| {np.nanmean(np.abs(dF)):.2f} %")
    A.to_csv(os.path.join(args.dst, f"animal_nights_streaks{args.out}.csv"), index=False, float_format="%.4f")
    S = pd.read_csv(os.path.join(args.src, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    g = A.groupby("streak")
    for col, fn in (("F25", lambda d: np.exp(np.log(d.F25).median())), ("R25", lambda d: np.exp(np.log(d.R25).median())),
                    ("F25_iqr", lambda d: d.F25.quantile(.75) - d.F25.quantile(.25)), ("R25_iqr", lambda d: d.R25.quantile(.75) - d.R25.quantile(.25)),
                    ("F_min", lambda d: d.F.min()), ("F_max", lambda d: d.F.max()), ("R_min", lambda d: d.R.min()), ("R_max", lambda d: d.R.max()), ("n", lambda d: len(d))):
        if col in S.columns:
            v = g.apply(fn); v = v[v.index.isin(S.index)]; S.loc[v.index, col] = v        # -1 = unassigned, not a streak
    S.to_csv(os.path.join(args.dst, f"streaks_coarse{args.out}.csv"))
    shutil.copy(os.path.join(args.src, f"paper_ids{args.suffix}.csv"), os.path.join(args.dst, f"paper_ids{args.out}.csv"))
    d3, d4 = os.path.join(args.src, f"animal_days_streaks{args.suffix}.csv"), os.path.join(args.dst, f"animal_days_streaks{args.suffix}.csv")
    if os.path.exists(d3) and os.path.exists(d4):
        D = carry(pd.read_csv(d3), pd.read_csv(d4)); D.to_csv(os.path.join(args.dst, f"animal_days_streaks{args.out}.csv"), index=False, float_format="%.4f"); print(f"days: {len(D):,} carried")
    for extra in ("candidate_decisions.csv",):
        p = os.path.join(args.src, extra)
        if os.path.exists(p) and not os.path.exists(os.path.join(args.dst, extra)):
            shutil.copy(p, os.path.join(args.dst, extra))
    print(f"wrote *{args.out}.csv in {args.dst}")


if __name__ == "__main__":
    main()
