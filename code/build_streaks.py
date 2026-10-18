"""Streaks: the month's animal-nights clustered in temperature-corrected
(carrier, rate).

Each animal-night is brought to 25 degC with the pooled within-animal slopes
(rate bR, carrier bF, in fraction per degC, as printed by
temperature_effects.py), then clustered by average linkage in
(log F25 / 0.10, log R25 / 0.12) cut at distance 1 -- i.e. ~10 % in carrier
and ~12 % in rate. A cluster with >= MIN_N animal-nights is a streak. Streak
ids are the cluster labels and are NOT stable between runs; match streaks
across runs by (F25, R25).

Usage:
    python analysis/callers/build_streaks.py analysis/callers/runs/month_v2 --bR 0.0163 --bF 0.0032
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

F_TOL, R_TOL = 0.10, 0.12
MIN_N = 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--bR", type=float, required=True, help="pooled within-animal rate slope, fraction per degC")
    ap.add_argument("--bF", type=float, required=True, help="pooled within-animal carrier slope, fraction per degC")
    ap.add_argument("--along-tol", type=float, default=None,
                    help="anisotropic clustering: tolerance ALONG the coupling line (log-rate units, e.g. 0.30); "
                         "with --perp-tol replaces the isotropic 10 %% / 12 %% metric")
    ap.add_argument("--perp-tol", type=float, default=0.035, help="tolerance ACROSS the line (log-carrier units)")
    ap.add_argument("--coupling", type=float, default=None, help="d log F / d log R of the lines; default bF/bR")
    ap.add_argument("--coupling-hf", type=float, default=None,
                    help="line slope for singer-nights with F25 >= --hf-split Hz (katydids run at ~0.25, crickets at ~0.55)")
    ap.add_argument("--hf-split", type=float, default=10000.0)
    ap.add_argument("--suffix", default="", help="suffix for the output files, e.g. _aniso")
    args = ap.parse_args()
    root = args.root
    M = pd.read_csv(os.path.join(root, "temperature_animal_nights.csv"))
    M["R25"] = M.R * np.exp(args.bR * (25 - M["T"]))
    M["F25"] = M.F * np.exp(args.bF * (25 - M["T"]))
    if args.along_tol is None:
        X = np.column_stack([np.log(M.F25) / F_TOL, np.log(M.R25) / R_TOL])
    else:
        # rotate into (along, across) the common coupling line: a species'
        # singer-nights spread along the line (individual variation and the
        # residual temperature effect) but stay on it, so the tolerance along
        # can be several times the tolerance across
        k = args.coupling if args.coupling is not None else args.bF / args.bR
        Z = np.column_stack([np.log(M.R25), np.log(M.F25)])
        kk = np.full(len(M), k)
        if args.coupling_hf is not None:
            kk[M.F25.to_numpy() >= args.hf_split] = args.coupling_hf
        # per-point rotation into (along, across) the line of slope kk; the two
        # slope groups never meet because their across coordinates differ by far
        # more than the tolerance
        nrmk = np.hypot(1.0, kk)
        along = (Z[:, 0] + kk * Z[:, 1]) / nrmk        # projection on the unit line direction (~log R units)
        across = (-kk * Z[:, 0] + Z[:, 1]) / nrmk      # perpendicular distance from the line (~log F units)
        X = np.column_stack([along / args.along_tol, across / args.perp_tol])
        print(f"anisotropic: along {args.along_tol:.2f} (log R), across {args.perp_tol:.3f} (log F), line slope {k:.2f}"
              + (f" below / {args.coupling_hf:.2f} above {args.hf_split:.0f} Hz" if args.coupling_hf is not None else ""))
    M["streak"] = sch.fcluster(sch.linkage(X, "average"), t=1.0, criterion="distance")
    S = pd.read_csv(os.path.join(root, "temperature_season_by_type.csv")).set_index("type")
    M = M.join(S[["R_perC", "F_perC"]], on="type_corrected")
    g = M.groupby("streak")
    T = g.agg(n=("an", "size"), F25=("F25", "median"), F25_iqr=("F25", lambda s: s.quantile(.75) - s.quantile(.25)),
              R25=("R25", "median"), R25_iqr=("R25", lambda s: s.quantile(.75) - s.quantile(.25)),
              F_min=("F", "min"), F_max=("F", "max"), R_min=("R", "min"), R_max=("R", "max"),
              n_sites=("site", "nunique"), n_nights=("night", "nunique"), depth=("depth", "median"),
              R_perC=("R_perC", "median"), F_perC=("F_perC", "median"))
    T["coupling"] = (pd.to_numeric(T.F_perC, errors="coerce") / pd.to_numeric(T.R_perC, errors="coerce")).round(2)
    T = T[T.n >= MIN_N].sort_values("F25")
    pd.set_option("display.width", 220)
    print(f"{len(T)} streaks with >= {MIN_N} animal-nights ({T.n.sum()} of {len(M)} animal-nights)")
    print(T.round(1).to_string())
    T.to_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), float_format="%.3f")
    M[["an", "recorder", "site", "night", "F", "R", "T", "F25", "R25", "depth", "streak", "type_corrected"]].to_csv(
        os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"), index=False, float_format="%.3f")
    print(f"wrote streaks_coarse{args.suffix}.csv, animal_nights_streaks{args.suffix}.csv")


if __name__ == "__main__":
    main()
