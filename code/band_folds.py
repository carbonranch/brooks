"""Fine folds for the candidate bands of a split type (exploratory, 2026-10-15; MBE: "syllable shape is going to be the
real value here").

Takes the band catalog written beside the year's tables (animal_nights_streaks<suffix>.csv, streaks_coarse<suffix>.csv,
paper_ids<suffix>.csv with band ids like 5a) and runs type_reference_figure.build over the listed band streaks only:
up to 40 of each band's clearest singer-nights near 25 C (widened for cold bands), the deepest clean window of each,
the fine fold (64 phase bins) with the per-window descriptors (duration, duty, glide, bend, bandwidth). Cache in
<root>/<cache-name>/ (folds_<key>.npy, windows_<key>.csv, sample.csv), which band_shape_figure.py then compares.

Usage:
  python analysis/callers/band_folds.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9bands \
      --streaks 36011 36012 31011 31012 31013 34011 34012 54011 54012 54013 59011 59012 \
      --unit-roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --jobs 8
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import type_reference_figure as TR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9bands"); ap.add_argument("--streaks", type=int, nargs="+", required=True)
    ap.add_argument("--unit-roots", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--cache-name", default="type_reference_bands")
    args = ap.parse_args()
    # a trimmed catalog holding only the bands, so build() folds nothing else (its chirp half still runs over the chirp catalog)
    sub = args.suffix + "only"
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv")); A[A.streak.isin(args.streaks)].to_csv(os.path.join(args.root, f"animal_nights_streaks{sub}.csv"), index=False)
    S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0); S.loc[[s for s in args.streaks if s in S.index]].to_csv(os.path.join(args.root, f"streaks_coarse{sub}.csv"))
    P = pd.read_csv(os.path.join(args.root, f"paper_ids{args.suffix}.csv")); P[P.streak.isin(args.streaks)].to_csv(os.path.join(args.root, f"paper_ids{sub}.csv"), index=False)
    TR.MIN_N = min(TR.MIN_N, int(S.loc[[s for s in args.streaks if s in S.index], "n"].min()))     # small bands must not be skipped
    TR.build(args.root, sub, args.unit_roots, args.jobs, 0, os.path.join(args.root, args.cache_name))
    print("done")


if __name__ == "__main__":
    main()
