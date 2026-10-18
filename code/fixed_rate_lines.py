"""Find fixed-rate lines per recorder-month in a detection run: machines at a
site (a pump, a fan, a heater) modulate broadband noise at one rate that never
drifts with temperature, and the detector reports them as callers in every
band they leak into. Recorder 8604 at House: 59.25 Hz over 2-7 kHz, Nov 2024
to Jan 2025 (60 % of its January detections).

For each recorder-month, syllable rates are binned at BIN Hz. A bin is a line
when it holds >= MIN_N detections and >= MIN_SHARE of the recorder-month's
detections, is >= MIN_CONTRAST times its neighbouring bins (+-2 and +-3 bins),
and its detections come from >= MIN_UNITS recorder-periods. Adjacent line bins
merge. Animals drift 10-30 % over a night, so a real chorus cannot make a
0.25 Hz line that survives a month.

Usage:
    python analysis/callers/fixed_rate_lines.py <run_root> [<run_root> ...] [--out PATH]
Writes <run_root>/fixed_rate_lines.csv for each root given (or --out for one
merged table): rec, site, month, R, n, share, contrast, units, F_c_med, F_span.
postfilter.load_lines(run_root) reads it.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

BIN = 0.25
MIN_N = 40
MIN_SHARE = 0.03
MIN_CONTRAST = 15.0
MIN_UNITS = 5
UNIT_RE = re.compile(r"BARLT_(\d+)_(.+)_(\d{8})_(night|day)$")


def _collect_unit(u: str) -> pd.DataFrame | None:
    m = UNIT_RE.match(os.path.basename(u))
    if not m or not os.path.isdir(u):
        return None
    rows = []
    for p in glob.glob(os.path.join(u, "*_animals.csv")):
        if os.path.basename(p).startswith("night_"):
            continue
        try:
            a = pd.read_csv(p, usecols=["F_c", "R"])
        except Exception:  # noqa: BLE001
            continue
        a = a[a.R.notna()]
        if not a.empty:
            rows.append(a)
    if not rows:
        return None
    a = pd.concat(rows, ignore_index=True)
    a["rec"] = m.group(1); a["site"] = m.group(2); a["month"] = m.group(3)[:6]; a["unit"] = os.path.basename(u)
    return a


def collect(run_root: str, jobs: int = 1) -> pd.DataFrame:
    units = sorted(glob.glob(os.path.join(run_root, "BARLT_*")))
    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            parts = pool.map(_collect_unit, units, chunksize=8)
    else:
        parts = [_collect_unit(u) for u in units]
    rows = [a for a in parts if a is not None]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["F_c", "R", "rec", "site", "month", "unit"])


def find_lines(A: pd.DataFrame) -> pd.DataFrame:
    out = []
    A = A.copy()
    A["rbin"] = (A.R.astype(float) / BIN).round().astype(int)
    for (rec, site, month), g in A.groupby(["rec", "site", "month"]):
        n = len(g)
        if n < MIN_N:
            continue
        vc = g.rbin.value_counts()
        cand = []
        for rb, c in vc.items():
            if c < MIN_N or c < MIN_SHARE * n:
                continue
            nb = vc.reindex([rb - 3, rb - 2, rb + 2, rb + 3]).fillna(0).mean()
            contrast = c / max(nb, 1.0)
            if contrast < MIN_CONTRAST:
                continue
            if g[g.rbin == rb].unit.nunique() < MIN_UNITS:
                continue
            cand.append((rb, c, contrast))
        cand.sort()
        # merge adjacent bins
        groups: list[list[tuple]] = []
        for cnd in cand:
            if groups and cnd[0] - groups[-1][-1][0] <= 1:
                groups[-1].append(cnd)
            else:
                groups.append([cnd])
        for grp in groups:
            bins = [c[0] for c in grp]
            s = g[g.rbin.isin(bins)]
            out.append(dict(rec=rec, site=site, month=month, R=round(float(s.R.mean()), 3), n=len(s), share=round(len(s) / n, 4),
                            contrast=round(max(c[2] for c in grp), 1), units=s.unit.nunique(),
                            F_c_med=round(float(s.F_c.median()), 0), F_span=round(float(s.F_c.quantile(0.9) - s.F_c.quantile(0.1)), 0)))
    cols = ["rec", "site", "month", "R", "n", "share", "contrast", "units", "F_c_med", "F_span"]
    return pd.DataFrame(out, columns=cols).sort_values(["rec", "month", "R"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_roots", nargs="+")
    ap.add_argument("--out", default=None, help="write one merged table here instead of <run_root>/fixed_rate_lines.csv per root")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    tables = []
    for root in args.run_roots:
        A = collect(root, args.jobs)
        L = find_lines(A)
        print(f"{root}: {len(A)} rated detections, {L.rec.nunique() if len(L) else 0} recorders with lines, {len(L)} lines")
        if len(L):
            pd.set_option("display.width", 200)
            print(L.to_string(index=False))
        if args.out is None:
            L.to_csv(os.path.join(root, "fixed_rate_lines.csv"), index=False)
        tables.append(L)
    if args.out is not None:
        pd.concat(tables, ignore_index=True).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
