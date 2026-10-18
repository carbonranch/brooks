"""The v2 corpus run: window_blobs2.py over the archive, one calendar month
per stage, DETECTION ONLY -- no night linker inside the run. Linking is
cheap (night_global.py --all: minutes per 500 nights) and still evolving,
so it is done afterwards in batch over the finished window tables; the
detector is the settled, expensive part (MBE, 2026-09-26: stop the old
method, move to v2 starting with Nov 2024 - Nov 2025).

Output roots are runs/v2/y<year>/<recorder>_<site>_<yyyymmdd>_<period>/
with the per-window *_animals.csv / _parts.csv / .csv tables, exactly as the
old run laid them out. Resumable: run_month.py skips windows whose outputs
exist.

Usage:
    python analysis/callers/run_corpus2.py --first-month 2024-11 --last-month 2025-11 --jobs 22 [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

from run_month import recorder_periods

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "v2")


def month_stages(units, first_month: str, last_month: str, newest_first: bool) -> list[tuple[str, str, int]]:
    per_month = defaultdict(int)
    for rec, site, d, per, n in units:
        per_month[(d.year, d.month)] += 1
    y0, m0 = (int(x) for x in first_month.split("-")); y1, m1 = (int(x) for x in last_month.split("-"))
    months = [ym for ym in per_month if (y0, m0) <= ym <= (y1, m1)]
    months = sorted(months, reverse=newest_first)
    out = []
    for y, m in months:
        first = date(y, m, 1)
        last = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
        out.append((f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}", per_month[(y, m)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-month", default="2024-11")
    ap.add_argument("--last-month", default="2025-11")
    ap.add_argument("--newest-first", action="store_true")
    ap.add_argument("--jobs", type=int, default=22)
    ap.add_argument("--min-hours", type=int, default=6)
    ap.add_argument("--periods", default="night,day")
    ap.add_argument("--engine", default="window_blobs2.py")
    ap.add_argument("--linker", default="none", help="in-run night linker; 'none' = detection only (link afterwards with night_global.py --all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    periods = tuple(p.strip() for p in args.periods.split(","))
    os.makedirs(RUNS, exist_ok=True)
    log = open(os.path.join(RUNS, "run_corpus2.log"), "a")

    def say(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    units = recorder_periods(f"{args.first_month}-01", f"{args.last_month}-31" if args.last_month[-2:] in ("01", "03", "05", "07", "08", "10", "12") else f"{args.last_month}-30", args.min_hours, periods)
    stages = month_stages(units, args.first_month, args.last_month, args.newest_first)
    say(f"v2 corpus: {len(units)} units, {sum(u[4] for u in units)} hours, {len(stages)} stages, jobs={args.jobs}, engine {args.engine}, linker {args.linker}")
    for first, last, n in stages:
        say(f"  stage {first} .. {last}: {n} units")
    if args.dry_run:
        return
    t_all = time.time()
    for first, last, n in stages:
        out_root = os.path.join(RUNS, f"y{first[:4]}")
        say(f"=== stage {first} .. {last} ({n} units) -> {out_root}")
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(HERE, "run_month.py"), "--first-night", first, "--last-night", last,
                            "--jobs", str(args.jobs), "--min-hours", str(args.min_hours), "--periods", args.periods,
                            "--out-root", out_root, "--engine", args.engine, "--linker", args.linker])
        say(f"=== stage {first} .. {last} done rc={p.returncode} in {(time.time() - t0) / 3600:.2f} h")
    say(f"v2 corpus finished in {(time.time() - t_all) / 3600:.2f} h")


if __name__ == "__main__":
    main()
