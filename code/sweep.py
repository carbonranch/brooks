"""Run window_blobs.py over a series of window centres, several at a time.

Usage:
    python analysis/callers/sweep.py --recorder BARLT_00008662 --site House \
        --start "2024-09-12 20:00" --end "2024-09-12 20:53" --step-s 60 \
        --out-dir analysis/callers/runs [--jobs 4] [--tag v2]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args, t: pd.Timestamp) -> str:
    stem = os.path.join(args.out_dir, f"{args.recorder}_{args.site}_{t:%Y%m%d_%H%M}{args.tag}")
    if os.path.exists(stem + "_animals.csv"):
        return f"{t:%H:%M} rc=0 skip (exists)"
    cmd = [sys.executable, os.path.join(HERE, args.engine), "--recorder", args.recorder,
           "--site", args.site, "--time", f"{t:%Y-%m-%d %H:%M:%S}", "--out", stem]
    if args.no_plot:
        cmd.append("--no-plot")
    p = subprocess.run(cmd, capture_output=True, text=True)
    tail = [l for l in p.stdout.splitlines() if "animals" in l or "wrote" in l]
    return f"{t:%H:%M} rc={p.returncode} " + (tail[0][:110] if tail else p.stderr.strip()[-200:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step-s", type=float, default=60.0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--engine", default="window_blobs.py",
                    help="window analysis script in this directory (window_blobs2.py for the v2 pipeline test)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    times = pd.date_range(args.start, args.end, freq=pd.Timedelta(seconds=args.step_s))
    with ThreadPoolExecutor(args.jobs) as ex:
        for line in ex.map(lambda t: run(args, t), times):
            print(line, flush=True)


if __name__ == "__main__":
    main()
