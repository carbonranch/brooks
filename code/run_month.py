"""Run the window inventory, night linking and carrier analysis over every
BARLT recorder-night on the volume in a date range. Resumable: windows whose
outputs exist are skipped, and a recorder-night whose night files exist is
skipped unless --redo-nights.

One recorder-night at a time, its windows in parallel, so the disk reads stay
within one recorder's directory.

Usage:
    python analysis/callers/run_month.py --first-night 2024-08-17 --last-night 2024-09-13 --jobs 12
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_ROOT = "/Volumes/8M2A/Data/Carbonranch/Ferguson/Audio"
TZ = os.environ.get("BROOKS_TZ", "America/Chicago")
NIGHT_START_H, NIGHT_END_H = 19, 7


AUDIO_ROOTS = [AUDIO_ROOT, "/Volumes/8M2C/Data/Carbonranch/Ferguson/Audio", "/Volumes/8M2D/Data/Carbonranch/Ferguson/Audio"]
# BROOKS_AUDIO_ROOTS=path1:path2 prepends other archives (same layout:
# <root>/<collection>/<site>/<recorder>_<dates>/<recorder>/<day>/S<start>_E<end>.wav)
AUDIO_ROOTS = [p for p in os.environ.get("BROOKS_AUDIO_ROOTS", "").split(":") if p] + AUDIO_ROOTS


def recorder_periods(first: str, last: str, min_hours: int, periods=("night",)):
    """(recorder, site, date, period, n_hours) for every BARLT recorder with at
    least min_hours hour-files in the period: 'night' is 19:00 local on the
    date to 07:00 the next morning, 'day' is 07:00 to 19:00 on the date."""
    rx = re.compile(r"S(\d{8}T\d{6})")
    hours: dict[tuple, set] = {}
    for root in AUDIO_ROOTS:
        for p in glob.glob(os.path.join(root, "*", "*", "*_*", "*", "*", "S*.wav")):   # any recorder with the S<start>_E<end> naming
            parts = p.split(os.sep)
            site, rec = parts[-5], parts[-3]
            m = rx.search(os.path.basename(p))
            if not m:
                continue
            t_utc = pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"), tz="UTC")
            t = t_utc.tz_convert(TZ)
            if t.hour >= NIGHT_START_H:
                key = (rec, site, t.date(), "night")
            elif t.hour < NIGHT_END_H:
                key = (rec, site, (t - timedelta(days=1)).date(), "night")
            else:
                key = (rec, site, t.date(), "day")
            if key[3] in periods:
                hours.setdefault(key, set()).add(t_utc.floor("h"))
    lo, hi = pd.Timestamp(first).date(), pd.Timestamp(last).date()
    return [(rec, site, d, per, len(hs)) for (rec, site, d, per), hs in sorted(hours.items())
            if lo <= d <= hi and len(hs) >= min_hours]


def recorder_nights(first: str, last: str, min_hours: int):
    return [(rec, site, d, n) for rec, site, d, per, n in recorder_periods(first, last, min_hours, ("night",))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-night", required=True)
    ap.add_argument("--last-night", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--min-hours", type=int, default=6)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs", "month"))
    ap.add_argument("--only-recorder", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo-nights", action="store_true")
    ap.add_argument("--redo-incomplete", action="store_true",
                    help="re-run nights with fewer window files than hours*12+1; existing windows are skipped")
    ap.add_argument("--periods", default="night", help="comma list of night,day")
    ap.add_argument("--step-s", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="list the units and exit")
    ap.add_argument("--engine", default="window_blobs.py",
                    help="window analysis script passed to sweep.py (window_blobs2.py for the v2 pipeline test)")
    ap.add_argument("--linker", default="night_animals.py",
                    help="night linker script (night_animals2.py matches chirpers on either rate)")
    args = ap.parse_args()
    periods = tuple(p.strip() for p in args.periods.split(","))

    os.makedirs(args.out_root, exist_ok=True)
    log = open(os.path.join(args.out_root, "run_month.log"), "a")

    def say(msg: str):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    rn = recorder_periods(args.first_night, args.last_night, args.min_hours, periods)
    if args.only_recorder:
        rn = [x for x in rn if x[0] == args.only_recorder]
    if args.limit:
        rn = rn[:args.limit]
    per_hour = 3600 / args.step_s
    say(f"{len(rn)} recorder-periods ({', '.join(periods)}), {sum(x[4] for x in rn)} hours, "
         f"~{int(sum(x[4] for x in rn) * per_hour)} windows, jobs={args.jobs}, step {args.step_s:.0f} s")
    if args.dry_run:
        return
    t_start = time.time()
    for k, (rec, site, date, per, nh) in enumerate(rn):
        # the month run named night directories without a period suffix; keep that
        suffix = "" if (per == "night" and os.path.basename(args.out_root) == "month") else f"_{per}"
        d = os.path.join(args.out_root, f"{rec}_{site}_{date:%Y%m%d}{suffix}")
        os.makedirs(d, exist_ok=True)
        tag = f"_{per}"
        done = os.path.exists(os.path.join(d, f"night{tag}_tracks.csv"))
        n_have = len(glob.glob(os.path.join(d, f"*{tag}_animals.csv")))
        incomplete = n_have < nh * per_hour
        if done and not args.redo_nights and not (args.redo_incomplete and incomplete):
            say(f"[{k + 1}/{len(rn)}] {rec} {site} {date} {per} already done")
            continue
        t0 = time.time()
        if per == "night":
            start = f"{date} {NIGHT_START_H:02d}:00"
            end = f"{date + timedelta(days=1)} {NIGHT_END_H:02d}:00"
        else:
            start = f"{date} {NIGHT_END_H:02d}:00"
            end = f"{date} {NIGHT_START_H:02d}:00"
        p = subprocess.run([sys.executable, os.path.join(HERE, "sweep.py"), "--recorder", rec, "--site", site,
                            "--start", start, "--end", end, "--step-s", str(args.step_s), "--jobs", str(args.jobs),
                            "--tag", tag, "--no-plot", "--out-dir", d, "--engine", args.engine], capture_output=True, text=True)
        ok = sum(1 for l in p.stdout.splitlines() if "rc=0" in l)
        bad = sum(1 for l in p.stdout.splitlines() if "rc=" in l and "rc=0" not in l)
        n_files = len(glob.glob(os.path.join(d, f"*{tag}_animals.csv")))
        msg = f"[{k + 1}/{len(rn)}] {rec} {site} {date} {per} ({nh} h): {ok} ok, {bad} failed, {n_files} windows, {time.time() - t0:.0f} s"
        if args.linker == "none":
            # detection only (the v2 corpus run): the night is linked afterwards in batch
            # (night_global.py --all); mark the unit done with an empty tracks file so a
            # relaunch skips it
            if n_files >= 12:
                open(os.path.join(d, f"night{tag}_tracks.csv"), "a").close()
            msg += " | not linked (detection only)"
        elif n_files >= 12:
            q = subprocess.run([sys.executable, os.path.join(HERE, args.linker), d, "--tag", tag, "--min-windows", "12"],
                               capture_output=True, text=True)
            summary = [l for l in q.stdout.splitlines() if "night-animals" in l or "spanning" in l]
            msg += " | " + "; ".join(s[:90] for s in summary[:2])
            if q.returncode != 0:
                msg += " | night_animals FAILED: " + q.stderr.strip().splitlines()[-1][:120]
            wa = os.path.join(d, f"night{tag}_window_animals.csv")
            if os.path.exists(wa):
                r = subprocess.run([sys.executable, os.path.join(HERE, "carrier_vs_rate.py"), wa, "--carrier", "F_fix",
                                    "--min-windows", "24"], capture_output=True, text=True)
                if r.returncode != 0:
                    msg += " | carrier_vs_rate: " + r.stderr.strip().splitlines()[-1][:100]
        say(msg)
    say(f"finished in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
