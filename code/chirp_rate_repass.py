"""Targeted re-pass of the CHIRP RATE over the year's windows (METHOD 19 AJ), without rerunning the detector.

The detector's sub-harmonic check divided a peak's rate by 2 or 3 once; a slow rhythm whose map peak sits on its 6th
or 9th harmonic was left at 2x or 3x its true chirp rate (type 7151: 2.2 Hz chirpers read at 5-7 Hz in a third of
their windows). window_blobs2.subharmonic_rate now iterates. Here every window-animal with a chirp rate (the slow
fundamental of a chirp-only animal, or the bout rhythm of a fast one) is re-read: the window's STFT, the animal's
band rows, its on-runs at the old chirp rate, then the iterated sub-harmonic check and a refinement at the new rate.
Results go to a sidecar beside the night's tables, night_night_global_window_animals_crv2.csv (t, animal, C_old,
C_new, m, depth_old, depth_new); night_global applies it before linking when BROOKS_SYLV2_LINK=1. Resumable (units
with a sidecar are skipped), parallel over units.

Usage:
    python analysis/callers/chirp_rate_repass.py --roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --jobs 16
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs2 as W2  # noqa: E402

SIDECAR = "night_night_global_window_animals_crv2.csv"
UNIT_RE = re.compile(r"(BARLT_\d+)_([A-Za-z0-9]+)_(\d{8})_(night|day)$")
COLS = ["t", "animal", "C_old", "C_new", "m", "depth_old", "depth_new", "note"]


def stft_power(sig, sr):
    f, t, Z = scipy.signal.stft(sig, fs=sr, window="hann", nperseg=W2.NPERSEG, noverlap=W2.NPERSEG - W2.HOP, boundary=None, padded=False)
    keep = (f >= W2.FMIN_HZ)
    return f[keep], np.abs(Z[keep]) ** 2, W2.HOP / sr


def depth(P, r_lo, r_hi, runs, dt, R):
    F = W2.fold_runs(P, r_lo, r_hi, runs, dt, R); d, _, _ = W2.row_phase(F); return float(np.mean(d))


def repass_unit(unit_dir: str):
    t_start = time.time()
    m = UNIT_RE.match(os.path.basename(unit_dir)); rec, site, tag = m.group(1), m.group(2), m.group(4)
    p = os.path.join(unit_dir, f"night_{tag}_global_window_animals.csv")
    W = pd.read_csv(p); W["t"] = pd.to_datetime(W.t, utc=True).dt.tz_convert(W2.TZ)
    sel = W.chirp_rate.notna()
    rows = []
    for t, g in W[sel].groupby("t"):
        try:
            sig, sr, t0, _ = W2.read_window(rec, site, t, 300.0)
        except Exception as e:  # noqa: BLE001
            rows += [dict(t=t.isoformat(), animal=int(a.animal), C_old=a.chirp_rate, C_new=a.chirp_rate, m=1, depth_old=np.nan, depth_new=np.nan, note=f"no audio: {type(e).__name__}") for a in g.itertuples()]
            continue
        f, P, dt = stft_power(sig, sr)
        for a in g.itertuples():
            r_lo = int(np.searchsorted(f, a.F_lo)); r_hi = min(len(f) - 1, max(r_lo, int(np.searchsorted(f, a.F_hi))))
            C0 = float(a.chirp_rate)
            runs, on_frac = W2.on_periods(P, r_lo, r_hi, dt, C0)
            if on_frac >= W2.ON_FRAC_CONT or not runs:
                runs = [W2.best_segment(P, r_lo, r_hi, dt, C0, W2.FOLD_S)]
            else:
                runs = sorted(runs, key=lambda sl: -(sl.stop - sl.start)); tot, kept = 0, []
                for sl in runs:
                    kept.append(sl); tot += sl.stop - sl.start
                    if tot * dt >= W2.FOLD_S:
                        break
                runs = sorted(kept, key=lambda sl: sl.start)
            d0 = depth(P, r_lo, r_hi, runs, dt, C0)
            C1, mm = W2.subharmonic_rate(P, r_lo, r_hi, runs, dt, C0)
            if mm > 1 and C1 >= W2.AM_LO:
                C1 = W2.refine_rate_runs(P, r_lo, r_hi, runs, dt, C1)
            d1 = depth(P, r_lo, r_hi, runs, dt, C1) if mm > 1 else d0
            rows.append(dict(t=t.isoformat(), animal=int(a.animal), C_old=round(C0, 4), C_new=round(float(C1), 4), m=int(mm), depth_old=round(d0, 4), depth_new=round(d1, 4), note=""))
    D = pd.DataFrame(rows, columns=COLS); D.to_csv(os.path.join(unit_dir, SIDECAR), index=False)
    return os.path.basename(unit_dir), int(sel.sum()), int((D.m > 1).sum()), time.time() - t_start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--units", nargs="*", default=None); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--period", default="night")
    args = ap.parse_args()
    units = []
    for r in args.roots:
        units += sorted(glob.glob(os.path.join(r, f"BARLT_*_{args.period}")))
    if args.units:
        units = [u for u in units if os.path.basename(u) in set(args.units)]
    units = [u for u in units if os.path.exists(os.path.join(u, f"night_{args.period}_global_window_animals.csv")) and not os.path.exists(os.path.join(u, SIDECAR))]
    if args.limit:
        units = units[:args.limit]
    print(f"{len(units)} units to do", flush=True)
    t0 = time.time(); done = 0; n_an = 0; n_ch = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(repass_unit, u): u for u in units}
        for fu in as_completed(futs):
            try:
                name, n, nc, dt_ = fu.result(); n_an += n; n_ch += nc
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {os.path.basename(futs[fu])}: {type(e).__name__}: {str(e)[:100]}", flush=True); continue
            done += 1
            if done % 50 == 0 or done == len(units):
                el = time.time() - t0
                print(f"  {done}/{len(units)} units, {n_an:,} chirp animals, {n_ch:,} changed ({100 * n_ch / max(1, n_an):.1f} %); {el / 60:.1f} min, ~{el / done * (len(units) - done) / 3600:.1f} h left", flush=True)
    print("done")


if __name__ == "__main__":
    main()
