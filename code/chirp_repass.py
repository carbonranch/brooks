"""Targeted re-pass of the v2 chirper handling over the year's windows, WITHOUT touching the corpus run.

For every night unit, every window that holds a chirp-fundamental animal (chirp_rate set, no bout_rate: the slow
peak is the fundamental) is read again, its STFT recomputed to find the animal's longest on-run, and for each such
animal the syllable rate is re-read with the TIME-DOMAIN reader chirp_channel.syllable_rate_from_chirps (chirps as
events in the band's envelope over that run, capped at 60 s; pulses counted inside each); then the twin rule:
another animal in the same window whose band overlaps and whose rate is within 3 % of the new syllable rate (and
has no chirp rate) is marked merged into the chirper. Results go to a SIDECAR table beside the night's tables,
`night_night_global_window_animals_sylv2.csv`, with one row per chirp-fundamental animal: t, animal, the old and
new syllable rate, the reader's interval spread, pulses per chirp, chirp length, chirp count, and the merged twin's
animal id (or -1). The
original tables are not modified; a later relink can consume the sidecar (METHOD 19 AG).

Resumable: units with a sidecar are skipped. Parallel over units with --jobs.

Usage:
    python analysis/callers/chirp_repass.py --roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --jobs 6
    python analysis/callers/chirp_repass.py --roots ... --units BARLT_00008662_House_20250920_night   # one unit
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
from chirp_channel import syllable_rate_from_chirps  # noqa: E402

SIDECAR = "night_night_global_window_animals_sylv2.csv"
UNIT_RE = re.compile(r"(BARLT_\d+)_([A-Za-z0-9]+)_(\d{8})_night$")


def stft_power(sig, sr):
    f, t, Z = scipy.signal.stft(sig, fs=sr, window="hann", nperseg=W2.NPERSEG, noverlap=W2.NPERSEG - W2.HOP, boundary=None, padded=False)
    keep = (f >= W2.FMIN_HZ)
    return f[keep], np.abs(Z[keep]) ** 2, W2.HOP / sr


def repass_unit(unit_dir: str) -> tuple[str, int, int, float]:
    t_start = time.time()
    m = UNIT_RE.match(os.path.basename(unit_dir))
    rec, site, d8 = m.group(1), m.group(2), m.group(3)
    p = os.path.join(unit_dir, "night_night_global_window_animals.csv")
    W = pd.read_csv(p)
    W["t"] = pd.to_datetime(W.t, utc=True).dt.tz_convert(W2.TZ)
    cf = W.chirp_rate.notna() & W.bout_rate.isna()
    out_rows = []
    for t, g in W[cf].groupby("t"):
        try:
            sig, sr, t0, _ = W2.read_window(rec, site, t, 300.0)
        except Exception as e:  # noqa: BLE001
            for a in g.itertuples():
                out_rows.append(dict(t=t.isoformat(), animal=int(a.animal), R_old=a.R, R_new=np.nan, iqr_pct=np.nan, pulses_per_chirp=np.nan, chirp_ms=np.nan, n_chirps=0, run_s=np.nan, merged_twin=-1, note=f"no audio: {type(e).__name__}"))
            continue
        f, P, dt = stft_power(sig, sr)
        others_all = W[(W.t == t)]
        for a in g.itertuples():
            r_lo = int(np.searchsorted(f, a.F_lo)); r_hi = min(len(f) - 1, int(np.searchsorted(f, a.F_hi)))
            Rc = float(a.chirp_rate)
            runs, on_frac = W2.on_periods(P, r_lo, r_hi, dt, Rc)
            longest = max(runs, key=lambda sl: sl.stop - sl.start) if runs else W2.best_segment(P, r_lo, r_hi, dt, Rc, W2.FOLD_S)
            t_a = longest.start * dt; t_b = min(longest.stop * dt, t_a + 60.0)
            res = syllable_rate_from_chirps(sig, sr, float(a.F_lo), float(a.F_hi), t_a, t_b)
            Rs = res["Rs"]
            twin = -1
            if np.isfinite(Rs):
                o = others_all[(others_all.animal != a.animal) & others_all.R.notna() & others_all.chirp_rate.isna()
                               & (others_all.F_lo <= a.F_hi) & (others_all.F_hi >= a.F_lo)]
                o = o[(o.R / Rs - 1).abs() < 0.03]
                if len(o):
                    twin = int(o.sort_values("depth", ascending=False).animal.iloc[0])
            out_rows.append(dict(t=t.isoformat(), animal=int(a.animal), R_old=a.R, R_new=round(float(Rs), 3) if np.isfinite(Rs) else np.nan,
                                 iqr_pct=round(float(res["iqr_pct"]), 1) if np.isfinite(res["iqr_pct"]) else np.nan, pulses_per_chirp=res["pulses_per_chirp"],
                                 chirp_ms=round(float(res["chirp_ms"]), 1) if np.isfinite(res["chirp_ms"]) else np.nan, n_chirps=res["n_chirps"],
                                 run_s=round(t_b - t_a, 1), merged_twin=twin, note=""))
    D = pd.DataFrame(out_rows, columns=["t", "animal", "R_old", "R_new", "iqr_pct", "pulses_per_chirp", "chirp_ms", "n_chirps", "run_s", "merged_twin", "note"])
    D.to_csv(os.path.join(unit_dir, SIDECAR), index=False)
    return os.path.basename(unit_dir), int(cf.sum()), int(W.t[cf].nunique()), time.time() - t_start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--units", nargs="*", default=None, help="unit directory names to run (default: all without a sidecar)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    units = []
    for r in args.roots:
        units += sorted(glob.glob(os.path.join(r, "BARLT_*_night")))
    if args.units:
        units = [u for u in units if os.path.basename(u) in set(args.units)]
    units = [u for u in units if os.path.exists(os.path.join(u, "night_night_global_window_animals.csv")) and not os.path.exists(os.path.join(u, SIDECAR))]
    if args.limit:
        units = units[:args.limit]
    print(f"{len(units)} units to re-pass with {args.jobs} workers", flush=True)
    done, t0 = 0, time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(repass_unit, u): u for u in units}
        for fu in as_completed(futs):
            try:
                name, n_an, n_win, secs = fu.result()
                done += 1
                if done % 20 == 0 or done <= 5:
                    el = time.time() - t0
                    print(f"[{done}/{len(units)}] {name}: {n_an} chirp animals in {n_win} windows, {secs:.0f} s; elapsed {el / 60:.1f} min, eta {el / done * (len(units) - done) / 3600:.1f} h", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {os.path.basename(futs[fu])}: {type(e).__name__}: {e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
