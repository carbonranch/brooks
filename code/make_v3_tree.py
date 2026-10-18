"""Build the v3 run tree: the v2 detections reused by symlink, with the night links to be redone on the chirp
re-pass (METHOD 19 AG/AH). Nothing under runs/v2 is modified.

For each run root (runs/v2/y2024, y2025) a v3 root (runs/v3/yYYYY) gets:
  - root-level files (fixed_rate_lines.csv, weather_hourly_hrrr.csv, logs) symlinked;
  - every unit as a real directory holding symlinks to every file of the v2 unit; DAY units keep their link
    outputs (no chirp re-pass for days); NIGHT units leave out the global link
    outputs (night_night_global_tracks / _window_animals / _drift.csv), which night_global.py rewrites here with
    BROOKS_SYLV2_LINK=1 (the sidecar night_night_global_window_animals_sylv2.csv is symlinked in too).

Usage:
    python analysis/callers/make_v3_tree.py [--roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025] [--out analysis/callers/runs/v3]
"""
from __future__ import annotations

import argparse
import glob
import os
import time

REDO = ("night_night_global_tracks.csv", "night_night_global_window_animals.csv", "night_night_global_drift.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["analysis/callers/runs/v2/y2024", "analysis/callers/runs/v2/y2025"])
    ap.add_argument("--out", default="analysis/callers/runs/v3")
    ap.add_argument("--units", nargs="*", default=None, help="only these unit names (for a test)")
    args = ap.parse_args()
    t0 = time.time(); n_links = 0
    for root in args.roots:
        root = os.path.abspath(root); year = os.path.basename(root)
        dst_root = os.path.abspath(os.path.join(args.out, year)); os.makedirs(dst_root, exist_ok=True)
        for f in os.listdir(root):
            src = os.path.join(root, f); dst = os.path.join(dst_root, f)
            if f.startswith("BARLT_"):
                if args.units is not None and f not in args.units:
                    continue
                os.makedirs(dst, exist_ok=True)                 # a real directory for every unit, so nothing can write into v2
                redo = REDO if f.endswith("_night") else ()     # day links stand (no chirp re-pass for days)
                for g in os.listdir(src):
                    if g in redo:
                        continue
                    d2 = os.path.join(dst, g)
                    if not os.path.lexists(d2):
                        os.symlink(os.path.join(src, g), d2); n_links += 1
            elif os.path.isfile(src) and not os.path.lexists(dst):
                os.symlink(src, dst); n_links += 1
        print(f"{year}: done, {n_links:,} links so far, {time.time() - t0:.0f} s", flush=True)
    print(f"v3 tree at {os.path.abspath(args.out)}: {n_links:,} symlinks in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
