"""Build a new run tree from an existing one, reusing every file by symlink except the global link outputs, which the
linker rewrites in the new tree (generalizes make_v3_tree.py: that one went v2 -> v3 and redid nights only).

For each source root (e.g. runs/v3/y2024, y2025) the new root (runs/<out>/yYYYY) gets root-level files symlinked and
every unit as a real directory of symlinks to the source unit's files, leaving out the link outputs
(night_<period>_global_tracks / _window_animals / _drift.csv) of the periods to redo; sidecars (_sylv2, _crv2) are
symlinked in. Symlinks to symlinks are resolved so the new tree points at the real files.

v4 (2026-10-15, METHOD 19 AV): the fixed-window carrier's pad made relative (BROOKS_FIX_PAD_REL), nights and days relinked.

Usage:
    python analysis/callers/make_run_tree.py --roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 \
        --out analysis/callers/runs/v4 --redo night day
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--redo", nargs="+", default=["night", "day"], choices=["night", "day"])
    ap.add_argument("--units", nargs="*", default=None, help="only these unit names (for a test)")
    args = ap.parse_args()
    redo_files = {p: (f"night_{p}_global_tracks.csv", f"night_{p}_global_window_animals.csv", f"night_{p}_global_drift.csv") for p in args.redo}
    t0 = time.time(); n_links = 0
    for root in args.roots:
        root = os.path.abspath(root); year = os.path.basename(root)
        dst_root = os.path.abspath(os.path.join(args.out, year)); os.makedirs(dst_root, exist_ok=True)
        for f in sorted(os.listdir(root)):
            src = os.path.join(root, f); dst = os.path.join(dst_root, f)
            if f.startswith("BARLT_"):
                if args.units is not None and f not in args.units:
                    continue
                os.makedirs(dst, exist_ok=True)
                period = f.rsplit("_", 1)[-1]
                skip = redo_files.get(period, ())
                for g in os.listdir(src):
                    if g in skip:
                        continue
                    d2 = os.path.join(dst, g)
                    if not os.path.lexists(d2):
                        os.symlink(os.path.realpath(os.path.join(src, g)), d2); n_links += 1
            elif (os.path.isfile(src) or os.path.islink(src)) and not os.path.lexists(dst):
                os.symlink(os.path.realpath(src), dst); n_links += 1
        print(f"{year}: done, {n_links:,} links so far, {time.time() - t0:.0f} s", flush=True)
    print(f"tree at {os.path.abspath(args.out)}: {n_links:,} symlinks in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
