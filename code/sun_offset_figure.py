"""Exploratory: each type's singing density around sunset and sunrise (MBE 2026-10-17: "for each type, density of
detections relative to sunrise and sunset, +/- 2 hours in either direction").

For every recorder-period (night and day units) the local sunset and sunrise of its date are computed (NOAA formula,
diel_pattern_figure.sun_times); every recorded five-minute window and every window-detection of a typed track is placed
at its offset from sunset and from sunrise; per type the density is detected windows per recorded window in 10-minute
offset bins over +/- 2 h, with the recorded windows counted only on the recorder-periods where the type sang (so the
density is relative to the type's own nights). Trill types take their night AND day tracks (the day tracks placed on
night lines); chirp types their chirp-plane tracks.

Writes <out>.png/.pdf (one panel per type: sunset half left, sunrise half right) and <out>.csv (the binned densities).

Usage:
  python analysis/callers/sun_offset_figure.py analysis/callers/runs/v5/year_nov24_nov25 --suffix _v9 \
      --roots analysis/callers/runs/v5/y2024 analysis/callers/runs/v5/y2025 --jobs 10 \
      --out analysis/callers/runs/figures_paper/exploratory/sun_offset_density
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ids  # noqa: E402
from diel_pattern_figure import LAT, LON, TZ, sun_times  # noqa: E402

BIN_MIN, SPAN_H = 10, 2.0
EDGES = np.arange(-SPAN_H * 60, SPAN_H * 60 + BIN_MIN, BIN_MIN)       # minutes
_TYPES = {}


def _unit(args):
    d, period, tracks_types = args                                   # tracks_types: {track: type} for this unit
    m = re.search(r"_(\d{8})_" + period + "$", d)
    date = pd.Timestamp(m.group(1))
    rise, sset = sun_times(LAT, LON, [date])[0]
    # sunset of the unit's date; sunrise of the following morning for night units, the same morning for day units
    base = pd.Timestamp(date, tz=TZ)
    t_set = base + pd.Timedelta(hours=sset); t_rise = base + pd.Timedelta(hours=rise) + (pd.Timedelta(days=1) if period == "night" else pd.Timedelta(0))
    rec = []
    for fn in os.listdir(d):
        mm = re.match(r"BARLT_\d+_[A-Za-z0-9]+_(\d{8})_(\d{2})(\d{2})_" + period + r"\.csv$", fn)
        if mm:
            rec.append(pd.Timestamp(f"{mm.group(1)[:4]}-{mm.group(1)[4:6]}-{mm.group(1)[6:]} {mm.group(2)}:{mm.group(3)}", tz=TZ))   # the file name carries the window's own date
    rec = pd.DatetimeIndex(rec)
    out = dict(rec_set=((rec - t_set).total_seconds() / 60).to_numpy(), rec_rise=((rec - t_rise).total_seconds() / 60).to_numpy(), det={})
    p = os.path.join(d, f"night_{period}_global_window_animals.csv")
    if tracks_types and os.path.exists(p):
        W = pd.read_csv(p, usecols=["t", "track"]); W = W[W.track.isin(tracks_types)]
        if len(W):
            t = pd.to_datetime(W.t, utc=True).dt.tz_convert(TZ)
            W = W.assign(off_set=(t - t_set).dt.total_seconds() / 60, off_rise=(t - t_rise).dt.total_seconds() / 60, type=W.track.map(tracks_types))
            for ty, g in W.groupby("type"):
                out["det"][ty] = (g.off_set.to_numpy(), g.off_rise.to_numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--out", required=True)
    args = ap.parse_args(); root = args.root
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0); S = S[(S.n >= 40) & (S.form == "trill")]; ids = paper_ids.load(root, args.suffix)
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv")); D = pd.read_csv(os.path.join(root, f"animal_days_streaks{args.suffix}.csv"))
    Sc = pd.read_csv(os.path.join(root, "chirp", "streaks_coarse_c2.csv"), index_col=0); Sc = Sc[Sc.n >= 40]
    Ac = pd.read_csv(os.path.join(root, "chirp", "animal_nights_streaks_c2.csv")); cid = pd.read_csv(os.path.join(root, "chirp", "paper_ids_chirp.csv")).set_index("streak").paper_id
    an_type = {}
    for r in A[A.streak.isin(S.index)].itertuples(): an_type[r.an] = str(ids[r.streak])
    for r in D[D.streak.isin(S.index)].itertuples(): an_type[r.an] = str(ids[r.streak])
    for r in Ac[Ac.streak.isin(Sc.index)].itertuples(): an_type[r.an] = str(cid[r.streak])
    by_unit = {}
    for an, ty in an_type.items():
        unit, tr = an.rsplit(":", 1); by_unit.setdefault(unit, {})[int(tr)] = ty
    jobs = []
    for r in args.roots:
        for d in sorted(glob.glob(os.path.join(r, "BARLT_*_night")) + glob.glob(os.path.join(r, "BARLT_*_day"))):
            unit = os.path.basename(d); period = "night" if unit.endswith("_night") else "day"
            jobs.append((d, period, by_unit.get(unit, {})))
    print(f"{len(jobs):,} units, {len(an_type):,} typed singer-periods", flush=True)
    with Pool(args.jobs) as pool:
        res = pool.map(_unit, jobs, chunksize=20)
    types = [str(ids[s]) for s in sorted(S.index, key=lambda s: ids[s])] + [str(cid[s]) for s in sorted(Sc.index, key=lambda s: int(str(cid[s])[1:]))]
    nb = len(EDGES) - 1
    det = {ty: (np.zeros(nb), np.zeros(nb)) for ty in types}; rec = {ty: (np.zeros(nb), np.zeros(nb)) for ty in types}
    for o in res:
        if not o["det"]:
            continue
        hs = np.histogram(o["rec_set"], EDGES)[0]; hr = np.histogram(o["rec_rise"], EDGES)[0]
        for ty, (ds, dr) in o["det"].items():
            det[ty][0][:] += np.histogram(ds, EDGES)[0]; det[ty][1][:] += np.histogram(dr, EDGES)[0]
            rec[ty][0][:] += hs; rec[ty][1][:] += hr
    rows = []
    for ty in types:
        for k, nm in ((0, "sunset"), (1, "sunrise")):
            dens = det[ty][k] / np.where(rec[ty][k] > 0, rec[ty][k], np.nan)
            for b in range(nb):
                rows.append(dict(type=ty, ref=nm, offset_min=(EDGES[b] + EDGES[b + 1]) / 2, detected=det[ty][k][b], recorded=rec[ty][k][b], density=dens[b]))
    T = pd.DataFrame(rows); T.to_csv(args.out + ".csv", index=False, float_format="%.4f")
    # the figure: 9 x 5 panels, each split sunset | sunrise
    ncol = 5; nrow = int(np.ceil(len(types) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 1.55 * nrow)); axs = axs.ravel()
    xc = (EDGES[:-1] + EDGES[1:]) / 2 / 60
    pos = {str(ids[s]): f"{S.loc[s, 'F25'] / 1000:.1f} kHz @ {S.loc[s, 'R25']:.0f} Hz" for s in S.index}; pos.update({str(cid[s]): f"{Sc.loc[s, 'F25'] / 1000:.1f} kHz @ chirps {Sc.loc[s, 'R25']:.1f}" for s in Sc.index})
    for i, ty in enumerate(types):
        ax = axs[i]; gap = 0.6
        for k, (col, x0) in enumerate((("tab:orange", 0.0), ("tab:blue", 2 * SPAN_H + gap))):
            dens = det[ty][k] / np.where(rec[ty][k] > 0, rec[ty][k], np.nan)
            ax.fill_between(x0 + xc + SPAN_H, 0, dens, color=col, alpha=0.6, lw=0); ax.plot(x0 + xc + SPAN_H, dens, color=col, lw=0.8)
            ax.axvline(x0 + SPAN_H, color="k", lw=0.6, ls="--")
        ax.set_xlim(-0.1, 4 * SPAN_H + gap + 0.1); ax.set_xticks([0, SPAN_H, 2 * SPAN_H, 2 * SPAN_H + gap, 3 * SPAN_H + gap, 4 * SPAN_H + gap]); ax.set_xticklabels(["-2", "set", "+2", "-2", "rise", "+2"], fontsize=6)
        ax.tick_params(labelsize=6, length=2); ax.set_title(f"{ty}  {pos[ty]}", fontsize=7, loc="left", pad=2); ax.grid(alpha=0.2, axis="y")
        ymax = np.nanmax(np.concatenate([det[ty][0] / np.where(rec[ty][0] > 0, rec[ty][0], np.nan), det[ty][1] / np.where(rec[ty][1] > 0, rec[ty][1], np.nan)]) + [1e-6]); ax.set_ylim(0, ymax * 1.1)
    for ax in axs[len(types):]:
        ax.set_axis_off()
    fig.text(0.005, 0.5, "singers of the type tracked at once per recorded window (on the type's own recorder-periods)", rotation=90, va="center", fontsize=8)
    fig.suptitle("Singing around sunset (orange, left halves) and sunrise (blue, right halves), +/- 2 h in 10-minute bins, per type", fontsize=10)
    fig.tight_layout(rect=(0.02, 0, 1, 0.98)); fig.savefig(args.out + ".png", dpi=120); fig.savefig(args.out + ".pdf")
    print(f"wrote {args.out}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
