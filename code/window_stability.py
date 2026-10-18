"""How stable is the per-window animal list across overlapping windows?

Reads the *_animals.csv written by window_blobs.py for a sweep of window
centres through one recorder-night, matches animals between consecutive
windows on fundamental band overlap and rate, and reports how many animals
persist, how many appear once, and whether a persisting animal keeps its
components. Drift over an hour (~-19 Hz carrier, ~-0.4 Hz rate) is well inside
the matching tolerances, so this measures the method, not the animals.

Usage:
    python analysis/callers/window_stability.py analysis/callers/runs/BARLT_00008662_House_20240912_20*_animals.csv
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R_TOL = 0.015       # rate match between consecutive windows; a label scatters ~0.5-1% for one animal
F_PAD = 100.0       # Hz slack on fundamental band overlap


def load(paths):
    frames = []
    for p in sorted(paths):
        m = re.search(r"_(\d{8})_(\d{4})(?:_[^_/]+)?_animals\.csv$", p)
        t = pd.Timestamp(f"{m.group(1)} {m.group(2)[:2]}:{m.group(2)[2:]}")
        A = pd.read_csv(p)
        A["t"] = t
        A["window"] = os.path.basename(p).replace("_animals.csv", "")
        frames.append(A)
    return pd.concat(frames, ignore_index=True)


def _bands(row) -> list[tuple[float, float]]:
    out = []
    for c in str(row.components).split(";"):
        if "-" in c:
            lo, hi = c.split("-")
            out.append((float(lo), float(hi)))
    return out or [(row.F_lo, row.F_hi)]


def match(prev: pd.DataFrame, cur: pd.DataFrame):
    """Greedy one-to-one matching of animals between two windows: same rate
    within R_TOL and ANY component band overlapping within F_PAD -- which
    component happens to be lowest flickers as faint ones come and go, and that
    is not the animal changing. Strongest first."""
    pairs = []
    for i, a in cur.iterrows():
        for j, b in prev.iterrows():
            if abs(a.R / b.R - 1) >= R_TOL:
                continue
            if any(lo1 <= hi2 + F_PAD and lo2 <= hi1 + F_PAD
                   for lo1, hi1 in _bands(a) for lo2, hi2 in _bands(b)):
                pairs.append((min(a.depth, b.depth), i, j))
    used_i, used_j, out = set(), set(), {}
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_i and j not in used_j:
            out[i] = j
            used_i.add(i)
            used_j.add(j)
    return out


def main() -> None:
    paths = sys.argv[1:]
    A = load(paths)
    windows = sorted(A.t.unique())
    A["track"] = -1
    next_track = 0
    prev = None
    for t in windows:
        cur = A[A.t == t]
        if prev is not None:
            m = match(prev, cur)
            for i, j in m.items():
                A.loc[i, "track"] = A.loc[j, "track"]
        for i in cur.index:
            if A.loc[i, "track"] < 0:
                A.loc[i, "track"] = next_track
                next_track += 1
        prev = A[A.t == t]

    per_w = A.groupby("t").agg(n=("animal", "size"), n_multi=("n_components", lambda s: (s > 1).sum()))
    tracks = A.groupby("track").agg(
        n_windows=("t", "nunique"), F_lo=("F_lo", "median"), F_hi=("F_hi", "median"), R=("R", "median"),
        R_sd=("R", "std"), depth=("depth", "median"), comp_min=("n_components", "min"),
        comp_max=("n_components", "max"), on_frac=("on_frac", "median"),
        t0=("t", "min"), t1=("t", "max")).sort_values("n_windows", ascending=False)
    n_w = len(windows)
    step_s = (windows[1] - windows[0]).total_seconds() if n_w > 1 else np.nan

    print(f"{n_w} windows, step {step_s:.0f} s; animals per window median {per_w.n.median():.0f} "
          f"(range {per_w.n.min()}-{per_w.n.max()})")
    print(f"{len(tracks)} tracks; {(tracks.n_windows == 1).sum()} seen once, "
          f"{(tracks.n_windows >= n_w * 0.5).sum()} in >= half the windows, "
          f"{(tracks.n_windows >= n_w * 0.9).sum()} in >= 90%")
    # consecutive-window persistence: of the animals in window k, how many are matched in k+1
    cont = []
    for a, b in zip(windows[:-1], windows[1:]):
        ta, tb = set(A[A.t == a].track), set(A[A.t == b].track)
        cont.append(len(ta & tb) / max(len(ta), 1))
    print(f"fraction of a window's animals present in the next window: median {np.median(cont):.2f}, "
          f"min {np.min(cont):.2f}")
    print("persistence by modulation depth of the track (median over its windows):")
    for lo, hi in ((0, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 1.1)):
        sel = tracks[(tracks.depth >= lo) & (tracks.depth < hi)]
        if len(sel):
            print(f"   depth {lo:.2f}-{hi:.2f}: {len(sel):3d} tracks, {(sel.n_windows == 1).mean():.0%} seen once, "
                  f"median {sel.n_windows.median():.0f} of {n_w} windows")
    print(f"component count stability on tracks in >= 5 windows: "
          f"{(tracks[tracks.n_windows >= 5].comp_min == tracks[tracks.n_windows >= 5].comp_max).mean():.0%} constant")
    pd.set_option("display.width", 200)
    print(tracks.head(30).to_string(float_format=lambda x: f"{x:.2f}"))

    stem = os.path.commonprefix([os.path.basename(p) for p in paths]).rstrip("_")
    out = os.path.join(os.path.dirname(paths[0]), f"stability_{stem}")
    tracks.to_csv(out + "_tracks.csv", float_format="%.3f")
    A.to_csv(out + "_animals.csv", index=False, float_format="%.3f")

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw=dict(height_ratios=[4, 1]))
    ax = axes[0]
    cmap = plt.cm.turbo
    Rn = (A.R - A.R.min()) / max(A.R.max() - A.R.min(), 1e-9)
    for tr, g in A.groupby("track"):
        g = g.sort_values("t")
        col = cmap(float(Rn[g.index].median()))
        ax.plot(g.t, (g.F_lo + g.F_hi) / 2, "-", color=col, lw=0.8 if len(g) > 1 else 0, alpha=0.7)
        ax.scatter(g.t, (g.F_lo + g.F_hi) / 2, s=8 + 60 * g.depth, color=[col], edgecolors="k", lw=0.3,
                   marker="o" if len(g) > 1 else "x")
        if len(g) >= max(3, n_w * 0.3):
            ax.text(g.t.iloc[-1], (g.F_lo.iloc[-1] + g.F_hi.iloc[-1]) / 2, f" {g.R.median():.1f} Hz",
                    fontsize=7, va="center")
    ax.set_yscale("log")
    ax.set_ylabel("fundamental band centre (Hz)")
    ax.set_title(f"{stem}: animals per window (marker size = depth, colour = rate, x = seen in one window only); "
                 f"{len(tracks)} tracks over {n_w} windows")
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(per_w.index, per_w.n, "k.-", label="animals")
    ax.plot(per_w.index, per_w.n_multi, "b.-", label=">1 component")
    ax.set_ylabel("count")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out + ".png", dpi=110, bbox_inches="tight")
    print(f"wrote {out}.png / _tracks.csv / _animals.csv")


if __name__ == "__main__":
    main()
