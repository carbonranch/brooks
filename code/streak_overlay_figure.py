"""The streak definition drawn over the raw data, in the ordinary plane.

Left: singer-hours (raw rate and carrier, coloured by temperature), with
each streak's extent drawn as a thin outline: the parallelogram spanned by
its singer-hours' 5-95 % range along the coupling line and +/- 2 sd across
it, plus a numeric tag at the centre. Right: the same points coloured by
streak label (grey = unassigned), so the reader can see whether the labels
follow the gaps in the data or cut through them.

Usage:
    python analysis/callers/streak_overlay_figure.py analysis/callers/runs/month_v2 --suffix _v4 --coupling 0.55 --out streak_overlay_v4
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streak_story_figure import style

BR, BF = 0.0154, 0.0046


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--suffix", default="_v4")
    ap.add_argument("--coupling", type=float, default=0.55)
    ap.add_argument("--coupling-hf", type=float, default=0.25)
    ap.add_argument("--hf-split", type=float, default=10000.0)
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--zoom", type=float, nargs=4, default=None, help="R_lo R_hi F_lo F_hi for an extra zoomed pair of panels")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv")).drop(columns=["streak"]).merge(A[["an", "streak"]], on="an")
    big = [s for s in S.index if S.loc[s, "n"] >= args.min_n]
    P["big"] = P.streak.isin(big)

    def outline(g):
        k = args.coupling if g.F.median() < args.hf_split else args.coupling_hf
        u = np.array([1.0, k]) / np.hypot(1.0, k)
        nrm = np.array([-u[1], u[0]])
        Z = np.column_stack([np.log(g.R), np.log(g.F)])
        c = Z.mean(axis=0)
        d = Z - c
        al, pp = d @ u, d @ nrm
        a0, a1 = np.percentile(al, 5), np.percentile(al, 95)
        w = 2 * pp.std()
        corners = [c + a0 * u - w * nrm, c + a1 * u - w * nrm, c + a1 * u + w * nrm, c + a0 * u + w * nrm, c + a0 * u - w * nrm]
        return np.exp(np.array(corners))

    def draw(axL, axR, sub, title_extra=""):
        sub = sub.sort_values("T")
        sc = axL.scatter(sub.R, sub.F, s=1.5, c=sub["T"], cmap="jet", vmin=12, vmax=32, alpha=0.35, edgecolors="none")
        for sid in big:
            g = sub[sub.streak == sid]
            if len(g) < 20:
                continue
            poly = outline(g)
            axL.plot(poly[:, 0], poly[:, 1], "-", color="k", lw=0.7, alpha=0.9)
            axL.annotate(str(sid), (np.exp(np.log(g.R).mean()), np.exp(np.log(g.F).mean())), fontsize=6, ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        style(axL, "syllable rate (Hz)", "carrier frequency (kHz)")
        axL.set_title(f"(A)  {len(sub):,} singer-hours coloured by temperature; outlines = each streak's 5–95 % extent along the line, ±2 sd across{title_extra}", fontsize=8.5, loc="left")
        cols = plt.cm.tab20(np.linspace(0, 1, 20))
        rng = np.random.default_rng(1)
        order = {sid: i for i, sid in enumerate(rng.permutation(big))}
        other = sub[~sub.big]
        axR.scatter(other.R, other.F, s=1.2, color="0.8", alpha=0.5, edgecolors="none")
        for sid in big:
            g = sub[sub.streak == sid]
            axR.scatter(g.R, g.F, s=1.5, color=cols[order[sid] % 20], alpha=0.5, edgecolors="none")
        style(axR, "syllable rate (Hz)", "carrier frequency (kHz)")
        axR.set_title(f"(B)  the same points coloured by streak label ({len(big)} streaks with ≥ {args.min_n} singer-nights; grey = unassigned)", fontsize=8.5, loc="left")
        return sc

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    sc = draw(axes[0], axes[1], P)
    fig.colorbar(sc, ax=axes[0], fraction=0.04, pad=0.02).set_label("HRRR air temperature at the hour (°C)")
    fig.tight_layout()
    out = args.out or os.path.join(root, f"streak_overlay{args.suffix}")
    if not os.path.dirname(out):
        out = os.path.join(root, out)
    fig.savefig(out + ".png", dpi=140); fig.savefig(out + ".pdf")
    print(f"wrote {out}.png / .pdf")
    if args.zoom:
        r0, r1, f0, f1 = args.zoom
        sub = P[P.R.between(r0, r1) & P.F.between(f0, f1)]
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6.6))
        sc2 = draw(axes2[0], axes2[1], sub, title_extra=" (zoom)")
        for ax in axes2:
            ax.set_xlim(r0, r1); ax.set_ylim(f0, f1)
            ax.set_yticks([]); ax.set_yticks([], minor=True)
            yt = [t for t in (2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000) if f0 <= t <= f1]
            ax.set_yticks(yt); ax.set_yticklabels([f"{t / 1000:g}" for t in yt])
            xt = [t for t in (10, 15, 20, 25, 30, 40, 50, 60, 80, 100) if r0 <= t <= r1]
            ax.set_xticks(xt); ax.set_xticklabels([str(t) for t in xt])
        fig2.colorbar(sc2, ax=axes2[0], fraction=0.04, pad=0.02).set_label("HRRR air temperature at the hour (°C)")
        fig2.tight_layout()
        fig2.savefig(out + "_zoom.png", dpi=140); fig2.savefig(out + "_zoom.pdf")
        print(f"wrote {out}_zoom.png / .pdf")


if __name__ == "__main__":
    main()
