"""Syllable shape by band: for each split type, the fine folds of its candidate bands side by side (band_folds.py cache):
per band the four highest-contrast exemplar folds and the pixel-wise median, then the per-window shape descriptors
(glide and curvature as % of carrier, duty, bandwidth) as strips, one color per band. Exploratory (2026-10-15).

Usage:
  python analysis/callers/band_shape_figure.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9bands \
      --groups 5:36011,36012 6:31011,31012,31013 9:34011,34012 16:54011,54012,54013 23:59011,59012 \
      --out analysis/callers/runs/figures_paper/exploratory/band_grids/band_shapes
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_EX = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9bands"); ap.add_argument("--cache-name", default="type_reference_bands")
    ap.add_argument("--groups", nargs="+", required=True, help="type:streak,streak,...") ; ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache = os.path.join(args.root, args.cache_name)
    ids = pd.read_csv(os.path.join(args.root, f"paper_ids{args.suffix}.csv")).set_index("streak").paper_id.to_dict()
    groups = [(g.split(":")[0], [int(x) for x in g.split(":")[1].split(",")]) for g in args.groups]
    ncol = max(len(b) for _, b in groups) * (N_EX + 1) + 3
    fig, axs = plt.subplots(len(groups), ncol, figsize=(1.15 * ncol, 1.5 * len(groups) + 0.6))
    cols = plt.get_cmap("tab10"); rows = []
    for gi, (t, bands) in enumerate(groups):
        k = 0; D_all = []
        for bi, sid in enumerate(bands):
            fp, wp = os.path.join(cache, f"folds_{sid}.npy"), os.path.join(cache, f"windows_{sid}.csv")
            if not os.path.exists(fp):
                for _ in range(N_EX + 1):
                    axs[gi, k].set_axis_off(); k += 1
                continue
            IM = np.load(fp); D = pd.read_csv(wp); D["band"] = str(ids.get(sid, sid)); D_all.append(D)
            order = np.argsort(-D.contrast.to_numpy())
            for e in range(N_EX):
                ax = axs[gi, k]; k += 1
                if e < len(order):
                    im = IM[order[e]]; ax.imshow(np.concatenate([im, im], axis=1), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=float(np.percentile(im, 99.5)), interpolation="nearest")
                    ax.set_title(f"{ids.get(sid, sid)}  {D.R.iloc[order[e]]:.0f} Hz", fontsize=6, color=cols(bi), pad=1.5)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(cols(bi)); sp.set_linewidth(1.2)
            ax = axs[gi, k]; k += 1
            med = np.median(IM, axis=0); ax.imshow(np.concatenate([med, med], axis=1), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=float(np.percentile(med, 99.5)), interpolation="nearest")
            ax.set_title(f"{ids.get(sid, sid)} median of {len(IM)}", fontsize=6, color=cols(bi), fontweight="bold", pad=1.5); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(cols(bi)); sp.set_linewidth(1.2)
            rows.append(dict(type=t, band=ids.get(sid, sid), n_folds=len(IM), glide_pct=100 * (D.glide_hz / D.F).median(), curve_pct=100 * (D.curve_hz / D.F).median(), duty=D.duty.median(), bw_hz=D.bw_hz.median(), R=D.R.median(), T_contrast=D.contrast.median()))
        while k < ncol - 3:
            axs[gi, k].set_axis_off(); k += 1
        if D_all:
            D = pd.concat(D_all)
            for m, (col, nm) in enumerate(((lambda d: 100 * d.glide_hz / d.F, "glide (% of carrier)"), (lambda d: 100 * d.curve_hz / d.F, "curvature (% of carrier)"), (lambda d: d.duty, "duty"))):
                ax = axs[gi, ncol - 3 + m]
                for bi, sid in enumerate(bands):
                    d = D[D.band == str(ids.get(sid, sid))]
                    if len(d):
                        y = col(d); ax.scatter(np.full(len(y), bi) + np.random.default_rng(0).uniform(-0.18, 0.18, len(y)), y, s=5, color=cols(bi), alpha=0.7, lw=0)
                        ax.hlines(np.median(y), bi - 0.3, bi + 0.3, color="k", lw=1)
                ax.set_xticks(range(len(bands))); ax.set_xticklabels([str(ids.get(s, s)) for s in bands], fontsize=6); ax.tick_params(labelsize=6); ax.set_title(nm, fontsize=6.5, pad=2)
                if m < 2:
                    ax.axhline(0, color="0.5", lw=0.5)
        axs[gi, 0].set_ylabel(f"type {t}", fontsize=8, fontweight="bold")
    fig.suptitle("syllable shape by candidate band: four highest-contrast exemplar folds and the median (two periods, +/- 8 % of carrier), then the per-window glide, curvature and duty", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(args.out + ".png", dpi=120); fig.savefig(args.out + ".pdf")
    T = pd.DataFrame(rows); T.to_csv(args.out + ".csv", index=False, float_format="%.3f"); print(T.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()
