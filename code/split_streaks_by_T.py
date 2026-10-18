"""Stage two of the streak definition: split a broad position cluster where
its singer-nights form more than one band in the position-vs-temperature
plane.

Stage one (build_streaks.py --along-tol 0.40 --perp-tol 0.08 --suffix _broad)
groups singer-nights that lie on one coupling line within a generous
tolerance along it. One species then forms a single tilted band in
(position along the line, temperature); two species sharing the line form
two parallel bands. Here, per broad streak:

1. the between-night slope of position on temperature is fitted robustly
   (Theil-Sen over singer-nights), so a minority band does not bend it;
2. each singer-night's position is moved to 25 degC with that slope;
3. Gaussian mixtures with 1, 2 and 3 components are fitted to the corrected
   positions (log-rate units) and compared by BIC; a split is accepted only
   if BIC improves by >= DBIC, every component holds >= MIN_N singer-nights
   and >= MIN_FRAC of the streak, and neighbouring components are separated
   by >= SEP_SD pooled standard deviations.

Streak ids become 100*broad + component (e.g. 6701, 6702); unsplit streaks
keep 100*broad + 1. Writes animal_nights_streaks<out>.csv and
streaks_coarse<out>.csv in the same format as build_streaks.py, plus a
figure of the corrected-position histograms with the fitted mixtures.

Usage:
    python analysis/callers/split_streaks_by_T.py analysis/callers/runs/month_v2 --in _broad --out _v3
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.mixture import GaussianMixture

BR, BF = 0.0154, 0.0046
MIN_N = 15
MIN_FRAC = 0.08
DBIC = 10.0
SEP_SD = 2.5
MAX_K = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--in", dest="suffix_in", default="_broad")
    ap.add_argument("--out", dest="suffix_out", default="_v3")
    ap.add_argument("--slope", type=float, default=None,
                    help="fixed between-night slope (%%/degC) for the 25 degC correction; default: each streak's own Theil-Sen slope, "
                         "which a second band can corrupt")
    ap.add_argument("--sep-sd", type=float, default=SEP_SD)
    ap.add_argument("--dbic", type=float, default=DBIC)
    ap.add_argument("--max-k", type=int, default=MAX_K)
    ap.add_argument("--sd-ratio", type=float, default=2.5, help="a component more than this many times wider than another is a tail, not a band")
    ap.add_argument("--coupling", type=float, default=None, help="line slope for the along-line coordinate; default bF/bR; use the ridges' slope (~0.55)")
    args = ap.parse_args()
    sep_sd, dbic_min, max_k = args.sep_sd, args.dbic, args.max_k
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix_in}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix_in}.csv"), index_col=0)
    k = args.coupling if args.coupling is not None else BF / BR
    u = np.array([1.0, k]) / np.hypot(1.0, k)
    Z = np.column_stack([np.log(A.R), np.log(A.F)])
    A["along"] = (Z @ u) / u[0]                      # log-rate equivalent along the common line

    A["streak2"] = -1
    rows, figs = [], []
    for sid in S.index:
        g = A[A.streak == sid]
        idx = g.index
        if len(g) < 2 * MIN_N:
            A.loc[idx, "streak2"] = 100 * sid + 1
            rows.append(dict(broad=sid, n=len(g), slope_pct_perC=np.nan, k_best=1, dbic=np.nan, split=False, note="too small to test"))
            continue
        # robust between-night slope, then corrected position
        if args.slope is not None:
            b = args.slope / 100
        elif g["T"].std() > 0.5:
            b = theilslopes(g.along.to_numpy(), g["T"].to_numpy())[0]
        else:
            b = BR
        x25 = (g.along + b * (25 - g["T"])).to_numpy().reshape(-1, 1)
        fits = {kk: GaussianMixture(kk, n_init=5, random_state=0).fit(x25) for kk in range(1, max_k + 1)}
        bic = {kk: fits[kk].bic(x25) for kk in fits}
        chosen, note = 1, ""
        for kk in range(2, max_k + 1):
            m = fits[kk]
            lab = m.predict(x25)
            counts = np.bincount(lab, minlength=kk)
            order = np.argsort(m.means_.ravel())
            mu = m.means_.ravel()[order]
            sd = np.sqrt(m.covariances_.ravel()[order])
            sep_ok = all((mu[i + 1] - mu[i]) >= sep_sd * np.sqrt((sd[i] ** 2 + sd[i + 1] ** 2) / 2) for i in range(kk - 1))
            size_ok = counts.min() >= MIN_N and counts.min() >= MIN_FRAC * len(g)
            width_ok = sd.max() / sd.min() <= args.sd_ratio
            if bic[chosen] - bic[kk] >= dbic_min and sep_ok and size_ok and width_ok:
                chosen = kk
            else:
                note = note or (f"k={kk}: " + ", ".join(w for w, ok in (("dBIC small", bic[chosen] - bic[kk] < dbic_min), ("components overlap", not sep_ok),
                                                                          ("component too small", not size_ok), ("component is a tail", not width_ok)) if ok))
        m = fits[chosen]
        order = np.argsort(m.means_.ravel())
        rank = np.empty(chosen, int)
        rank[order] = np.arange(chosen)
        comp = rank[m.predict(x25)]
        A.loc[idx, "streak2"] = 100 * sid + 1 + comp
        A.loc[idx, "x25"] = x25.ravel()
        rows.append(dict(broad=sid, n=len(g), slope_pct_perC=100 * b, k_best=chosen, dbic=bic[1] - bic[chosen] if chosen > 1 else bic[1] - bic[2],
                         split=chosen > 1, sizes=" / ".join(str(c) for c in np.bincount(comp, minlength=chosen)),
                         sep_pct=" / ".join(f"{100 * (m.means_.ravel()[order][i + 1] - m.means_.ravel()[order][i]):.0f}" for i in range(chosen - 1)),
                         note=note if chosen == 1 else ""))
        figs.append((sid, x25.ravel(), comp, m, order, b))

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))
    print(f"\n{len(S)} broad streaks -> {A.streak2.nunique()} after splitting ({R.split.sum()} split)")

    # ---- outputs in the build_streaks format
    A["streak"] = A.streak2
    g = A.groupby("streak")
    T = g.agg(n=("an", "size"), F25=("F25", "median"), F25_iqr=("F25", lambda s: s.quantile(.75) - s.quantile(.25)),
              R25=("R25", "median"), R25_iqr=("R25", lambda s: s.quantile(.75) - s.quantile(.25)),
              F_min=("F", "min"), F_max=("F", "max"), R_min=("R", "min"), R_max=("R", "max"),
              n_sites=("site", "nunique"), n_nights=("night", "nunique"), depth=("depth", "median"))
    T = T[(T.n >= MIN_N) & (T.index >= 0)].sort_values("F25")      # -1 = singer-nights outside any broad streak
    T.to_csv(os.path.join(root, f"streaks_coarse{args.suffix_out}.csv"), float_format="%.3f")
    A[["an", "recorder", "site", "night", "F", "R", "T", "F25", "R25", "depth", "streak", "type_corrected"]].to_csv(
        os.path.join(root, f"animal_nights_streaks{args.suffix_out}.csv"), index=False, float_format="%.3f")
    R.to_csv(os.path.join(root, f"streak_splits{args.suffix_out}.csv"), index=False, float_format="%.3f")
    print(f"{len(T)} streaks with >= {MIN_N} singer-nights; wrote streaks_coarse{args.suffix_out}.csv, animal_nights_streaks{args.suffix_out}.csv, streak_splits{args.suffix_out}.csv")

    # ---- figure: corrected-position histograms, split streaks first
    figs.sort(key=lambda t: (-(t[3].n_components > 1), -len(t[1])))
    show = figs[:20]
    ncol = 5
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow), squeeze=False)
    cols = ["tab:blue", "tab:red", "tab:green"]
    for ax, (sid, x, comp, m, order, b) in zip(axes.ravel(), show):
        xx = 100 * (x - np.median(x))
        bins = np.linspace(np.percentile(xx, 0.5) - 3, np.percentile(xx, 99.5) + 3, 40)
        for c in range(m.n_components):
            ax.hist(xx[comp == c], bins=bins, color=cols[c], alpha=0.6, label=f"component {c + 1} (n={int((comp == c).sum())})")
        xs = np.linspace(bins[0], bins[-1], 300)
        dens = np.zeros_like(xs)
        for w, mu, var in zip(m.weights_, m.means_.ravel(), m.covariances_.ravel()):
            dens += w * np.exp(-0.5 * ((xs / 100 + np.median(x)) - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        dens *= len(xx) * (bins[1] - bins[0]) / 100
        ax.plot(xs, dens, "k-", lw=1)
        ax.set_title(f"broad {sid}: {S.loc[sid, 'F25']:.0f} Hz @ {S.loc[sid, 'R25']:.1f} Hz, n={len(x)}\n"
                     f"{'SPLIT into ' + str(m.n_components) if m.n_components > 1 else 'one population'}; slope {100 * b:.1f} %/°C", fontsize=8, loc="left")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(show):]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        ax.set_xlabel("position along the line at 25 °C (% of rate, relative)", fontsize=8)
    fig.suptitle("stage two: singer-night positions brought to 25 °C with each streak's own slope; Gaussian mixture, split only where BIC, separation and size all agree", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(root, f"streak_splits{args.suffix_out}")
    fig.savefig(out + ".png", dpi=130)
    fig.savefig(out + ".pdf")
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
