"""The streak story in four panels, from the month's singer-nights.

  a. carrier against rate, every singer-night >= 2 h, no other information:
     the points fall on oblique lines (streaks);
  b. the same points coloured by air temperature: each line is a temperature
     gradient, cool at the lower-left end, warm at the upper-right;
  c. each singer-night brought to 25 degC with the pooled within-singer
     slopes: the lines collapse to compact clouds;
  d. the clouds clustered (average linkage, 10 % carrier / 12 % rate) and
     labelled where a name is known.

Writes the composite and each panel on its own (PDF + PNG).

Usage:
    python analysis/callers/streak_story_figure.py analysis/callers/runs/month_v2 --out streak_story
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# names for panel d (month_v2 streak ids). '?' marks a match on numbers only.
NAMES = {
    75: "Oecanthus texensis",
    46: "Oecanthus latipennis",
    60: "Gryllus rubens",
    43: "O. fultoni?",
    74: "O. pini?",
    53: "O. nigricornis grp?",
    72: "Gryllus (pennsylvanicus /\nvernalis)?",
    55: "G. fultoni?",
    101: "Anaxipha exigua?",
    84: "Conocephalus strictus?",
    32: "C. nemoralis?",
    125: "ground cricket / trig (unnamed)",
    128: "ground cricket / trig (unnamed, mixed)",
    62: "trig (unnamed, mixed)",
}
# the same names on the two-stage (v3) streak ids: 100*broad + component
NAMES_V3 = {
    401: "Oecanthus texensis",
    1101: "Oecanthus latipennis",
    4701: "Gryllus rubens",
    301: "O. pini?",
    302: "O. nigricornis grp?",
    4601: "Gryllus (pennsylvanicus / vernalis)?",
    6702: "Anaxipha exigua?",
    6301: "Conocephalus strictus?",
    3701: "C. nemoralis?",
    7001: "ground crickets / trigs (unnamed; two syllable shapes)",
    6901: "ground cricket / trig (unnamed)",
    5201: "trig (unnamed; several shapes)",
    6402: "unnamed 8.5 kHz @ 59 Hz",
    6401: "unnamed 7.9 kHz @ 51 Hz",
    5101: "unnamed 5.0 kHz @ 38 Hz",
    2601: "unnamed 21.7 kHz @ 24 Hz",
}
NAMES_BY_SUFFIX = {"": NAMES, "_v3": NAMES_V3}
XLIM, YLIM = (8, 105), (1500, 23000)
MIN_N = 15


def style(ax, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([10, 20, 30, 50, 100])
    ax.set_xticklabels(["10", "20", "30", "50", "100"])
    ax.set_yticks([2000, 3000, 5000, 10000, 20000])
    ax.set_yticklabels(["2", "3", "5", "10", "20"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, which="both", lw=0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="streak_story")
    ap.add_argument("--dot", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.15, help="point alpha; low values show density where streaks are dense")
    ap.add_argument("--cmap", default="jet", help="colour map for panel b")
    ap.add_argument("--bands", type=float, nargs="+", default=[19, 25, 29], help="centres of the temperature bands for the check figure")
    ap.add_argument("--band-width", type=float, default=1.0, help="half-width of each band in degC")
    ap.add_argument("--suffix", default="", help="read animal_nights_streaks<suffix>.csv / streaks_coarse<suffix>.csv")
    ap.add_argument("--no-names", action="store_true", help="panel d without the name key (labels belong to another clustering)")
    args = ap.parse_args()
    al = args.alpha
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    big = S[S.n >= MIN_N].index
    A["big"] = A.streak.isin(big)
    n_sn, n_streak = len(A), len(big)
    nights = pd.to_datetime(A.night)
    month = f"{nights.min():%-d %b} – {nights.max():%-d %b %Y}"

    panels = {}

    def panel_a(ax):
        ax.scatter(A.R, A.F, s=args.dot, color="k", alpha=al, edgecolors="none")
        style(ax, "syllable rate (Hz)", "carrier frequency (kHz)")
        ax.set_title(f"(A)  {n_sn:,} singer-nights, {month}", loc="left", fontsize=10, fontweight="bold")

    def panel_b(ax):
        sc = ax.scatter(A.R, A.F, s=args.dot, c=A["T"], cmap=args.cmap, vmin=12, vmax=32, alpha=min(1.0, 2 * al), edgecolors="none")
        style(ax, "syllable rate (Hz)", "carrier frequency (kHz)")
        ax.set_title("(B)  the same singer-nights coloured by air temperature", loc="left", fontsize=10, fontweight="bold")
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.solids.set_alpha(1.0)
        cb.set_label("night mean temperature (°C)")
        return sc

    def panel_c(ax):
        ax.scatter(A.R25, A.F25, s=args.dot, color="k", alpha=al, edgecolors="none")
        style(ax, "syllable rate at 25 °C (Hz)", "carrier frequency at 25 °C (kHz)")
        ax.set_title("(C)  each singer-night brought to 25 °C (1.5 %/°C rate, 0.46 %/°C carrier)", loc="left", fontsize=10, fontweight="bold")

    def panel_d(ax):
        rng = np.random.default_rng(0)
        ids = list(big)
        cols = plt.cm.tab20(np.linspace(0, 1, 20))
        order = {sid: i for i, sid in enumerate(rng.permutation(ids))}
        other = A[~A.big]
        ax.scatter(other.R25, other.F25, s=args.dot * 0.6, color="0.8", alpha=min(1.0, 2 * al), edgecolors="none")
        for sid in ids:
            g = A[A.streak == sid]
            ax.scatter(g.R25, g.F25, s=args.dot, color=cols[order[sid] % 20], alpha=min(1.0, 3 * al), edgecolors="none")
        # numeric tags at the streaks, key in the empty upper-left corner
        names = NAMES_BY_SUFFIX.get(args.suffix, {})
        named = [] if args.no_names else [(sid, n) for sid, n in names.items() if sid in S.index]
        if args.no_names:      # tag every streak with its id instead
            for sid in ids:
                ax.annotate(str(sid), (S.loc[sid, "R25"], S.loc[sid, "F25"]), xytext=(4, 3), textcoords="offset points", fontsize=6, color="k",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
        named.sort(key=lambda kv: -S.loc[kv[0], "F25"])
        for sid, name in named:
            x, y = S.loc[sid, "R25"], S.loc[sid, "F25"]
            ax.annotate(str(sid), (x, y), xytext=(5, 3), textcoords="offset points", fontsize=6.5, weight="bold", color="k",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        y0 = 0.98
        for sid, name in named:
            confirmed = "?" not in name and "unnamed" not in name
            ax.text(0.02, y0, f"{sid:>3}  {name.replace(chr(10), ' ')}", transform=ax.transAxes, fontsize=6.3, va="top", ha="left",
                    style="italic" if confirmed else "normal", family="monospace" if not confirmed else None)
            y0 -= 0.031
        style(ax, "syllable rate at 25 °C (Hz)", "carrier frequency at 25 °C (kHz)")
        ax.set_title(f"(D)  {n_streak} streaks with ≥ {MIN_N} singer-nights (colours); tentative names, italic = confirmed", loc="left", fontsize=10, fontweight="bold")

    panels = [("a", panel_a), ("b", panel_b), ("c", panel_c), ("d", panel_d)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for (tag, fn), ax in zip(panels, axes.ravel()):
        fn(ax)
    fig.tight_layout()
    out = os.path.join(args.root, args.out)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=130)
    for tag, fn in panels:
        f1, ax1 = plt.subplots(figsize=(6.6, 5.6))
        fn(ax1)
        f1.tight_layout()
        f1.savefig(f"{out}_{tag}.pdf")
        f1.savefig(f"{out}_{tag}.png", dpi=150)
        plt.close(f1)
    print(f"{n_sn} singer-nights, {n_streak} streaks; wrote {out}.pdf/.png and {out}_a..d.pdf/.png")

    # ---- check: raw positions of the singer-nights recorded near each band
    # temperature, with the streaks' 25 degC centres overlaid. If the
    # correction is right the ~25 degC band sits on the centres and the
    # cooler / warmer bands are displaced along the lines.
    hw = args.band_width
    fig2, axes2 = plt.subplots(1, len(args.bands), figsize=(6.4 * len(args.bands), 5.8), squeeze=False)
    Sb = S.loc[big]
    for ax, tc in zip(axes2.ravel(), args.bands):
        g = A[(A["T"] >= tc - hw) & (A["T"] <= tc + hw)]
        ax.scatter(g.R, g.F, s=args.dot * 1.5, color="k", alpha=min(1.0, 2.5 * al), edgecolors="none", label=f"singer-nights, raw")
        ax.scatter(Sb.R25, Sb.F25, s=22, marker="+", color="red", lw=0.9, label="streak centres at 25 °C")
        style(ax, "syllable rate (Hz), uncorrected", "carrier frequency (kHz), uncorrected")
        ax.set_title(f"{len(g):,} singer-nights with night mean T = {tc:g} ± {hw:g} °C", loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="lower right", markerscale=2)
    fig2.suptitle("check: raw positions of singer-nights recorded near a given temperature, against the streaks' 25 °C centres", fontsize=10)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(out + "_check_bands.pdf")
    fig2.savefig(out + "_check_bands.png", dpi=130)
    print("wrote " + out + "_check_bands.pdf/.png")


if __name__ == "__main__":
    main()
