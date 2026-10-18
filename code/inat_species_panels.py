"""Exploratory: one panel per iNaturalist species on the year's plane, points only (MBE 2026-10-17: "a separate panel
for each iNat species - and no lines").

Each panel shows the year's trill-type lines in gray (fitted log-log line per type between the 1st and 99th percentile
of its rates, type number at the warm end) over a thinned gray cloud of the singer-hours, with that species' iNaturalist
recordings as temperature-colored points (ERA5 2 m temperature at the observation's hour, turbo scale, black rims;
gray fill where no temperature). Species with at least --min-n measured recordings, crickets first (Gryllidae and
allies), then the rest; pages of --rows x --cols panels.

Usage:
  python analysis/callers/inat_species_panels.py analysis/callers/runs/v5/year_nov24_nov25 --suffix _v9 \
      --points analysis/callers/runs/figures_paper/exploratory/inat_points.csv \
      --out analysis/callers/runs/figures_paper/exploratory/inat_species_panels
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ids  # noqa: E402

CRICKET = ("Anaxipha", "Oecanthus", "Gryllus", "Allonemobius", "Eunemobius", "Neonemobius", "Hapithus", "Orocharis", "Cyrtoxipha", "Phyllopalpus",
           "Velarifictorus", "Anurogryllus", "Miogryllus", "Gryllodes", "Acheta", "Cycloptilum", "Neocurtilla", "Gryllotalpa", "Pictonemobius", "Falcicula")
TLO, THI = 10.0, 35.0
RLO, RHI, FLO, FHI = 8.0, 100.0, 1500.0, 20000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--points", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=5); ap.add_argument("--rows", type=int, default=6); ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args(); root = args.root
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0); S = S[(S.n >= 40) & (S.form == "trill")]; ids = paper_ids.load(root, args.suffix)
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["an", "R", "F", "T", "rc_frac"]).dropna(subset=["T"]).merge(A[["an", "streak"]], on="an")
    rc = P.groupby("an").rc_frac.mean(); P = P[~P.an.isin(rc.index[rc >= 0.5])]
    lines = []
    for s in S.index:
        g = P[P.streak == s]; x, y = np.log(g.R), np.log(g.F); b, a = np.polyfit(x, y, 1); xx = np.percentile(x, [1, 99])
        lines.append((str(ids[s]), np.exp(xx), np.exp(a + b * xx)))
    bg = P.sample(min(40000, len(P)), random_state=0)
    O = pd.read_csv(args.points); O = O[O.pulse_rate.between(RLO * 0.8, RHI * 1.25) & (O.carrier_kHz * 1000).between(FLO * 0.8, FHI * 1.25)]
    cnt = O.groupby("species").size(); sp = [s for s in cnt.index if cnt[s] >= args.min_n]
    sp = sorted(sp, key=lambda s: (0 if s.startswith(CRICKET) else 1, -cnt[s], s))
    per = args.rows * args.cols; n_pages = int(np.ceil(len(sp) / per))
    print(f"{len(sp)} species with >= {args.min_n} recordings on {n_pages} page(s); {int(O.species.isin(sp).sum()):,} points")
    ticks_r = [10, 20, 30, 50, 100]; ticks_f = [2000, 3000, 5000, 8000, 15000]
    with PdfPages(args.out + ".pdf") as pdf:
        for pg in range(n_pages):
            fig, axs = plt.subplots(args.rows, args.cols, figsize=(8.5, 11)); axs = axs.ravel()
            for k, ax in enumerate(axs):
                i = pg * per + k
                if i >= len(sp):
                    ax.set_axis_off(); continue
                s = sp[i]; g = O[O.species == s]
                ax.scatter(bg.R, bg.F, s=0.3, color="0.82", lw=0, rasterized=True)
                for name, xx, yy in lines:
                    ax.plot(xx, yy, color="0.55", lw=0.5); ax.annotate(name, (xx[1], yy[1]), xytext=(1.5, 0), textcoords="offset points", fontsize=3.5, color="0.4", va="center")
                has_t = g["T"].notna() if "T" in g.columns else pd.Series(False, index=g.index)
                ax.scatter(g.pulse_rate[~has_t], g.carrier_kHz[~has_t] * 1000, s=12, color="0.5", edgecolors="k", lw=0.4, zorder=5)
                ax.scatter(g.pulse_rate[has_t], g.carrier_kHz[has_t] * 1000, s=12, c=g["T"][has_t], cmap="turbo", norm=Normalize(TLO, THI), edgecolors="k", lw=0.4, zorder=6)
                ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(RLO, RHI); ax.set_ylim(FLO, FHI)
                ax.xaxis.set_major_locator(FixedLocator(ticks_r)); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
                ax.yaxis.set_major_locator(FixedLocator(ticks_f)); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
                ax.tick_params(labelsize=5.5, length=2, pad=1.5); ax.grid(True, alpha=0.15)
                gen = s.split()[0]; nm = f"{gen[0]}. {' '.join(s.split()[1:])}" if " " in s else s
                ax.set_title(f"{nm}  (n = {len(g)}{'' if s.startswith(CRICKET) else ', not a cricket'})", fontsize=6.5, loc="left", pad=2, style="italic")
                if k % args.cols == 0:
                    ax.set_ylabel("carrier (kHz)", fontsize=6, labelpad=1)
                else:
                    ax.set_yticklabels([])
                if k // args.cols == args.rows - 1 or i + args.cols >= len(sp):
                    ax.set_xlabel("syllable rate (Hz)", fontsize=6, labelpad=1)
                else:
                    ax.set_xticklabels([])
            cax = fig.add_axes([0.72, 0.972, 0.20, 0.009]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(TLO, THI), cmap="turbo"), cax=cax, orientation="horizontal")
            cb.set_label("ERA5 air temperature at the observation (°C); gray = none", fontsize=6, labelpad=1.5); cb.ax.tick_params(labelsize=5.5, length=2)
            fig.suptitle(f"iNaturalist recordings by species on the year's plane (page {pg + 1} of {n_pages}): type lines in gray, numbers at the warm end", fontsize=8, x=0.06, ha="left", y=0.99)
            fig.subplots_adjust(left=0.07, right=0.98, top=0.925, bottom=0.04, wspace=0.08, hspace=0.28)
            pdf.savefig(fig); fig.savefig(f"{args.out}_p{pg + 1}.png", dpi=130); plt.close(fig)
    print(f"wrote {args.out}.pdf and {args.out}_p<k>.png")


if __name__ == "__main__":
    main()
