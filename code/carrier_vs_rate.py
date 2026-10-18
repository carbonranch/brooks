"""Within-animal carrier frequency against rate, over the windows of a sweep.

Rate is the thermometer (the literature's temperature relation). The question
here is whether the CARRIER moves with it inside one animal, and by how much.
Reads the tracked animals written by window_stability.py.

Usage:
    python analysis/callers/carrier_vs_rate.py analysis/callers/runs/stability_<stem>_animals.csv [--min-windows 20]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("animals_csv")
    ap.add_argument("--min-windows", type=int, default=20)
    ap.add_argument("--carrier", default="F_c", choices=["F_c", "F_pk", "F_ref", "F_fix"])
    args = ap.parse_args()

    A = pd.read_csv(args.animals_csv)
    A["t"] = pd.to_datetime(A.t, utc=True).dt.tz_convert("America/Chicago")
    A = A[A[args.carrier].notna()]
    n_per = A.groupby("track").t.nunique()
    keep = n_per[n_per >= args.min_windows].index
    A = A[A.track.isin(keep)].sort_values("t")
    if A.empty:
        raise SystemExit("no track long enough")

    fig, axes = plt.subplots(3, len(keep), figsize=(5.2 * len(keep), 12), squeeze=False)
    rows = []
    for j, tr in enumerate(sorted(keep, key=lambda k: A[A.track == k].F_c.median())):
        g = A[A.track == tr]
        F, R, t = g[args.carrier].to_numpy(), g.R.to_numpy(), g.t
        hours = (t - t.min()).dt.total_seconds().to_numpy() / 3600
        pf = np.polyfit(hours, F, 1)
        pr = np.polyfit(hours, R, 1)
        pfr = np.polyfit(R, F, 1)
        r_fr = np.corrcoef(R, F)[0, 1]
        rows.append(dict(track=tr, n=len(g), F_med=np.median(F), R_med=np.median(R),
                         dF_dt_Hz_per_h=pf[0], dR_dt_Hz_per_h=pr[0],
                         dF_dR_Hz_per_Hz=pfr[0], corr_F_R=r_fr,
                         rel_slope=(pfr[0] * np.median(R) / np.median(F)),
                         F_scatter_Hz=np.std(F - np.polyval(pf, hours)),
                         R_scatter_Hz=np.std(R - np.polyval(pr, hours))))
        ax = axes[0, j]
        ax.plot(t, F, "k.-", lw=0.7)
        ax.set_title(f"track {tr}: {np.median(F):.0f} Hz @ {np.median(R):.1f} Hz, {len(g)} windows\n"
                     f"carrier {pf[0]:+.0f} Hz/h, rate {pr[0]:+.2f} Hz/h", fontsize=9)
        ax.set_ylabel(f"carrier {args.carrier} (Hz)")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax = axes[1, j]
        ax.plot(t, R, "b.-", lw=0.7)
        ax.set_ylabel("rate (Hz)")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax = axes[2, j]
        sc = ax.scatter(R, F, c=hours, cmap="viridis", s=18)
        xx = np.linspace(R.min(), R.max(), 10)
        ax.plot(xx, np.polyval(pfr, xx), "r-", lw=1)
        ax.set_xlabel("rate (Hz)")
        ax.set_ylabel(f"carrier (Hz)")
        ax.set_title(f"dF/dR = {pfr[0]:+.0f} Hz per Hz  (r = {r_fr:+.2f}); "
                     f"relative {rows[-1]['rel_slope']:+.2f} %/%", fontsize=9)
        plt.colorbar(sc, ax=ax, label="hours from start")
    fig.tight_layout()
    out = args.animals_csv.replace("_animals.csv", f"_carrier_vs_rate_{args.carrier}.png")
    fig.savefig(out, dpi=110)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    T.to_csv(out.replace(".png", ".csv"), index=False, float_format="%.4f")
    print(f"wrote {out} / .csv")


if __name__ == "__main__":
    main()
