"""Each trill type's syllable rate (or carrier) against air temperature, one panel per type on a letter page (MBE
2026-10-17: "two figures - Figure 18 and 19 for now - that show T vs R and T vs F for the trillers").

Points are the type's singer-hours (HRRR air temperature at the hour), with the fitted line of log value on temperature
and its slope in percent per degree in the panel head; all panels share the axes. Also writes <out>.csv with the per-type
fits (value at 25 C, slope, r, n).

Usage:
  python analysis/callers/temperature_panels_figure.py analysis/callers/runs/v5/year_nov24_nov25 --suffix _v9 --var R --number 18 --out analysis/callers/runs/figures_paper/F18_rate_vs_temperature
  python analysis/callers/temperature_panels_figure.py analysis/callers/runs/v5/year_nov24_nov25 --suffix _v9 --var F --number 19 --out analysis/callers/runs/figures_paper/F19_carrier_vs_temperature
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

NCOL = 5
TLO, THI = 5.0, 36.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--var", choices=["R", "F"], default="R"); ap.add_argument("--number", default="18"); ap.add_argument("--out", required=True)
    ap.add_argument("--within", action="store_true", help="within-singer version: each track's hours as deviations from the track's own mean temperature and mean log value; pooled slope and the median of per-track slopes (tracks with >= 4 hours and >= 2 C of range)")
    args = ap.parse_args(); root = args.root; var = args.var
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0); S = S[(S.n >= 40) & (S.form == "trill")]
    ids = paper_ids.load(root, args.suffix); types = sorted(S.index, key=lambda s: ids[s])
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["an", "R", "F", "T", "rc_frac"]).dropna(subset=["T"]).merge(A[["an", "streak"]], on="an")
    rc = P.groupby("an").rc_frac.mean(); P = P[~P.an.isin(rc.index[rc >= 0.5])]
    name = "syllable rate (Hz)" if var == "R" else "carrier (kHz)"; lo, hi = (10, 100) if var == "R" else (1800, 12000)
    ticks = [10, 20, 30, 50, 100] if var == "R" else [2000, 3000, 5000, 8000]
    fmt = (lambda v, p: f"{v:g}") if var == "R" else (lambda v, p: f"{v / 1000:g}")
    page = PaperPage(number=args.number, title=f"{'Syllable rate' if var == 'R' else 'Carrier frequency'} against air temperature{' within singers' if args.within else ''}, type by type")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    nrow = int(np.ceil(len(types) / NCOL)); X0, X1, TOP = 1.35, 7.3, 1.15
    GAPX, GAPY, HEAD = 0.14, 0.12, 0.20; PW = (X1 - X0 - (NCOL - 1) * GAPX) / NCOL; PITCH = (Hc - (TOP - 1.0) - 0.15) / nrow; PH = PITCH - GAPY - HEAD
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20)); rows = []
    for i, s in enumerate(types):
        rr, cc = divmod(i, NCOL); x0 = X0 + cc * (PW + GAPX); top = TOP + rr * PITCH + HEAD
        ax = fig.add_axes(rect(x0, x0 + PW, top, PH)); g = P[P.streak == s]
        if args.within:
            g = g.copy(); g["lv"] = np.log(g[var]); n_tr = g.groupby("an").size(); rng = g.groupby("an")["T"].agg(lambda x: x.max() - x.min())
            g = g[g.an.isin(n_tr.index[(n_tr >= 4) & (rng >= 2)])]
            g["dT"] = g["T"] - g.groupby("an")["T"].transform("mean"); g["dv"] = 100 * (g.lv - g.groupby("an").lv.transform("mean"))
            t, v = g.dT.to_numpy(), g.dv.to_numpy()
            ax.scatter(t, v, s=0.5 if len(g) > 2000 else 2.5, color=cols[i % 20], alpha=0.35 if len(g) > 2000 else 0.7, lw=0, rasterized=True)
            k = np.polyfit(t, v, 1)[0] / 100 if len(g) > 20 else np.nan; a = 0.0; r = np.corrcoef(t, v)[0, 1] if len(g) > 20 else np.nan
            per_track = g.groupby("an").apply(lambda d: np.polyfit(d.dT, d.dv, 1)[0]) if len(g) else pd.Series(dtype=float)
            tt = np.array([-6, 6]); ax.plot(tt, 100 * k * tt, color="k", lw=0.9); ax.axhline(0, color="0.8", lw=0.4); ax.axvline(0, color="0.8", lw=0.4)
            ax.set_xlim(-7, 7); ax.set_ylim(-35, 35); ax.set_xticks([-5, 0, 5]); ax.set_yticks([-30, -15, 0, 15, 30])
            head2 = f"{100 * k:+.1f} %/°C; singers' median {per_track.median():+.1f}, n {len(per_track)}"
        else:
            v = g[var].to_numpy(); t = g["T"].to_numpy()
            ax.scatter(t, v, s=0.5, color=cols[i % 20], alpha=0.35, lw=0, rasterized=True)
            k, a = np.polyfit(t, np.log(v), 1); r = np.corrcoef(t, np.log(v))[0, 1]; tt = np.array([np.percentile(t, 1), np.percentile(t, 99)])
            ax.plot(tt, np.exp(a + k * tt), color="k", lw=0.9)
            ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.set_xlim(TLO, THI); ax.set_xticks([10, 20, 30]); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(fmt))
            head2 = f"{100 * k:+.1f} % per °C"
        ax.tick_params(labelsize=BODY_PT - 1, length=2, pad=1.5); ax.grid(True, alpha=0.2)
        if cc != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(("change in " + name.split(" (")[0] + " (%)") if args.within else name, fontsize=BODY_PT - 1, labelpad=1)
        if rr != nrow - 1 and i + NCOL < len(types):
            ax.set_xticklabels([])
        if rr == nrow - 1 or i + NCOL >= len(types):
            ax.set_xlabel("change in air temperature (°C)" if args.within else "air temperature (°C)", fontsize=BODY_PT - 1, labelpad=1)
        v25 = np.exp(a + 25 * k) if not args.within else np.nan
        ax.set_title(f"{ids[s]}  {S.loc[s, 'F25'] / 1000:.1f} kHz @ {S.loc[s, 'R25']:.0f} Hz\n{head2}", fontsize=BODY_PT - 0.5 - (1.2 if args.within else 0), loc="left", pad=2, linespacing=1.05)   # two lines: one line ran into the next column
        rows.append(dict(type=ids[s], streak=s, F25=int(S.loc[s, "F25"]), R25=round(S.loc[s, "R25"], 1), var=var, value_at_25C=v25, slope_pct_per_C=100 * k, r=r, n_hours=len(g), T_lo=tt[0], T_hi=tt[1]))
    page.finish(fig, args.out)
    D = pd.DataFrame(rows); D.to_csv(args.out + ".csv", index=False, float_format="%.3f")
    print(f"{var}: slopes {D.slope_pct_per_C.min():+.1f} to {D.slope_pct_per_C.max():+.1f} %/C, median {D.slope_pct_per_C.median():+.1f}; wrote {args.out}.pdf / .png / .csv")


if __name__ == "__main__":
    main()
