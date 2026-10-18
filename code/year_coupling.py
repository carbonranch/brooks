"""Carrier-rate coupling per streak over the year, measured WITHIN each singer-period:
the slope of log F_fix on log R across the windows of one track through its night
(tracks with >= MIN_WIN windows and >= MIN_RANGE Hz of rate range). One point per
streak: median and IQR over its tracks, against the streak's carrier, colored by the
subfamily of its SINA candidate by position (streak_sina_matches.csv from
match_sina.py, if present).

Reads the night singer-periods of animal_nights_streaks<suffix>.csv, whose ids are
<unit>:<track>, and the unit's night_night_global_window_animals.csv (F_fix from the
global linker's fixed-window carrier). The per-track table is cached as
year_coupling_tracks<suffix>.csv; delete it to recompute.

Usage:
    python analysis/callers/year_coupling.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --jobs 12
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_WIN, MIN_RANGE = 12, 0.3
GROUP = {"oecanthinae": "tree crickets (Oecanthinae)", "trigonidiinae": "trigs (Trigonidiinae)", "nemobiinae": "ground crickets (Nemobiinae)",
         "gryllinae": "field crickets (Gryllinae)", "eneopterinae": "Eneopterinae", "tettigoniidae": "katydids (Tettigoniidae)",
         "tettigoniinae": "katydids (Tettigoniidae)", "conocephalinae": "katydids (Tettigoniidae)", "phaneropterinae": "katydids (Tettigoniidae)"}
COL = {"tree crickets (Oecanthinae)": "tab:green", "trigs (Trigonidiinae)": "tab:orange", "ground crickets (Nemobiinae)": "tab:brown",
       "field crickets (Gryllinae)": "tab:red", "Eneopterinae": "tab:pink", "katydids (Tettigoniidae)": "tab:blue", "no candidate": "0.5"}
def _unit(args):
    unit, tracks, roots = args          # roots travel with the job: macOS spawns workers by re-import, so a module global would be empty
    for r in roots:
        p = os.path.join(r, unit, "night_night_global_window_animals.csv")
        if os.path.exists(p):
            break
    else:
        return []
    try:
        W = pd.read_csv(p, usecols=["track", "R", "F_fix"])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for t in tracks:
        g = W[(W.track == t) & W.F_fix.notna() & (W.R > 0)]
        if len(g) < MIN_WIN:
            continue
        rr = g.R.max() - g.R.min()
        if rr < MIN_RANGE:
            out.append(dict(an=f"{unit}:{t}", n=len(g), R_range=rr, slope=np.nan, r=np.nan)); continue
        x, y = np.log(g.R.to_numpy()), np.log(g.F_fix.to_numpy())
        b = np.polyfit(x, y, 1)[0]
        out.append(dict(an=f"{unit}:{t}", n=len(g), R_range=rr, slope=b, r=np.corrcoef(x, y)[0, 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--suffix", default="_v6")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=40, help="streaks with at least this many singer-periods are plotted")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[S.harmonic_of < 0] if "harmonic_of" in S.columns else S
    cache = os.path.join(root, f"year_coupling_tracks{args.suffix}.csv")
    try:
        Tk = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame()
    except pd.errors.EmptyDataError:
        Tk = pd.DataFrame()
    if Tk.empty or "an" not in Tk.columns:
        A["unit"] = A.an.str.split(":").str[0]; A["track"] = A.an.str.split(":").str[1].astype(int)
        jobs = [(u, g.track.tolist(), args.roots) for u, g in A.groupby("unit")]
        with Pool(args.jobs) as pool:
            parts = pool.map(_unit, jobs, chunksize=16)
        Tk = pd.DataFrame([r for p in parts for r in p])
        Tk.to_csv(cache, index=False, float_format="%.4f")
    J = A[["an", "streak", "F", "R", "T"]].merge(Tk, on="an")
    print(f"{len(J):,} singer-periods with >= {MIN_WIN} windows; {J.slope.notna().sum():,} with >= {MIN_RANGE} Hz of rate range")
    J = J[J.slope.notna()]
    g = J.groupby("streak").slope.agg(n_tracks="size", within="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75),
                                      pos_share=lambda s: float((s > 0).mean()))
    S = S.join(g)
    mp = os.path.join(root, "streak_sina_matches.csv")
    if os.path.exists(mp):
        M = pd.read_csv(mp); top = M[M["rank"] == 1].set_index("streak")
        S["candidate"] = top.species.reindex(S.index)
        S["group"] = top.subfamily.reindex(S.index).astype(str).str.lower().map(GROUP).fillna("no candidate")
    else:
        S["candidate"] = ""; S["group"] = "no candidate"
    S = S[(S.n >= args.min_n) & S.within.notna()]
    S.to_csv(os.path.join(root, f"year_coupling{args.suffix}.csv"), float_format="%.3f")
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    print(S[["n", "F25", "R25", "n_tracks", "within", "q25", "q75", "pos_share", "candidate", "group"]].sort_values("within").round(2).to_string())
    print("\nby group: median within-animal coupling")
    print(S.groupby("group").within.agg(["median", "min", "max", "size"]).round(2).to_string())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for grp, gg in S.groupby("group"):
        ax.errorbar(gg.F25, gg.within, yerr=[gg.within - gg.q25, gg.q75 - gg.within], fmt="o", color=COL.get(grp, "0.5"),
                    ms=6, elinewidth=0.8, capsize=2, alpha=0.85, label=f"{grp} ({len(gg)})")
        for r in gg.itertuples():
            ax.annotate(str(r.Index), (r.F25, r.within), fontsize=6, xytext=(3, 3), textcoords="offset points", color="0.3")
    ax.axhline(0, color="k", lw=0.6); ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("streak carrier at 25 °C (Hz)")
    ax.set_ylabel("within-animal coupling: % carrier per % rate (median, IQR over tracks)")
    ax.set_title(f"Carrier-rate coupling within singer-periods over the year (streaks with >= {args.min_n} singer-periods);\n"
                 f"color = subfamily of the SINA candidate by position; dotted = clockwork limit", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, which="both", lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"year_coupling{args.suffix}.png"), dpi=130); fig.savefig(os.path.join(root, f"year_coupling{args.suffix}.pdf"))
    print(f"wrote {root}/year_coupling{args.suffix}.png / .csv")


if __name__ == "__main__":
    main()
