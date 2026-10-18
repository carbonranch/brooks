"""Walker & Funk (2014, J. Orthoptera Res. 23:1-38) supplementary song table (SMTbls_Songs.xlsx, downloaded by MBE to
analysis/callers/anaxipha/) against the year's type lines: per recording the species, air temperature, pulse rate and
carrier; per species the rate and carrier lines on temperature; and the Figure 12 test (z across and along each type's
line at the recording's temperature) to say which type, if any, each species' recordings fall on.

Writes anaxipha/anaxipha_songs_tidy.csv (one row per recording with T, rate and carrier), anaxipha/anaxipha_vs_types.csv
(per species x type: n in the type's rate range, median z across, median z along, share within 2.5 sd) and prints the
best type per species. The overlay CSV for year_plane_figure --overlay (pulse_rate, carrier_kHz, species, T) is written
to <figures>/exploratory/anaxipha_published_points.csv.

Usage:
  python analysis/callers/anaxipha_published.py analysis/callers/runs/v5/year_nov24_nov25 --suffix _v9
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ids  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "anaxipha", "smtbls_songs.xlsx")
NEAR = {"AR", "LA", "TX", "OK", "MS", "MO", "TN"}
SD_ACROSS_FLOOR, T_UNC, Z_ON = 0.03, 2.0, 2.5


def tidy() -> pd.DataFrame:
    x = pd.ExcelFile(SRC); rows = []
    for sh in x.sheet_names:
        t = x.parse(sh)
        cols = {str(c).strip().lower(): c for c in t.columns}
        rate = next((cols[k] for k in ("p/s", "pulses/s", "p/s avg", "p/s avg", "pulse/ sec calc") if k in cols), None)
        if rate is None:
            rate = next((cols[k] for k in cols if k.startswith("p/s avg") or k == "p/s" or k == "pulse/ sec calc"), None)
        T = next((cols[k] for k in ("tempc", "temp (°c)") if k in cols), None)
        F = next((cols[k] for k in ("khz", "freq (khz)") if k in cols), None)
        st = next((cols[k] for k in ("state",) if k in cols), None)
        cage = next((cols[k] for k in ("field/cage", "field/caged", "lab/field") if k in cols), None)
        if rate is None or T is None or F is None:
            print(f"  {sh}: columns not found (rate {rate}, T {T}, F {F}); skipped"); continue
        d = pd.DataFrame({"species": f"Anaxipha {sh}", "T": pd.to_numeric(t[T], errors="coerce"), "pulse_rate": pd.to_numeric(t[rate], errors="coerce"),
                          "carrier_kHz": pd.to_numeric(t[F], errors="coerce"), "state": t[st].astype(str).str.strip().str.upper() if st is not None else "",
                          "setting": t[cage].astype(str).str.lower() if cage is not None else "", "year": pd.to_numeric(t["Year"], errors="coerce") if "Year" in t else np.nan})
        rows.append(d.dropna(subset=["T", "pulse_rate", "carrier_kHz"]))
    D = pd.concat(rows, ignore_index=True)
    D = D[(D.pulse_rate > 0) & (D.carrier_kHz > 1) & (D.carrier_kHz < 20) & D["T"].between(0, 40)]
    return D


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--figures", default=os.path.join(HERE, "runs", "figures_paper"))
    args = ap.parse_args()
    D = tidy(); out_dir = os.path.join(HERE, "anaxipha")
    D.to_csv(os.path.join(out_dir, "anaxipha_songs_tidy.csv"), index=False, float_format="%.3f")
    print(f"{len(D)} recordings with temperature, pulse rate and carrier, {D.species.nunique()} species")
    print(D.groupby("species").agg(n=("T", "size"), T_lo=("T", "min"), T_hi=("T", "max"), rate_med=("pulse_rate", "median"), kHz_med=("carrier_kHz", "median"), near=("state", lambda s: int(s.isin(NEAR).sum()))).round(1).to_string())
    # the types' lines: carrier on rate (log-log) over singer-hours, and rate on temperature (singer-hours), as streak_id_figure does
    A = pd.read_csv(os.path.join(args.root, f"animal_nights_streaks{args.suffix}.csv")); S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    P = pd.read_csv(os.path.join(args.root, "streak_hourly_points_all.csv"), usecols=["an", "R", "F", "T"]).dropna(subset=["T"]).merge(A[["an", "streak"]], on="an")
    ids = paper_ids.load(args.root, args.suffix)
    big = [s for s in S.index if S.loc[s, "n"] >= 40 and S.loc[s, "form"] != "chirp"]
    L = {}
    for s in big:
        g = P[P.streak == s]; x, y = np.log(g.R), np.log(g.F); b, a = np.polyfit(x, y, 1); sd = max(SD_ACROSS_FLOOR, float((y - (a + b * x)).std()))
        kT, aT = np.polyfit(g["T"], x, 1); sdT = float((x - (aT + kT * g["T"])).std())
        L[s] = dict(a=a, b=b, sd=sd, kT=kT, aT=aT, sdT=max(0.08, sdT), lo=float(np.percentile(x, 1)), hi=float(np.percentile(x, 99)), F25=S.loc[s, "F25"], R25=S.loc[s, "R25"])
    rows = []
    for sp, g in D.groupby("species"):
        x, y = np.log(g.pulse_rate), np.log(g.carrier_kHz * 1000)
        for s, l in L.items():
            m = (x >= l["lo"] - 0.1) & (x <= l["hi"] + 0.1)
            if m.sum() < 5: continue
            z_across = (y[m] - (l["a"] + l["b"] * x[m])) / l["sd"]
            exp_lnR = l["aT"] + l["kT"] * g["T"][m]; z_along = (x[m] - exp_lnR) / np.sqrt(l["sdT"] ** 2 + (l["kT"] * T_UNC) ** 2)
            rows.append(dict(species=sp, type=ids[s], position=f"{l['F25'] / 1000:.1f} kHz @ {l['R25']:.0f} Hz", n_in_range=int(m.sum()), n_total=len(g), z_across_med=float(np.median(z_across)), z_along_med=float(np.median(z_along)),
                             share_on=float(((np.abs(z_across) <= Z_ON) & (np.abs(z_along) <= Z_ON)).mean()), share_across_on=float((np.abs(z_across) <= Z_ON).mean())))
    R = pd.DataFrame(rows).sort_values(["species", "share_on", "z_across_med"], ascending=[True, False, True]); R.to_csv(os.path.join(out_dir, "anaxipha_vs_types.csv"), index=False, float_format="%.3f")
    print("\nbest type per species (share of recordings within 2.5 sd across AND along at matched temperature; then by |z across|):")
    for sp, g in R.groupby("species"):
        top = g.head(3)
        print(f"  {sp:22s} " + " | ".join(f"type {r.type} ({r.position}): on {r.share_on:.0%} of {r.n_in_range}, z across {r.z_across_med:+.1f}, along {r.z_along_med:+.1f}" for r in top.itertuples()))
    # species lines on temperature, for the text
    print("\nper-species lines: rate = exp(a + k T), carrier likewise; and carrier on rate")
    for sp, g in D.groupby("species"):
        if len(g) < 10: continue
        kR, aR = np.polyfit(g["T"], np.log(g.pulse_rate), 1); kF, aF = np.polyfit(g["T"], np.log(g.carrier_kHz), 1); b, a = np.polyfit(np.log(g.pulse_rate), np.log(g.carrier_kHz), 1)
        print(f"  {sp:22s} n={len(g):3d}  rate {np.exp(aR + 25 * kR):5.1f} Hz at 25 C, {100 * kR:+.1f} %/C | carrier {np.exp(aF + 25 * kF):.2f} kHz, {100 * kF:+.2f} %/C | log carrier on log rate slope {b:.2f}")
    O = D.rename(columns={})[["pulse_rate", "carrier_kHz", "species", "T"]].copy(); O["species"] = O.species.str.replace("Anaxipha ", "", regex=False); O["n_overlapping"] = 1
    os.makedirs(os.path.join(args.figures, "exploratory"), exist_ok=True); O.to_csv(os.path.join(args.figures, "exploratory", "anaxipha_published_points.csv"), index=False, float_format="%.3f")
    print(f"\nwrote {out_dir}/anaxipha_songs_tidy.csv, anaxipha_vs_types.csv and the overlay CSV")


if __name__ == "__main__":
    main()
