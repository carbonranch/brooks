"""Measure the iNaturalist recordings (inat_fetch.py) the way the SINA and InsectSet clips were measured, attach an air
temperature to each from ERA5 (Open-Meteo archive, 2 m temperature at the observation's coordinates and hour), and test
each recording against the year's type lines at matched temperature (the Figure 12 test, as anaxipha_published.py).

Writes <root>/measurements.csv (per recording: carrier, band, syllable rate, depth, chirp rate, bout test; ERA5
temperature), <root>/inat_vs_types.csv (per recording the best type, z across and along), <root>/inat_species_table.csv,
and the overlay CSV <figures>/exploratory/inat_points.csv (pulse_rate, carrier_kHz, species, T) for
year_plane_figure --overlay.

Usage:
  python -I analysis/callers/inat_measure.py analysis/callers/runs/inat --year analysis/callers/runs/v5/year_nov24_nov25 --jobs 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests

os.environ.setdefault("BROOKS_CHIRP", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paper_ids  # noqa: E402
from insectset_measure import _one  # noqa: E402

SD_ACROSS_FLOOR, T_UNC, Z_ON = 0.03, 2.0, 2.5
NEAR = {"AR", "LA", "TX", "OK", "MO", "MS", "TN"}


def era5_temperature(D: pd.DataFrame, cache: str) -> pd.Series:
    """2 m temperature (C) at each row's coordinates at the observed hour, ERA5 via Open-Meteo; cached per observation."""
    C = pd.read_csv(cache, index_col=0).iloc[:, 0] if os.path.exists(cache) else pd.Series(dtype=float)
    out = {}
    for r in D.drop_duplicates("obs_id").itertuples():
        if r.obs_id in C.index:
            out[r.obs_id] = C[r.obs_id]; continue
        t = pd.to_datetime(r.observed, utc=True, errors="coerce") if isinstance(r.observed, str) and "T" in r.observed else pd.NaT
        if pd.isna(t) or not np.isfinite(float(r.lat)) or not np.isfinite(float(r.lon)):
            out[r.obs_id] = np.nan; continue
        try:
            q = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(latitude=float(r.lat), longitude=float(r.lon), start_date=t.strftime("%Y-%m-%d"), end_date=t.strftime("%Y-%m-%d"), hourly="temperature_2m", timezone="UTC"), timeout=60).json()
            hh = pd.to_datetime(q["hourly"]["time"], utc=True); tt = np.array(q["hourly"]["temperature_2m"], dtype=float)
            out[r.obs_id] = float(np.interp(t.value, hh.asi8, tt)) if np.isfinite(tt).all() else np.nan
        except Exception:  # noqa: BLE001
            out[r.obs_id] = np.nan
        time.sleep(0.12)
        if len(out) % 200 == 0:
            pd.Series({**C.to_dict(), **out}).to_csv(cache); print(f"  temperatures: {len(out)}", flush=True)
    S = pd.Series({**C.to_dict(), **out}); S.to_csv(cache)
    return D.obs_id.map(S)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--year", required=True); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--figures", default=os.path.join(HERE, "runs", "figures_paper"))
    args = ap.parse_args()
    D = pd.read_csv(os.path.join(args.root, "observations.csv")); D = D[D.wav.astype(str).str.endswith(".wav") & D.wav.astype(str).map(os.path.exists)].copy()
    print(f"{len(D):,} recordings to measure", flush=True)
    # band hint: the species' SINA page frequency where SINA has the species (68 of 103 taxa), as the InsectSet measurement used it
    hp = os.path.join(args.root, "sina_khz_by_name.csv"); hint = pd.read_csv(hp, index_col=0).iloc[:, 0].to_dict() if os.path.exists(hp) else {}
    with Pool(args.jobs) as pool:
        res = pool.map(_one, [(w, hint.get(t)) for w, t in zip(D.wav, D.taxon)], chunksize=4)
    M = pd.DataFrame(res); M["wav"] = D.wav.values
    # AAC encodes 1024-sample frames, 43.07 per second at 44.1 kHz: a frame-rate ripple the detector reads as a syllable rate in ~9 % of m4a files; drop it and its half and double
    art = M.R.between(42.8, 43.3) | M.R.between(21.4, 21.7) | M.R.between(85.8, 86.4)
    M.loc[art & (M.status == "ok"), "status"] = "codec frame-rate artifact"; print(f"  {int(art.sum())} measurements at the AAC frame rate set aside", flush=True)
    M = M.merge(D.drop(columns=[c for c in ("file_name",) if c in D.columns]), on="wav", how="left")
    M["T_era5"] = era5_temperature(M, os.path.join(args.root, "era5_cache.csv"))
    M.to_csv(os.path.join(args.root, "measurements.csv"), index=False, float_format="%.4f")
    ok = M[(M.status == "ok") & M.R.notna()]
    print(f"{(M.status == 'ok').sum():,} measured; {len(ok):,} with a carrier and syllable rate; {ok.T_era5.notna().sum():,} of those with a temperature; chirpers by bout test {int(ok.is_chirp.fillna(False).sum())}", flush=True)
    # the type lines
    Y = args.year; A = pd.read_csv(os.path.join(Y, f"animal_nights_streaks{args.suffix}.csv")); S = pd.read_csv(os.path.join(Y, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    P = pd.read_csv(os.path.join(Y, "streak_hourly_points_all.csv"), usecols=["an", "R", "F", "T"]).dropna(subset=["T"]).merge(A[["an", "streak"]], on="an"); ids = paper_ids.load(Y, args.suffix)
    big = [s for s in S.index if S.loc[s, "n"] >= 40 and S.loc[s, "form"] != "chirp"]; L = {}
    for s in big:
        g = P[P.streak == s]; x, y = np.log(g.R), np.log(g.F); b, a = np.polyfit(x, y, 1); sd = max(SD_ACROSS_FLOOR, float((y - (a + b * x)).std()))
        kT, aT = np.polyfit(g["T"], x, 1); L[s] = dict(a=a, b=b, sd=sd, kT=kT, aT=aT, sdT=max(0.08, float((x - (aT + kT * g["T"])).std())), lo=float(np.percentile(x, 1)), hi=float(np.percentile(x, 99)))
    rows = []
    for r in ok.itertuples():
        x, y = np.log(r.R), np.log(r.F_c); best = None
        for s, l in L.items():
            if not (l["lo"] - 0.15 <= x <= l["hi"] + 0.15):
                continue
            za = (y - (l["a"] + l["b"] * x)) / l["sd"]
            zl = (x - (l["aT"] + l["kT"] * r.T_era5)) / np.sqrt(l["sdT"] ** 2 + (l["kT"] * T_UNC) ** 2) if np.isfinite(r.T_era5) else np.nan
            score = abs(za) + (abs(zl) if np.isfinite(zl) else 0)
            if best is None or score < best[0]:
                best = (score, ids[s], za, zl, f"{S.loc[s, 'F25'] / 1000:.1f} kHz @ {S.loc[s, 'R25']:.0f} Hz")
        rows.append(dict(obs_id=r.obs_id, sound_id=r.sound_id, taxon=r.taxon, state=r.state, observed=r.observed, T=r.T_era5, F=r.F_c, R=r.R, depth=r.depth, is_chirp=r.is_chirp, chirp_rate=r.chirp_rate, license=r.license,
                         best_type=best[1] if best else None, type_position=best[4] if best else None, z_across=best[2] if best else np.nan, z_along=best[3] if best else np.nan,
                         on_line=bool(best and abs(best[2]) <= Z_ON and (not np.isfinite(best[3]) or abs(best[3]) <= Z_ON)) if best else False))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(args.root, "inat_vs_types.csv"), index=False, float_format="%.3f")
    print(f"on a type's line (across and, where a temperature exists, along): {int(R.on_line.sum()):,} of {len(R):,}", flush=True)
    sp = R.groupby("taxon").agg(n=("obs_id", "size"), n_T=("T", lambda v: int(v.notna().sum())), on=("on_line", "sum"), best=("best_type", lambda v: v.value_counts().index[0] if v.notna().any() else None), best_share=("best_type", lambda v: v.value_counts(normalize=True).iloc[0] if v.notna().any() else np.nan),
                                F_med=("F", "median"), R_med=("R", "median"), T_med=("T", "median"), states=("state", lambda v: ",".join(sorted(set(v))))).sort_values("n", ascending=False)
    sp.to_csv(os.path.join(args.root, "inat_species_table.csv"), float_format="%.2f"); pd.set_option("display.width", 220); print(sp.head(40).round(2).to_string())
    O = ok[["R", "F_c", "taxon", "T_era5"]].rename(columns={"R": "pulse_rate", "F_c": "carrier_kHz", "taxon": "species", "T_era5": "T"}); O["carrier_kHz"] = O.carrier_kHz / 1000; O["n_overlapping"] = 1
    os.makedirs(os.path.join(args.figures, "exploratory"), exist_ok=True); O.to_csv(os.path.join(args.figures, "exploratory", "inat_points.csv"), index=False, float_format="%.3f")
    print("wrote measurements.csv, inat_vs_types.csv, inat_species_table.csv and the overlay CSV", flush=True)


if __name__ == "__main__":
    main()
