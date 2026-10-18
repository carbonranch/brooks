"""Stage three of the streak definition: merge neighbouring streaks whose
boundary the data do not support.

First version (2026-09-22) let merged groups grow their across-line sd and
so chain: on the year it merged the Gryllus line into the 5.4 kHz @ 40 Hz
line (19 merges, 67 -> 48). Now: candidates must be within MAX_PERP = 5 %
across (real neighbours sit 7-12 % apart), the across sd is capped at
SD_CAP, and a strongly supported two-band split along the line (dBIC >= 50,
separation >= 2 sd) keeps a pair apart.

For every pair of streaks whose singer-hours are adjacent in the plane
(across-line offset < MAX_PERP, along-line ranges overlapping or within
MAX_GAP), pool the two and ask whether a two-band model of position-at-25C
is required, exactly as split_streaks_by_T.py asks of a single broad
streak. If a single band explains the pooled points (BIC does not improve
by DBIC, or the two components are closer than SEP_SD pooled sd, or the
across-line offset between the two is below PERP_SD times the pooled
across-line scatter), the pair is one population and is merged. Repeats to
convergence, largest pairs first, and re-labels.

Usage:
    python analysis/callers/merge_streaks.py analysis/callers/runs/month_v2 --in _v4 --out _v6 --coupling 0.55
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

MAX_PERP = 0.05      # log-carrier: only neighbours this close across the line may merge (real neighbouring
                     # species sit 7-12 % apart with ~3 % scatter; Sept 2024 and the year agree)
MAX_GAP = 0.05       # log-rate: along-line ranges may be separated by at most this
DBIC = 50.0
SEP_SD = 2.0         # two along-line bands this far apart (pooled sd) with dBIC >= DBIC keep a pair apart
PERP_SD = 2.5        # across-line offset in units of across-line sd needed to keep two streaks apart
SD_CAP = 0.03        # across-line sd is capped here so a wide merged group cannot swallow its neighbours (no chaining)
MIN_N = 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--in", dest="sin", default="_v4")
    ap.add_argument("--out", dest="sout", default="_v6")
    ap.add_argument("--coupling", type=float, default=0.55)
    ap.add_argument("--coupling-hf", type=float, default=0.25)
    ap.add_argument("--hf-split", type=float, default=10000.0)
    ap.add_argument("--slope", type=float, default=2.4, help="%%/degC used to bring positions to 25 C")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.sin}.csv"))
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv")).drop(columns=["streak"]).merge(A[["an", "streak"]], on="an")
    P = P[P.streak >= 0].copy()

    def frame(F):
        k = args.coupling if F < args.hf_split else args.coupling_hf
        u = np.array([1.0, k]) / np.hypot(1.0, k)
        return u, np.array([-u[1], u[0]])

    def coords(g):
        u, nrm = frame(g.F.median())
        Z = np.column_stack([np.log(g.R), np.log(g.F)])
        along = (Z @ u) / u[0] + args.slope / 100 * (25 - g["T"].to_numpy())     # log-rate equivalent at 25 C
        perp = (Z @ nrm) / nrm[1]                                                  # log-carrier equivalent
        return along, perp

    labels = P.streak.to_numpy().copy()
    merged_log = []
    while True:
        P["lab"] = labels
        stats = {}
        for sid, g in P.groupby("lab"):
            al, pp = coords(g)
            stats[sid] = dict(n=len(g), a_lo=np.percentile(al, 10), a_hi=np.percentile(al, 90), a_med=np.median(al), p_med=np.median(pp), p_sd=pp.std(), F=g.F.median(), R=g.R.median())
        sids = sorted(stats, key=lambda s: -stats[s]["n"])
        best = None
        for i, a in enumerate(sids):
            for b in sids[i + 1:]:
                sa, sb = stats[a], stats[b]
                dperp = abs(sa["p_med"] - sb["p_med"])
                if dperp > MAX_PERP:
                    continue
                gap = max(sa["a_lo"], sb["a_lo"]) - min(sa["a_hi"], sb["a_hi"])
                if gap > MAX_GAP:
                    continue
                pooled_sd = min(np.sqrt((sa["p_sd"] ** 2 + sb["p_sd"] ** 2) / 2), SD_CAP)
                perp_z = dperp / max(pooled_sd, 1e-6)
                g = P[P.lab.isin([a, b])]
                al, _ = coords(g)
                x = al.reshape(-1, 1)
                m1, m2 = GaussianMixture(1).fit(x), GaussianMixture(2, n_init=3, random_state=0).fit(x)
                dbic = m1.bic(x) - m2.bic(x)
                mu = np.sort(m2.means_.ravel()); sd = np.sqrt(m2.covariances_.ravel())
                sep = (mu[1] - mu[0]) / np.sqrt((sd ** 2).mean())
                two_bands_along = dbic >= DBIC and sep >= SEP_SD
                two_across = perp_z >= PERP_SD
                if not two_bands_along and not two_across:
                    score = perp_z + max(0, sep - 1)     # weakest boundary first
                    if best is None or score < best[0]:
                        best = (score, a, b, dperp, perp_z, dbic, sep)
        if best is None:
            break
        _, a, b, dperp, perp_z, dbic, sep = best
        keep, drop = (a, b) if stats[a]["n"] >= stats[b]["n"] else (b, a)
        merged_log.append(dict(kept=keep, merged=drop, n_kept=stats[keep]["n"], n_merged=stats[drop]["n"], F_kept=stats[keep]["F"], R_kept=stats[keep]["R"],
                               F_merged=stats[drop]["F"], R_merged=stats[drop]["R"], dperp_pct=100 * dperp, perp_z=perp_z, dbic_along=dbic, sep_along_sd=sep))
        labels[labels == drop] = keep
        print(f"merge {drop} ({stats[drop]['F']:.0f} Hz @ {stats[drop]['R']:.1f}, n={stats[drop]['n']}) into {keep} ({stats[keep]['F']:.0f} @ {stats[keep]['R']:.1f}, n={stats[keep]['n']}): "
              f"across {100 * dperp:.1f} % = {perp_z:.1f} sd; along dBIC {dbic:.0f}, sep {sep:.1f} sd", flush=True)

    # write in the build_streaks format, relabelling singer-nights
    an_lab = P.groupby("an").lab.first()
    A2 = A.copy()
    A2["streak"] = A2.an.map(an_lab).fillna(A2.streak).astype(int)
    g = A2[A2.streak >= 0].groupby("streak")
    T = g.agg(n=("an", "size"), F25=("F25", "median"), F25_iqr=("F25", lambda s: s.quantile(.75) - s.quantile(.25)),
              R25=("R25", "median"), R25_iqr=("R25", lambda s: s.quantile(.75) - s.quantile(.25)),
              F_min=("F", "min"), F_max=("F", "max"), R_min=("R", "min"), R_max=("R", "max"),
              n_sites=("site", "nunique"), n_nights=("night", "nunique"), depth=("depth", "median"))
    T = T[T.n >= MIN_N].sort_values("F25")
    # carrier harmonics that escaped the within-window linkage: a streak at
    # m x a lower streak's carrier (m = 2, 3; 6 %) with the same rate (3 %)
    # is that streak's harmonic, not a sonospecies
    T["harmonic_of"] = -1
    for sid, r in T.iterrows():
        for m in (2, 3):
            cand = T[(T.index != sid) & (np.abs(T.R25 / r.R25 - 1) < 0.03) & (np.abs(T.F25 * m / r.F25 - 1) < 0.06) & (T.n >= r.n)]
            if len(cand):
                T.loc[sid, "harmonic_of"] = int(cand.n.idxmax())
                break
    nh = int((T.harmonic_of >= 0).sum())
    for sid, r in T[T.harmonic_of >= 0].iterrows():
        print(f"harmonic: {sid} ({r.F25:.0f} Hz @ {r.R25:.1f}, n={int(r.n)}) = {round(r.F25 / T.loc[r.harmonic_of, 'F25'])} x {r.harmonic_of} ({T.loc[r.harmonic_of, 'F25']:.0f} @ {T.loc[r.harmonic_of, 'R25']:.1f})")
    T.to_csv(os.path.join(root, f"streaks_coarse{args.sout}.csv"), float_format="%.3f")
    print(f"{nh} streaks flagged as carrier harmonics; {int((T.harmonic_of < 0).sum())} sonospecies-level streaks with >= {MIN_N} singer-nights")
    A2.to_csv(os.path.join(root, f"animal_nights_streaks{args.sout}.csv"), index=False, float_format="%.3f")
    pd.DataFrame(merged_log).to_csv(os.path.join(root, f"streak_merges{args.sout}.csv"), index=False, float_format="%.3f")
    n0 = A[A.streak >= 0].streak.nunique()
    print(f"\n{n0} streaks -> {A2[A2.streak >= 0].streak.nunique()} after {len(merged_log)} merges; {len(T)} with >= {MIN_N} singer-nights; wrote streaks_coarse{args.sout}.csv, animal_nights_streaks{args.sout}.csv, streak_merges{args.sout}.csv")


if __name__ == "__main__":
    main()
