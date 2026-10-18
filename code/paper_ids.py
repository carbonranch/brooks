"""Paper-facing type numbers over the pipeline's streak ids.

The pipeline's four-digit streak ids (100 x the stage-one cluster label + a component digit, + 50 for a chirp type)
are keys into the run's files and change with every re-clustering; only their last two digits mean anything. For
the paper every TRILL type gets a plain number in order of carrier at 25 degC, lowest first: the types with at
least MIN_N singer-periods take 1..K, the smaller ones continue from K+1 in the same order, so supplement pages can
still cite them. Chirp types are numbered C1..Cn on the chirp plane (chirp_year.py), so the main plane's chirp-form
ids are left out here (2026-10-11).

    python analysis/callers/paper_ids.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v9
writes <root>/paper_ids<suffix>.csv (streak, paper_id, F25, R25, n, form, candidate). Figure scripts read it with
paper_ids.load(root, suffix) -> dict streak -> paper_id, and label(S, sid, ids) for the standard "3  3.6 kHz @ 36 Hz".
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

MIN_N = 40


def assign(S: pd.DataFrame, min_n: int = MIN_N, trill_only: bool = True) -> pd.DataFrame:
    S = S[(S.harmonic_of < 0) if "harmonic_of" in S.columns else slice(None)].copy()
    if trill_only and "form" in S.columns:          # chirp types are numbered on the chirp plane (chirp_year.py: C1..)
        S = S[S.form != "chirp"]
    big = S[S.n >= min_n].sort_values("F25"); small = S[S.n < min_n].sort_values("F25")
    order = list(big.index) + list(small.index)
    return pd.DataFrame({"streak": order, "paper_id": np.arange(1, len(order) + 1), "F25": S.loc[order, "F25"].round(0).astype(int).values,
                         "R25": S.loc[order, "R25"].round(1).values, "n": S.loc[order, "n"].astype(int).values,
                         "form": S.loc[order, "form"].values if "form" in S.columns else ""})


def path(root: str, suffix: str) -> str:
    return os.path.join(root, f"paper_ids{suffix}.csv")


def load(root: str, suffix: str) -> dict:
    """streak id -> paper number (builds the table if it is missing)."""
    p = path(root, suffix)
    if not os.path.exists(p):
        S = pd.read_csv(os.path.join(root, f"streaks_coarse{suffix}.csv"), index_col=0)
        assign(S).to_csv(p, index=False)
    T = pd.read_csv(p)
    def _id(v):                                     # plain numbers stay ints; a band label like "5a" (exploratory split catalogs) stays a string
        try:
            return int(v)
        except (TypeError, ValueError):
            return str(v)
    return dict(zip(T.streak.astype(int), [_id(v) for v in T.paper_id]))


def short_species(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    g = name.split("|")[0].strip().split()
    return f"{g[0][0]}. {g[-1]}" if len(g) >= 2 else name


def label(S: pd.DataFrame, sid: int, ids: dict, catalog: pd.DataFrame | None = None, species: bool = False) -> str:
    """'3  3.6 kHz @ 36 Hz' (+ ' (O. texensis)' when species and the catalog accepts a name)."""
    s = f"{ids.get(sid, sid)}  {S.loc[sid, 'F25'] / 1000:.1f} kHz @ {S.loc[sid, 'R25']:.0f} Hz"
    if species and catalog is not None and sid in catalog.index:
        st = catalog.candidate_status.get(sid, ""); nm = catalog.candidate.get(sid, np.nan)
        if isinstance(st, str) and st in ("recording", "position") and isinstance(nm, str):
            s += f" ({short_species(nm)})"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    S = pd.read_csv(os.path.join(args.root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    cp = os.path.join(args.root, f"type_catalog{args.suffix}.csv")
    T = assign(S, args.min_n)
    if os.path.exists(cp):
        C = pd.read_csv(cp).set_index("type")
        T["candidate"] = [short_species(C.candidate.get(s, np.nan)) if C.candidate_status.get(s, "") in ("recording", "position") else "" for s in T.streak]
    T.to_csv(path(args.root, args.suffix), index=False)
    print(T.to_string(index=False)); print(f"wrote {path(args.root, args.suffix)}")


if __name__ == "__main__":
    main()
