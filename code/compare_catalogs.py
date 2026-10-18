"""Compare two year catalogs (e.g. v3 and v4 after a relink): match each type of the first to the nearest type of the
second by 25 C position (log carrier, log rate), and tabulate counts, positions, streak slopes and within-singer
slopes side by side, plus the types without a partner. Exploratory (2026-10-15, the v4 relink; METHOD 19 AV).

Usage:
  python analysis/callers/compare_catalogs.py analysis/callers/runs/v3/year_nov24_nov25 analysis/callers/runs/v4/year_nov24_nov25 \
      --suffix _v9 --out analysis/callers/runs/figures_paper/exploratory/catalog_v3_vs_v4
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ids  # noqa: E402

MIN_N = 40
MATCH_TOL = 0.08          # log units of combined distance for a match


def load(root: str, suffix: str) -> pd.DataFrame:
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{suffix}.csv"), index_col=0)
    S = S[(S.n >= MIN_N) & (S.form != "chirp")].copy()
    ids = paper_ids.load(root, suffix); S["pid"] = [ids.get(s, s) for s in S.index]
    lp = os.path.join(root, f"streak_lines{suffix}.csv")
    if os.path.exists(lp):
        L = pd.read_csv(lp).set_index("streak"); S["streak_slope"] = L.slope.reindex(S.index)
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{suffix}.csv"))
    tp = os.path.join(root, f"year_coupling_tracks{suffix}.csv")
    if os.path.exists(tp):
        Tk = pd.read_csv(tp); J = A[["an", "streak"]].merge(Tk, on="an"); J = J[(J.n >= 12)]
        S["within_slope"] = J.groupby("streak").slope.median().reindex(S.index); S["n_tracks"] = J.groupby("streak").size().reindex(S.index)
    return S


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs=2, default=["v3", "v4"])
    args = ap.parse_args()
    A, B = load(args.a, args.suffix), load(args.b, args.suffix); na, nb = args.names
    xa = np.column_stack([np.log(A.R25), np.log(A.F25)]); xb = np.column_stack([np.log(B.R25), np.log(B.F25)])
    rows = []; used = set()
    for i, (sa, ra) in enumerate(A.iterrows()):
        d = np.hypot(*(xb - xa[i]).T); j = int(np.argmin(d))
        sb = B.index[j] if d[j] <= MATCH_TOL and B.index[j] not in used else None
        if sb is not None:
            used.add(sb)
        rb = B.loc[sb] if sb is not None else None
        rows.append({f"{na}_type": ra.pid, f"{na}_F25": int(ra.F25), f"{na}_R25": round(ra.R25, 1), f"{na}_n": int(ra.n),
                     f"{nb}_type": rb.pid if rb is not None else "", f"{nb}_F25": int(rb.F25) if rb is not None else np.nan, f"{nb}_R25": round(rb.R25, 1) if rb is not None else np.nan,
                     f"{nb}_n": int(rb.n) if rb is not None else np.nan, "dist": round(d[j], 3),
                     f"{na}_streak_slope": ra.get("streak_slope", np.nan), f"{nb}_streak_slope": rb.get("streak_slope", np.nan) if rb is not None else np.nan,
                     f"{na}_within": ra.get("within_slope", np.nan), f"{nb}_within": rb.get("within_slope", np.nan) if rb is not None else np.nan})
    D = pd.DataFrame(rows).sort_values(f"{na}_F25")
    extra = B[~B.index.isin(used)]
    pd.set_option("display.width", 250)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{nb} types without a {na} partner:")
    print(extra[["pid", "F25", "R25", "n"] + [c for c in ("streak_slope", "within_slope") if c in extra.columns]].to_string(float_format=lambda v: f"{v:.2f}"))
    D.to_csv(args.out + ".csv", index=False, float_format="%.3f"); extra.to_csv(args.out + "_unmatched.csv")
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax = axs[0]
    ax.scatter(A.R25, A.F25, s=A.n.clip(40, 4000) / 25, facecolors="none", edgecolors="k", label=na); ax.scatter(B.R25, B.F25, s=B.n.clip(40, 4000) / 25, color="tab:red", alpha=0.5, lw=0, label=nb)
    for r in A.itertuples():
        ax.annotate(str(r.pid), (r.R25, r.F25), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rate at 25 °C (Hz)"); ax.set_ylabel("carrier at 25 °C (Hz)"); ax.legend(); ax.set_title("type positions (area = singer-nights)"); ax.grid(alpha=0.3)
    m = D.dropna(subset=[f"{nb}_within"])
    ax = axs[1]; ax.scatter(m[f"{na}_within"], m[f"{nb}_within"], color="k", s=14)
    for r in m.itertuples():
        ax.annotate(str(getattr(r, f"{na}_type")), (getattr(r, f"{na}_within"), getattr(r, f"{nb}_within")), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.plot([0, 0.8], [0, 0.8], color="0.5", lw=0.5); ax.set_xlabel(f"within-singer slope, {na}"); ax.set_ylabel(f"within-singer slope, {nb}"); ax.set_title("within-singer coupling, matched types"); ax.grid(alpha=0.3)
    ax = axs[2]; ax.scatter(m[f"{na}_streak_slope"], m[f"{nb}_streak_slope"], color="k", s=14)
    for r in m.itertuples():
        ax.annotate(str(getattr(r, f"{na}_type")), (getattr(r, f"{na}_streak_slope"), getattr(r, f"{nb}_streak_slope")), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.plot([0.3, 0.7], [0.3, 0.7], color="0.5", lw=0.5); ax.set_xlabel(f"streak slope, {na}"); ax.set_ylabel(f"streak slope, {nb}"); ax.set_title("streak slopes, matched types"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=120); print(f"wrote {args.out}.png / .csv")


if __name__ == "__main__":
    main()
