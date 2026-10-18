"""Why is the streak slope steeper than the within-singer coupling for the high-carrier types? Two tests
(MBE, 2026-10-13):
  1. same recorder, same night: pairs of tracks of one type that overlap in time on one recorder share the air
     temperature, so the covariance of their carrier and rate offsets is between-individual, not temperature.
     Per type: the slope of log F offset on log R offset over such pairs (offsets from the pair's mean), and the
     correlation; compared with the within-track coupling and the streak slope.
  2. syllable duration (pulse_shapes: dur_ms per singer-period) against carrier BETWEEN singers of a type at matched
     temperature (residuals of both on T), and within tracks where the per-window fold gives it (not available per
     window here, so the between-singer test only): a size signature would make shorter syllables go with higher
     carriers.
Writes explore/streak_slope_origin.csv and .png.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_ids  # noqa: E402

R = "analysis/callers/runs/v3/year_nov24_nov25"
OUT = "analysis/callers/runs/figures_paper/explore/streak_slope_origin"


def main() -> None:
    A = pd.read_csv(f"{R}/animal_nights_streaks_v9.csv"); S = pd.read_csv(f"{R}/streaks_coarse_v9.csv", index_col=0)
    P = pd.read_csv(f"{R}/streak_hourly_points_all.csv").drop(columns=["streak"]).merge(A[["an", "streak"]], on="an", how="left")
    P["hour"] = pd.to_datetime(P.hour, utc=True)
    Tk = pd.read_csv(f"{R}/year_coupling_tracks_v9.csv").set_index("an"); SL = pd.read_csv(f"{R}/streak_lines_v9.csv").set_index("streak")
    Pq = pd.read_csv(f"{R}/pulse_shapes_v9.csv"); Pq = Pq[Pq.status == "ok"].set_index("an")
    ids = paper_ids.load(R, "_v9")
    big = [s for s in S.index if S.loc[s, "n"] >= 40 and S.loc[s, "form"] != "chirp"]
    rows = []
    for s in big:
        g = P[P.streak == s].copy(); g["unit"] = g.an.str.split(":").str[0]
        # test 1: simultaneous pairs on one recorder-night
        dx, dy = [], []
        for u, gu in g.groupby("unit"):
            tracks = gu.an.unique()
            if len(tracks) < 2:
                continue
            byh = gu.groupby(["hour", "an"]).agg(lr=("R", lambda v: np.log(v).mean()), lf=("F", lambda v: np.log(v).mean())).reset_index()
            for h, gh in byh.groupby("hour"):
                if len(gh) < 2:
                    continue
                lr, lf = gh.lr.to_numpy(), gh.lf.to_numpy()
                dx += list(lr - lr.mean()); dy += list(lf - lf.mean())
        dx, dy = np.array(dx), np.array(dy)
        same = dict(n_pairs_hours=len(dx), slope_same_night=np.nan, r_same_night=np.nan)
        if len(dx) >= 30 and dx.std() > 0:
            same["slope_same_night"] = float(np.polyfit(dx, dy, 1)[0]); same["r_same_night"] = float(np.corrcoef(dx, dy)[0, 1])
        # test 2: syllable duration against carrier between singers, temperature removed
        m = g.groupby("an").agg(lf=("F", lambda v: np.log(v).median()), lr=("R", lambda v: np.log(v).median()), T=("T", "median")).join(Pq[["dur_ms"]].rename(columns={"dur_ms": "dur_med"}), how="inner")
        m = m[m.dur_med.notna() & (m.dur_med > 0)]
        dur = dict(n_dur=len(m), slope_dur_on_F=np.nan, r_dur_F=np.nan, r_dur_R=np.nan)
        if len(m) >= 20:
            rf = m.lf - np.polyval(np.polyfit(m["T"], m.lf, 1), m["T"]); rd = np.log(m.dur_med) - np.polyval(np.polyfit(m["T"], np.log(m.dur_med), 1), m["T"])
            rr = m.lr - np.polyval(np.polyfit(m["T"], m.lr, 1), m["T"])
            dur["slope_dur_on_F"] = float(np.polyfit(rf, rd, 1)[0]); dur["r_dur_F"] = float(np.corrcoef(rf, rd)[0, 1]); dur["r_dur_R"] = float(np.corrcoef(rr, rd)[0, 1])
        within = float(Tk.loc[Tk.index.isin(g.an.unique()), "slope"].median())
        rows.append(dict(type=int(ids.get(s, s)), F25=round(S.loc[s, "F25"]), streak=round(float(SL.loc[s, "slope"]), 2), within=round(within, 2), **same, **dur))
    T = pd.DataFrame(rows).sort_values("F25")
    for c in ("slope_same_night", "r_same_night", "slope_dur_on_F", "r_dur_F", "r_dur_R"):
        T[c] = T[c].round(2)
    pd.set_option("display.width", 220); print(T.to_string(index=False))
    lo, hi = T[T.F25 < 4000], T[T.F25 >= 4000]
    for name, q in (("<4 kHz", lo), (">=4 kHz", hi)):
        print(f"{name}: streak {q.streak.median():.2f}, within {q.within.median():.2f}, same-night between-singer slope {q.slope_same_night.median():.2f} (r {q.r_same_night.median():.2f}); "
              f"duration on carrier (T removed) slope {q.slope_dur_on_F.median():.2f}, r {q.r_dur_F.median():.2f}; duration vs rate r {q.r_dur_R.median():.2f}")
    T.to_csv(OUT + ".csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(T.within, T.slope_same_night, s=30, color="tab:red", label="between singers, same recorder-night (air T shared)", zorder=3)
    ax.scatter(T.within, T.streak, s=30, facecolors="none", edgecolors="0.3", label="streak slope (Figure 7C)", zorder=3)
    for r in T.itertuples():
        ax.annotate(str(r.type), (r.within, r.slope_same_night), fontsize=7, xytext=(3, 2), textcoords="offset points", color="tab:red")
    ax.plot([0, 0.8], [0, 0.8], "k:", lw=0.8); ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.9)
    ax.set_xlabel("within-singer coupling (median over tracks)"); ax.set_ylabel("slope of carrier offset on rate offset"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("test 1: simultaneous singers of one type on one recorder", loc="left", fontsize=10, fontweight="bold")
    ax = axes[1]
    ax.scatter(T.F25, T.r_dur_F, s=30, color="tab:blue", label="syllable duration vs carrier", zorder=3)
    ax.scatter(T.F25, T.r_dur_R, s=30, color="tab:orange", label="syllable duration vs rate", zorder=3)
    for r in T.itertuples():
        ax.annotate(str(r.type), (r.F25, r.r_dur_F), fontsize=7, xytext=(3, 2), textcoords="offset points", color="tab:blue")
    ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log"); ax.set_xlim(1800, 11000); ax.set_ylim(-1, 1)
    ax.set_xlabel("type carrier at 25 °C (Hz)"); ax.set_ylabel("correlation between singers, temperature removed"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("test 2: syllable duration against carrier and rate between singers", loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT + ".png", dpi=110); print("wrote", OUT)


if __name__ == "__main__":
    main()
