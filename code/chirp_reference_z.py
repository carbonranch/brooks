"""The SINA reference clips on the chirp plane: z of every chirping clip against every chirp type's line, as
streak_id_figure.py does for the trill types on the syllable-rate plane (MBE, 2026-10-13: "and no chirpers?").

A chirp type's line is fitted to its singer-nights in log carrier against log chirp rate (chirp/animal_nights_streaks_c2),
and its chirp rate against air temperature to its singer-hours (chirp/streak_hourly_points_all). For a clip that the
bout test calls a chirper (sina_chirp_rates.py: contrast > 50x), z = its distance across the line in units of the line's
scatter (floor SD_FLOOR), z_along = (its log chirp rate - the rate the type chirps at the clip's temperature) / the type's
rate scatter at a given temperature widened by a T_UNC thermometer uncertainty; the best line is the smallest hypotenuse.

Writes <out>_reference_z.csv (same columns as F12's table: wav, name, species, T, R = chirp rate, F, streak, z, z_along, z_tot,
R_expected, T_in_range, best, paper_id) and <out>.png: the chirp plane with the lines, the chirping clips as points, and
one row per type with its clips by z_along (an exploratory twin of Figure 12 for the chirp types).

Usage:
  python analysis/callers/chirp_reference_z.py analysis/callers/runs/v3/year_nov24_nov25 --out analysis/callers/runs/figures_paper/explore/F12_chirp
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
from sina_range import in_range_mask, NEAR_STATES  # noqa: E402

Z_ON = 2.5
SD_FLOOR = 0.03
SD_R_FLOOR = 0.08
T_UNC = 2.0
MIN_N = 40


def short(name: str) -> str:
    g = name.split("|")[0].strip().split()
    return f"{g[0][0]}. {g[-1]}" if len(g) >= 2 else name


def fit_lines(A, S, H):
    lines = {}
    for sid in S.index:
        g = A[A.streak == sid]; x, y = np.log(g.R.to_numpy()), np.log(g.F.to_numpy())
        h = H[(H.streak == sid) & H["T"].notna()]
        bT, aT = np.polyfit(h["T"], np.log(h.R), 1)
        sdT = float(np.hypot(max((np.log(h.R) - (aT + bT * h["T"])).std(), SD_R_FLOOR), bT * T_UNC))
        if len(g) >= 8 and x.std() > 0.02:
            b, a = np.polyfit(x, y, 1)
        else:
            b, a = 0.3, np.log(g.F.median()) - 0.3 * np.log(g.R.median())
        resid = y - (a + b * x)
        lines[sid] = dict(a=a, b=b, sd=max(resid.std(), SD_FLOOR), x_lo=x.min() - 0.15, x_hi=x.max() + 0.15, x_min=np.percentile(x, 1), x_max=np.percentile(x, 99),
                          aT=aT, bT=bT, sdT=sdT, T_lo=h["T"].quantile(0.02), T_hi=h["T"].quantile(0.98))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); args = ap.parse_args()
    root = args.root; cdir = os.path.join(root, "chirp")
    S = pd.read_csv(os.path.join(cdir, "streaks_coarse_c2.csv"), index_col=0); S = S[S.n >= MIN_N]
    A = pd.read_csv(os.path.join(cdir, "animal_nights_streaks_c2.csv")); A = A[A.streak.isin(S.index)]
    H = pd.read_csv(os.path.join(cdir, "streak_hourly_points_all.csv"), usecols=["an", "R", "F", "T"]).merge(A[["an", "streak"]], on="an")
    cid = pd.read_csv(os.path.join(cdir, "paper_ids_chirp.csv")).set_index("streak").paper_id.to_dict()
    M = pd.read_csv(os.path.join(root, "sina", "sina_reference_chirp.csv"))
    Mm = pd.read_csv(os.path.join(root, "sina", "sina_reference_measurements.csv")).drop_duplicates("wav").set_index("wav")
    M["name"] = M.wav.map(Mm.name); M["group"] = M.wav.map(Mm.subfamily); M["page"] = M.wav.map(Mm.page); M["subfamily"] = M.wav.map(Mm.subfamily)
    M = M[(M.status == "ok") & M.is_chirp & M.F_c.notna() & M.chirp_rate.notna()].copy()
    M["in_range"] = in_range_mask(M, os.path.join(root, "sina")).to_numpy()
    print(f"{len(M)} chirping clips; {int(M.in_range.sum())} of species with records in the region ({NEAR_STATES}) or exempt small crickets")
    lines = fit_lines(A, S, H)
    rows = []
    for m in M.itertuples():
        x, y = np.log(m.chirp_rate), np.log(m.F_c)
        zs = {sid: (y - (L["a"] + L["b"] * x)) / L["sd"] for sid, L in lines.items() if L["x_lo"] <= x <= L["x_hi"]}
        za = {sid: (x - (lines[sid]["aT"] + lines[sid]["bT"] * m.T)) / lines[sid]["sdT"] for sid in zs}
        zt = {sid: float(np.hypot(zs[sid], za[sid])) for sid in zs}
        best = min(zt, key=zt.get) if zt else None
        for sid, z in zs.items():
            L = lines[sid]
            rows.append(dict(wav=m.wav, name=short(m.name), species=m.species, T=m.T, R=m.chirp_rate, F=m.F_c, group=m.group, streak=sid, z=z, z_along=za[sid], z_tot=zt[sid],
                             R_expected=float(np.exp(L["aT"] + L["bT"] * m.T)), T_in_range=bool(L["T_lo"] <= m.T <= L["T_hi"]), best=(sid == best), paper_id=cid.get(sid, sid), in_range=bool(m.in_range)))
        M.loc[m.Index, "zbest"] = abs(zs[best]) if best is not None else np.inf
    Z = pd.DataFrame(rows); Z.to_csv(args.out + "_reference_z_all.csv", index=False, float_format="%.3f")
    Z = Z[Z.in_range].copy(); Z.to_csv(args.out + "_reference_z.csv", index=False, float_format="%.3f")
    on = Z[Z.best & (Z.z.abs() <= Z_ON)]; both = on[on.z_along.abs() <= Z_ON]
    print(f"{len(M)} chirping clips; {on.wav.nunique()} within {Z_ON} sd of their best chirp line across it; {both.wav.nunique()} also within {Z_ON} sd along it; "
          f"{both.streak.nunique()} of {len(S)} chirp types have at least one")
    pd.set_option("display.width", 220)
    print(both.sort_values(["paper_id", "z_tot"])[["paper_id", "name", "T", "R", "F", "z", "z_along", "R_expected"]].to_string(index=False))

    # figure: plane + rows
    fig = plt.figure(figsize=(13, 9)); gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(H.R, H.F, s=0.4, color="0.85", lw=0, rasterized=True)
    for sid, L in lines.items():
        xx = np.linspace(L["x_min"], L["x_max"], 20); ax.plot(np.exp(xx), np.exp(L["a"] + L["b"] * xx), color="0.25", lw=0.8)
        ax.annotate(str(cid.get(sid, sid)), (np.exp(xx[-1]), np.exp(L["a"] + L["b"] * xx[-1])), fontsize=7, xytext=(3, 0), textcoords="offset points", va="center")
    outr = M[~M.in_range]; ax.scatter(outr.chirp_rate, outr.F_c, s=18, facecolors="none", edgecolors="0.6", lw=0.6, zorder=3, label=f"species without records in the region ({len(outr)})")
    Mi = M[M.in_range]; hit = Mi.zbest <= Z_ON
    ax.scatter(Mi.chirp_rate[hit], Mi.F_c[hit], s=22, color="tab:red", edgecolors="k", lw=0.4, zorder=4, label=f"in-range chirping clip within {Z_ON} sd of a line ({int(hit.sum())})")
    ax.scatter(Mi.chirp_rate[~hit], Mi.F_c[~hit], s=22, facecolors="none", edgecolors="tab:red", lw=0.8, zorder=3, label=f"other in-range chirping clips ({int((~hit).sum())})")
    for m in Mi[hit].itertuples():
        ax.annotate(short(m.name), (m.chirp_rate, m.F_c), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1.2, 12); ax.set_ylim(1500, 20000)
    ax.xaxis.set_major_locator(FixedLocator([1.5, 2, 3, 4, 5, 7, 10])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}")); ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("chirp rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("the chirp plane: the year's chirping singer-hours, the 15 chirp types' lines, the chirping SINA clips", fontsize=10, loc="left")
    ax2 = fig.add_subplot(gs[0, 1])
    order = S.sort_values("F25", ascending=False).index
    for i, sid in enumerate(order):
        zz = Z[(Z.streak == sid) & Z.best & (Z.z.abs() <= Z_ON)]
        ax2.axhspan(i - 0.5, i + 0.5, color="0.95" if i % 2 else "white", lw=0)
        ax2.axvspan(-1, 1, color="tab:orange", alpha=0.08, lw=0)
        for r in zz.itertuples():
            filled = abs(r.z) <= 1
            ax2.scatter(r.z_along, i, s=28, color="tab:red" if filled else "none", edgecolors="tab:red", lw=0.8, zorder=3)
            ax2.annotate(f"{r.name} {r.T:.0f}°", (r.z_along, i), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax2.set_yticks(range(len(order))); ax2.set_yticklabels([f"{cid.get(s, s)}  {S.loc[s, 'F25'] / 1000:.1f} kHz @ {S.loc[s, 'R25']:.1f} Hz" for s in order], fontsize=8)
    ax2.set_ylim(len(order) - 0.5, -0.5); ax2.set_xlim(-8, 8); ax2.axvline(0, color="k", lw=0.5)
    ax2.set_xlabel("z along the line (clip's chirp rate against the rate the type chirps at the clip's temperature)")
    ax2.set_title(f"clips on each chirp line (|z across| <= {Z_ON}; filled <= 1), placed by z along", fontsize=10, loc="left")
    fig.suptitle("Exploratory: SINA reference clips matched to the chirp types on the chirp plane (clip chirp rate from the bout test, sina_chirp_rates.py)", fontsize=10.5)
    fig.savefig(args.out + ".png", dpi=110, bbox_inches="tight"); print("wrote", args.out + ".png")


if __name__ == "__main__":
    main()
