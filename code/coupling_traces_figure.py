"""Figure 8 with the singers' nightly paths (MBE 2026-10-17: "update Figure 8 to have those same 6 types but to have a
subpanel for each that has the scatter plot on the left and the traces from the singer_night ... the traces with the
squiggles for each night"): six example tracks, one per chosen type, each as a subpanel = the coupling scatter (its
windows' carrier against rate as percent of the night median, colored by air temperature, the singer's fitted slope
black, the type's streak slope dashed red; coupling_figure's choice of example) and, to its right, the type on the plane
(its singer-hours gray) with the hourly paths of eight of its singers through their nights (singer_night_grid's choice:
the widest within-night temperature ranges, distinct recorder-nights), the example singer in black. Below, (B) the
coupling per type as in coupling_figure.

Usage:
  python analysis/callers/coupling_traces_figure.py analysis/callers/runs/v5/year_nov24_nov25 --suffix _v9 \
      --roots analysis/callers/runs/v5/y2024 analysis/callers/runs/v5/y2025 --examples 4301 3601 5501 5201 4601 4801 \
      --number 8 --out analysis/callers/runs/figures_paper/F08_coupling
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coupling_figure as CF  # noqa: E402
import paper_ids  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402

XLIM, YLIM, TLO, THI = 16.0, 10.0, 12.0, 32.0
N_PATHS = 7                      # other singers drawn beside the example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--examples", type=int, nargs="+", default=[4301, 3601, 5501, 5201, 4601, 4801]); ap.add_argument("--number", default="8"); ap.add_argument("--out", required=True)
    args = ap.parse_args(); root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv")); Tk = pd.read_csv(os.path.join(root, f"year_coupling_tracks{args.suffix}.csv"))
    C = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0); C = C[C.form != "chirp"] if "form" in C.columns else C
    ids = paper_ids.load(root, args.suffix); SL = pd.read_csv(os.path.join(root, f"streak_lines{args.suffix}.csv")).set_index("streak")
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["hour", "R", "F", "T", "an"]).dropna(subset=["T"]); P["hour"] = pd.to_datetime(P.hour, utc=True)
    J = A[["an", "streak", "T", "site", "night"]].merge(Tk, on="an").merge(C[["R25", "F25", "n"]], left_on="streak", right_index=True)
    J["rel"] = J.R_range / J.R25; J["nwin"] = J["n_x"] if "n_x" in J.columns else J["n"]
    good = J[(J.nwin >= CF.MIN_WIN) & J.rel.between(CF.REL_LO, CF.REL_HI) & J.slope.notna()]
    g = good.groupby("streak").slope.agg(n_tracks="size", within="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
    B = C[C.n >= CF.MIN_N].join(g, how="inner")
    ex = []
    for s in args.examples:
        cand = good[(good.streak == s) & (good.nwin >= CF.EX_MIN_WIN) & good.rel.between(*CF.EX_REL_RANGE)]
        if cand.empty or s not in B.index:
            print(f"no example track for {s}"); continue
        row = cand.assign(d=(cand.slope - B.loc[s, "within"]).abs()).sort_values(["d", "nwin"], ascending=[True, False]).iloc[0]; ex.append((s, row))
        print(f"example {s} (type {ids.get(s, s)}): {row.an}, {int(row.nwin)} windows, slope {row.slope:.2f} (type median {B.loc[s, 'within']:.2f})")

    page = PaperPage(number=args.number, title="Carrier follows rate within a singer")
    Wc, Hc = 7.2, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    X0, X1, TOP = fx0 + 0.42, fx0 + Wc - 0.08, 1.35
    COLG, HEAD, ROWG = 0.42, 0.20, 0.30
    sub_w = (X1 - X0 - COLG) / 2; SC_W = 1.05; PH = 0.95; PITCH = PH + HEAD + ROWG
    tnorm = Normalize(TLO, THI); tcmap = plt.get_cmap("turbo"); cols10 = plt.get_cmap("tab10"); first_ax = None; sc = None
    for i, (s, row) in enumerate(ex):
        r_, c_ = divmod(i, 2); x0 = X0 + c_ * (sub_w + COLG); top = TOP + r_ * PITCH + HEAD
        ax = fig.add_axes(rect(x0, x0 + SC_W, top, PH)); first_ax = first_ax or ax
        gt, unit = CF.load_track(row.an, args.roots)
        x = 100 * (gt.R / gt.R.median() - 1); y = 100 * (gt.F_fix / gt.F_fix.median() - 1)
        sc = ax.scatter(x, y, c=gt["T"], cmap=tcmap, norm=tnorm, s=6, lw=0.2, edgecolors="k", zorder=2)
        lx, ly = np.log(gt.R), np.log(gt.F_fix); b, a = np.polyfit(lx, ly, 1)
        xx = np.linspace(x.min(), x.max(), 50); rr = gt.R.median() * (1 + xx / 100); ax.plot(xx, 100 * (np.exp(a + b * np.log(rr)) / gt.F_fix.median() - 1), color="k", lw=0.9, zorder=3)
        bs = float(SL.loc[s, "slope"]) if s in SL.index else np.nan
        if np.isfinite(bs):
            xs = np.linspace(-XLIM, XLIM, 20); ax.plot(xs, 100 * ((1 + xs / 100) ** bs - 1), color="tab:red", lw=0.8, ls="--", zorder=2)
        ax.plot([-XLIM, XLIM], [-XLIM, XLIM], color="0.5", lw=0.5, ls=":", zorder=1); ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
        ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-YLIM, YLIM); ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, -5, 0, 5, 10]); ax.tick_params(labelsize=BODY_PT - 1, length=1.5, pad=1.5)
        ax.set_xlabel("rate (% of night median)", fontsize=BODY_PT - 1, labelpad=1); ax.set_ylabel("carrier (%)", fontsize=BODY_PT - 1, labelpad=1)
        ax.text(0.97, 0.04, f"singer {b:.2f}\nstreak {bs:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=BODY_PT - 1, linespacing=1.1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        site = unit.split("_")[2]; night = pd.Timestamp(unit.split("_")[3]).strftime("%d %b %Y")
        fig.text((x0 - fx0) / Wc, 1 - (top - HEAD - 1.0) / Hc, f"Type {ids.get(s, s)}, {gt.F_fix.median() / 1000:.1f} kHz @ {gt.R.median():.0f} Hz; {site}, {night}", fontsize=BODY_PT + 0.5, fontweight="bold", va="top", ha="left")
        # the paths: the type's singer-hours gray, N_PATHS singers with the widest within-night temperature range as hourly paths, this singer black
        axp = fig.add_axes(rect(x0 + SC_W + 0.42, x0 + sub_w, top, PH))
        Hs = P[P.an.isin(A[A.streak == s].an)]
        axp.scatter(Hs.R, Hs.F, s=0.5, color="0.85", lw=0, rasterized=True)
        st = Hs.groupby("an").agg(n=("T", "size"), dT=("T", lambda v: v.max() - v.min())).join(A.set_index("an")[["depth", "recorder", "night"]])
        st = st[(st.n >= 6) & (st.depth >= st.depth.quantile(0.4))].sort_values("dT", ascending=False)
        picks, used = [], set()
        for an_, r2 in st.iterrows():
            if (r2.recorder, r2.night) in used or an_ == row.an:
                continue
            used.add((r2.recorder, r2.night)); picks.append(an_)
            if len(picks) >= N_PATHS:
                break
        for k, an_ in enumerate(picks):
            h_ = Hs[Hs.an == an_].sort_values("hour"); axp.plot(h_.R, h_.F, color=cols10(k % 10), lw=0.7, alpha=0.85); axp.scatter(h_.R.iloc[[0]], h_.F.iloc[[0]], s=4, color=cols10(k % 10), zorder=5)
        h0 = Hs[Hs.an == row.an].sort_values("hour"); axp.plot(h0.R, h0.F, color="k", lw=1.3, zorder=6); axp.scatter(h0.R.iloc[[0]], h0.F.iloc[[0]], s=9, color="k", zorder=7)
        Hp = pd.concat([Hs[Hs.an.isin(picks)], h0]); xlo, xhi = Hp.R.min() / 1.25, Hp.R.max() * 1.25; ylo, yhi = Hp.F.min() / 1.15, Hp.F.max() * 1.15
        axp.set_xscale("log"); axp.set_yscale("log"); axp.set_xlim(xlo, xhi); axp.set_ylim(ylo, yhi)
        xt = [t for t in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100] if xlo <= t <= xhi]; yt = [t for t in [1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000] if ylo <= t <= yhi]
        axp.xaxis.set_major_locator(FixedLocator(xt)); axp.yaxis.set_major_locator(FixedLocator(yt)); axp.xaxis.set_minor_locator(FixedLocator([])); axp.yaxis.set_minor_locator(FixedLocator([]))
        axp.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axp.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axp.tick_params(labelsize=BODY_PT - 1, length=1.5, pad=1.5); axp.set_xlabel("syllable rate (Hz)", fontsize=BODY_PT - 1, labelpad=1); axp.set_ylabel("carrier (kHz)", fontsize=BODY_PT - 1, labelpad=1); axp.grid(alpha=0.25, lw=0.4)
        for sp in axp.spines.values():
            sp.set_linewidth(0.5)
        axp.set_title("eight singers' nightly paths; this one black", fontsize=BODY_PT - 1.5, loc="left", pad=2)
    page.label(first_ax, "A", "Six singers through a night, and the nightly paths of singers of their types (dot = first hour)", y=1 + (HEAD + 0.08) / PH)
    cb_top = TOP + 3 * PITCH - ROWG + 0.30
    cax = fig.add_axes(rect(X1 - 1.7, X1 - 0.1, cb_top, 0.07))      # under the right column, clear of the scatter's x label
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("air temperature at the window (°C)", fontsize=BODY_PT - 1, labelpad=1); cb.ax.tick_params(labelsize=BODY_PT - 1, length=1.5)

    # (B)
    b_top = cb_top + 0.55; b_h = Hc - (b_top - 1.0) - 0.15
    ax = fig.add_axes(rect(X0, X1, b_top, b_h))
    ax.errorbar(B.F25, B.within, yerr=[B.within - B.q25, B.q75 - B.within], fmt="o", color="0.25", ms=3.5, elinewidth=0.7, capsize=1.5, alpha=0.9)
    for r in B.itertuples():
        ax.annotate(str(ids.get(r.Index, r.Index)), (r.F25, r.within), fontsize=5, xytext=(3, 2), textcoords="offset points", color="0.25")
    common = [s for s in B.index if s in SL.index]
    if common:
        ax.scatter(B.loc[common, "F25"], SL.loc[common, "slope"], s=22, marker="_", color="tab:red", lw=1.2, zorder=4, label="streak slope (Figure 7C)")
        ax.scatter([], [], s=12, color="0.25", label="within-singer coupling, median and IQR over tracks"); ax.legend(fontsize=5, loc="upper right", frameon=False)
    ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="k", lw=0.5, ls=":"); ax.set_xscale("log"); ax.set_xlim(1800, 10000)
    ax.xaxis.set_major_locator(FixedLocator([2000, 3000, 4000, 5000, 6000, 8000, 10000])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_xlabel("type carrier at 25 °C (kHz)"); ax.set_ylabel("coupling within singers (% carrier per % rate)"); ax.grid(alpha=0.3, which="major", lw=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    page.label(ax, "B", "Coupling per type", y=1 + 0.05 / b_h)
    page.finish(fig, args.out); print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
