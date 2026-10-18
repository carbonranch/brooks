"""Figure 14: the day singers (house style). Figures 7 to 12 are night only; the detector and linker run on the day
periods too (07:00 to 19:00), and the day singer-periods are assigned to the night catalog's lines where they fall on one.

  (A) every day singer-hour on the carrier against syllable-rate plane, colored by air temperature (Fig 7A's axes)
  (B) the same hours by type: night types with >= MIN_N day singer-periods in their paper numbers, day-dominant types
      the night catalog does not number as D1, D2 ... by carrier; the rest gray
  (C) singers tracked at once by hour of the day, stacked by type, over the day months (recorded day hours from the
      day units' per-window files)
  (D) the day share of each type, day singer-periods over all singer-periods, against its carrier

Usage:
  python analysis/callers/day_plane_figure.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --out analysis/callers/runs/figures_paper/F14_day
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

TZ = "America/Chicago"
MIN_N = 40
FLO, FHI = 1500.0, 20000.0
RLO, RHI = 10.0, 100.0
TLO, THI = 12.0, 36.0


def style(ax):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(RLO, RHI); ax.set_ylim(FLO, FHI)
    ax.xaxis.set_major_locator(FixedLocator([10, 15, 20, 30, 50, 70, 100])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000])); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_xlabel("syllable rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(True, which="major", alpha=0.25)


def recorded_day_hours_by_hour(roots, first, last, cache):
    """Recorded day windows per local hour (07..18) over the period, from the day units' per-window files."""
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0).iloc[:, 0]
    counts = pd.Series(0, index=range(7, 20), dtype=float)
    for r in roots:
        for d in sorted(glob.glob(os.path.join(r, "BARLT_*_day"))):
            m = re.search(r"_(\d{8})_day$", d)
            if not m or not (first <= pd.Timestamp(m.group(1)) <= last):
                continue
            for fn in os.listdir(d):
                mm = re.match(r"BARLT_\d+_[A-Za-z0-9]+_\d{8}_(\d{2})\d{2}_day\.csv$", fn)
                if mm:
                    counts[int(mm.group(1))] += 300 / 3600
    counts.to_csv(cache)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--first", default="2024-11-01"); ap.add_argument("--last", default="2025-11-30"); ap.add_argument("--number", default="14"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root; first, last = pd.Timestamp(args.first), pd.Timestamp(args.last)
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    D = pd.read_csv(os.path.join(root, f"animal_days_streaks{args.suffix}.csv"))
    H = pd.read_csv(os.path.join(os.path.dirname(root.rstrip("/")), "year_figs", "nov24_nov25_day_hours.csv"))
    H["hour"] = pd.to_datetime(H.hour, utc=True).dt.tz_convert(TZ)
    H = H.merge(D[["an", "streak"]], left_on="tid", right_on="an", how="left"); H["streak"] = H.streak.fillna(-1).astype(int)
    H = H[H.R.between(RLO, RHI) & H.F.between(FLO, FHI)]
    ids = paper_ids.load(root, args.suffix)
    n_day = D[D.streak >= 0].groupby("streak").size(); n_night = S.n
    share = (n_day / (n_day + n_night.reindex(n_day.index).fillna(0))).dropna()
    big_day = [s for s in n_day.index if n_day[s] >= MIN_N and (("form" not in S.columns) or S.loc[s, "form"] != "chirp")]
    # labels: the paper number only for the 25 night TYPES (>= MIN_N night singer-nights); a night streak below that cut
    # that is big by day is a day singer the night catalog does not type: D1, D2 .. by carrier (MBE, 2026-10-13: "where do
    # types 26+ come from? we had 25 for the night")
    night_type = {s for s in big_day if s in ids and n_night.get(s, 0) >= MIN_N}
    unnumbered = sorted([s for s in big_day if s not in night_type], key=lambda s: S.loc[s, "F25"])
    lab = {s: str(ids[s]) for s in night_type}; lab.update({s: f"D{i + 1}" for i, s in enumerate(unnumbered)})
    print(f"{len(H):,} day singer-hours from {H.tid.nunique():,} tracks; {len(D):,} day singer-periods, {(D.streak >= 0).sum():,} on a line; "
          f"{len(big_day)} types with >= {MIN_N} day singer-periods ({len(unnumbered)} without a night number)")
    for s in sorted(big_day, key=lambda s: S.loc[s, "F25"]):
        print(f"   {lab[s]:>3s}  {S.loc[s, 'F25']:.0f} Hz @ {S.loc[s, 'R25']:.1f}  day {int(n_day[s])}, night {int(n_night[s])}, day share {share[s]:.2f}")

    page = PaperPage(number=args.number, title="The day singers")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    XL, XR = (1.35, 4.20), (4.70, 7.25); TOP1, H1 = 1.15, 2.85; TOP2, H2 = 4.75, 2.3
    rng = np.random.default_rng(0); Hp = H.iloc[rng.permutation(len(H))]
    # (A)
    axA = fig.add_axes(rect(*XL, TOP1, H1))
    sc = axA.scatter(Hp.R, Hp.F, s=1.5, c=Hp["T"], cmap="turbo", norm=Normalize(TLO, THI), alpha=0.6, edgecolors="none", rasterized=True)
    style(axA); cax = axA.inset_axes([0.05, 0.92, 0.40, 0.03]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("air temperature (°C)", fontsize=BODY_PT); cb.ax.tick_params(labelsize=BODY_PT)
    page.label(axA, "A", "Day singer-hours by temperature", y=1 + 0.05 / H1)
    # (B)
    axB = fig.add_axes(rect(*XR, TOP1, H1))
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20)); order = {s: i for i, s in enumerate(np.random.default_rng(1).permutation(big_day))}
    other = Hp[~Hp.streak.isin(big_day)]; axB.scatter(other.R, other.F, s=1.2, color="0.82", alpha=0.5, edgecolors="none", rasterized=True)
    for s in big_day:
        g = Hp[Hp.streak == s]
        if len(g):
            axB.scatter(g.R, g.F, s=1.5, color=cols[order[s] % 20], alpha=0.6, edgecolors="none", rasterized=True)
            axB.annotate(lab[s], (np.exp(np.log(g.R).mean()), np.exp(np.log(g.F).mean())), fontsize=5, ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cols[order[s] % 20], lw=0.6, alpha=0.9))
    style(axB); axB.set_ylabel("")
    page.label(axB, "B", "Day singer-hours by type", y=1 + 0.05 / H1)
    # (C) singers tracked at once by hour of the day, stacked by type
    axC = fig.add_axes(rect(*XL, TOP2, H2))
    rec = recorded_day_hours_by_hour(args.roots, first, last, os.path.join(root, "recorded_day_hours_by_hour.csv"))
    hrs = list(range(7, 19))
    byh = H.assign(h=H.hour.dt.hour).groupby(["streak", "h"]).size().unstack("h").reindex(columns=hrs).fillna(0)
    top = [s for s in sorted(big_day, key=lambda s: -n_day[s])][:10]
    stack = [byh.loc[s].to_numpy() / rec.reindex(hrs).to_numpy() for s in top] + [byh.loc[~byh.index.isin(top)].sum(axis=0).to_numpy() / rec.reindex(hrs).to_numpy()]
    axC.stackplot(np.array(hrs) + 0.5, *stack, colors=[cols[order[s] % 20] for s in top] + ["0.8"], labels=[f"{lab[s]}  {S.loc[s, 'F25'] / 1000:.1f} kHz @ {S.loc[s, 'R25']:.0f} Hz" for s in top] + ["other"], linewidth=0)
    axC.set_xlim(7, 19); axC.set_xticks(range(7, 20, 2)); axC.set_xlabel("hour of the day"); axC.set_ylabel("singers tracked at once")
    axC.legend(fontsize=5, frameon=False, loc="upper left", ncol=1, handlelength=1.0)
    page.label(axC, "C", "Singers tracked at once by hour of the day", y=1 + 0.05 / H2)
    # (D) day share against carrier
    axD = fig.add_axes(rect(*XR, TOP2, H2))
    both = [s for s in share.index if (n_day[s] + n_night.get(s, 0)) >= MIN_N and (("form" not in S.columns) or S.loc[s, "form"] != "chirp")]
    for s in both:
        c = cols[order[s] % 20] if s in order else "0.4"
        axD.scatter(S.loc[s, "F25"], share[s], s=14 + 0.02 * (n_day[s] + n_night.get(s, 0)), color=c, edgecolors="k", lw=0.4, zorder=3)
        if s in lab:
            axD.annotate(lab[s], (S.loc[s, "F25"], share[s]), fontsize=5, xytext=(3, 2), textcoords="offset points")
    axD.set_xscale("log"); axD.set_xlim(1800, 12000); axD.set_ylim(-0.02, 1.02)
    axD.xaxis.set_major_locator(FixedLocator([2000, 3000, 5000, 8000, 10000])); axD.xaxis.set_minor_locator(FixedLocator([])); axD.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    axD.set_xlabel("type carrier at 25 °C (kHz)"); axD.set_ylabel("share of singer-periods in the day"); axD.grid(alpha=0.25)
    page.label(axD, "D", "Day share of each type", y=1 + 0.05 / H2)
    page.finish(fig, args.out)
    pd.DataFrame({"streak": big_day, "label": [lab[s] for s in big_day], "F25": S.loc[big_day, "F25"].values, "R25": S.loc[big_day, "R25"].values,
                  "n_day": n_day[big_day].values, "n_night": n_night.reindex(big_day).fillna(0).values, "day_share": share.reindex(big_day).values}).to_csv(args.out + "_types.csv", index=False)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
