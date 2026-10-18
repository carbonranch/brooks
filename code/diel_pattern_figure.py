"""Figure 15 (provisional; was 16 until MBE reversed the two on 2026-10-14): the types across the 24 hours of the day (MBE, 2026-10-14: "a figure showing the temporal
pattern of types across 24 hours").

Singers tracked at once per recorded hour, by local clock hour, pooled over the year (Nov 2024 - Nov 2025): night
singer-hours from the night tracks (19:00-07:00), day singer-hours from the day tracks assigned to the night lines
(07:00-19:00); the denominator is the number of recorded hours at that clock hour over the year (from the per-window
detector files), so hours recorded by fewer recorders are not read as quieter.

  (A) singers tracked at once by hour of the day, stacked by type (the TOP most active types named, the rest gray);
      median sunset and sunrise of the singing season (April to October) dashed
  (B) each type's own 24-hour profile, scaled to its peak: the 25 trill types and the 6 chirp types (night half only: the
      chirp plane is built from the night periods) in order of their activity-weighted mean clock time (circular mean),
      afternoon first; columns = clock hours; the peak hour ringed, the mean time at the right

Usage:
  python analysis/callers/diel_pattern_figure.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --out analysis/callers/runs/figures_paper/F15_hour_of_day
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
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

TZ = "America/Chicago"
LAT, LON = 33.75, -93.49
TOP = 16
SEASON = (4, 10)          # months for the sunset/sunrise lines and for the "season" shading


def recorded_hours_by_hour(roots, first, last, period, cache):
    """Recorded windows per local clock hour over the year, from the per-window detector files of the <period> units."""
    if os.path.exists(cache):
        s = pd.read_csv(cache, index_col=0).iloc[:, 0]; s.index = s.index.astype(int); return s
    counts = pd.Series(0.0, index=range(24))
    for r in roots:
        for d in sorted(glob.glob(os.path.join(r, f"BARLT_*_{period}"))):
            m = re.search(r"_(\d{8})_" + period + "$", d)
            if not m or not (first <= pd.Timestamp(m.group(1)) <= last):
                continue
            for fn in os.listdir(d):
                mm = re.match(r"BARLT_\d+_[A-Za-z0-9]+_\d{8}_(\d{2})\d{2}_" + period + r"\.csv$", fn)
                if mm:
                    counts[int(mm.group(1))] += 300 / 3600
    counts.to_csv(cache)
    return counts


def sun_times(lat, lon, dates):
    """NOAA-style sunrise/sunset (local clock, hours) for a set of dates; good to a few minutes."""
    out = []
    for d in dates:
        doy = d.dayofyear; gamma = 2 * np.pi / 365 * (doy - 1 + (12 - 12) / 24)
        eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
        decl = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
        ha = np.degrees(np.arccos(np.cos(np.radians(90.833)) / (np.cos(np.radians(lat)) * np.cos(decl)) - np.tan(np.radians(lat)) * np.tan(decl)))
        rise_utc = 720 - 4 * (lon + ha) - eqtime; set_utc = 720 - 4 * (lon - ha) - eqtime
        off = pd.Timestamp(d, tz=TZ).utcoffset().total_seconds() / 60
        out.append(((rise_utc + off) / 60 % 24, (set_utc + off) / 60 % 24))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--first", default="2024-11-01"); ap.add_argument("--last", default="2025-11-30"); ap.add_argument("--number", default="15"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root; first, last = pd.Timestamp(args.first), pd.Timestamp(args.last)
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0); S = S[(S.n >= 40) & (S.form == "trill")]
    ids = paper_ids.load(root, args.suffix); trill = sorted(S.index, key=lambda s: ids[s])
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["hour", "an"]).merge(A[["an", "streak"]], on="an")
    P["h"] = pd.to_datetime(P.hour, utc=True).dt.tz_convert(TZ).dt.hour
    D = pd.read_csv(os.path.join(root, f"animal_days_streaks{args.suffix}.csv"))
    Hd = pd.read_csv(os.path.join(os.path.dirname(root.rstrip("/")), "year_figs", "nov24_nov25_day_hours.csv"), usecols=["hour", "tid"]).merge(D[["an", "streak"]], left_on="tid", right_on="an")
    Hd["h"] = pd.to_datetime(Hd.hour, utc=True).dt.tz_convert(TZ).dt.hour
    Sc = pd.read_csv(os.path.join(root, "chirp", "streaks_coarse_c2.csv"), index_col=0); Sc = Sc[Sc.n >= 40]
    Ac = pd.read_csv(os.path.join(root, "chirp", "animal_nights_streaks_c2.csv")); cid = pd.read_csv(os.path.join(root, "chirp", "paper_ids_chirp.csv")).set_index("streak").paper_id
    Pc = pd.read_csv(os.path.join(root, "chirp", "streak_hourly_points_all.csv"), usecols=["hour", "an"]).merge(Ac[["an", "streak"]], on="an"); Pc = Pc[Pc.streak.isin(Sc.index)]
    Pc["h"] = pd.to_datetime(Pc.hour, utc=True).dt.tz_convert(TZ).dt.hour
    rec_n = recorded_hours_by_hour(args.roots, first, last, "night", os.path.join(root, "recorded_night_hours_by_hour.csv"))
    rec_d = recorded_hours_by_hour(args.roots, first, last, "day", os.path.join(root, "recorded_day_hours_by_hour.csv"))
    rec = pd.Series(0.0, index=range(24))
    for h in range(24):
        rec[h] = rec_n.get(h, 0.0) + rec_d.get(h, 0.0)
    # singer-hours by type and clock hour: night + day for the trill types
    hours = range(24)
    Mt = pd.DataFrame(0.0, index=trill, columns=hours)
    for s, h in P[P.streak.isin(trill)].groupby(["streak", "h"]).size().items():
        Mt.loc[s[0], s[1]] += h
    for s, h in Hd[Hd.streak.isin(trill)].groupby(["streak", "h"]).size().items():
        Mt.loc[s[0], s[1]] += h
    chirp = sorted(Sc.index, key=lambda s: int(str(cid[s])[1:]))
    Mc = pd.DataFrame(0.0, index=chirp, columns=hours)
    for s, h in Pc.groupby(["streak", "h"]).size().items():
        Mc.loc[s[0], s[1]] += h
    per_t = Mt.div(rec.replace(0, np.nan), axis=1).fillna(0); per_c = Mc.div(rec.replace(0, np.nan), axis=1).fillna(0)
    labels = {s: str(ids[s]) for s in trill}; labels.update({s: str(cid[s]) for s in chirp})
    pos = {s: f"{S.loc[s, 'F25'] / 1000:.1f} kHz @ {S.loc[s, 'R25']:.0f} Hz" for s in trill}; pos.update({s: f"{Sc.loc[s, 'F25'] / 1000:.1f} kHz @ chirps {Sc.loc[s, 'R25']:.1f} Hz" for s in chirp})
    allp = pd.concat([per_t, per_c])
    # sunset / sunrise: medians over the season's days
    days = pd.date_range(first, last, freq="D"); days = days[(days.month >= SEASON[0]) & (days.month <= SEASON[1])]
    st = sun_times(LAT, LON, days); rise, sset = np.median(st[:, 0]), np.median(st[:, 1])
    print(f"recorded hours by clock hour: night {rec_n.sum():,.0f}, day {rec_d.sum():,.0f}; season sunrise {rise:.2f}, sunset {sset:.2f} (local, median Apr-Oct)")
    tot = allp.sum(axis=1).sort_values(ascending=False); top = list(tot.index[:TOP])
    for s in trill + chirp:
        prof = allp.loc[s]; print(f"  {labels[s]:>3s} {pos[s]:>26s}  peak {int(prof.idxmax()):02d}:00  day share {prof[7:19].sum() / max(prof.sum(), 1e-9):.2f}")

    page = PaperPage(number=args.number, title="The types across the 24 hours of the day")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top_, h):
        return [(x0 - fx0) / Wc, 1 - (top_ - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    x = np.arange(24) + 0.5
    # (A) stacked
    axA = fig.add_axes(rect(1.45, 7.05, 1.15, 2.4))
    cmap = plt.get_cmap("tab20"); col = {s: cmap(i % 20) for i, s in enumerate(top)}
    stack = [allp.loc[s].to_numpy() for s in top] + [allp.loc[~allp.index.isin(top)].sum(axis=0).to_numpy()]
    axA.stackplot(x, *stack, colors=[col[s] for s in top] + [(0.82, 0.82, 0.82, 1)], linewidth=0,
                  labels=[f"{labels[s]}  {pos[s]}" for s in top] + [f"{len(allp) - len(top)} other types"])
    for v in (sset, rise):
        axA.axvline(v, color="k", lw=0.7, ls="--")
    axA.set_xlim(0, 24); axA.set_xticks(range(0, 25, 3)); axA.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)])
    axA.set_ylabel("singers tracked at once\n(singer-hours per recorded hour)"); axA.set_xlabel("hour of the day (local)")
    axA.legend(fontsize=5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.54, 1.0), handlelength=1.0, columnspacing=1.0, labelspacing=0.3)   # over the quiet midday hours
    ymax = axA.get_ylim()[1]
    axA.text(sset + 0.15, ymax * 0.5, "sunset", fontsize=BODY_PT, ha="left", va="center", rotation=90); axA.text(rise - 0.15, ymax * 0.5, "sunrise", fontsize=BODY_PT, ha="right", va="center", rotation=90)
    page.label(axA, "A", "Singers tracked at once by hour of the day", y=1 + 0.05 / 2.4)
    # (B) heatmap of row-normalized profiles, rows ordered by each type's activity-weighted mean clock time (a circular mean, so
    # a night straddling midnight averages to about 01:00), from the afternoon through the night to the morning (MBE, 2026-10-14)
    ang = 2 * np.pi * (np.arange(24) + 0.5) / 24
    def mean_hour(prof):
        w = prof.to_numpy(); c, s = (w * np.cos(ang)).sum(), (w * np.sin(ang)).sum()
        return (np.degrees(np.arctan2(s, c)) / 15) % 24
    mh = {s: mean_hour(allp.loc[s]) for s in trill + chirp}
    rows = sorted(trill + chirp, key=lambda s: (mh[s] - 12) % 24); n = len(rows)
    for s in rows:
        print(f"  mean time {labels[s]:>3s}: {int(mh[s]):02d}:{int(60 * (mh[s] % 1)):02d}")
    Hm = np.array([allp.loc[s].to_numpy() / max(allp.loc[s].max(), 1e-9) for s in rows])
    B_TOP, B_H = 4.15, 0.138 * n            # 31 rows end near 8.4 in, leaving room for the x labels and the colorbar above the legend line
    axB = fig.add_axes(rect(2.15, 7.05, B_TOP, B_H))
    im = axB.imshow(Hm, aspect="auto", cmap="gray", vmin=0, vmax=1, extent=[0, 24, n, 0], interpolation="nearest")   # grayscale, black = quiet, white = the type's peak hour (MBE, 2026-10-14)
    for i, s in enumerate(rows):
        pk = int(allp.loc[s].idxmax()); axB.scatter([pk + 0.5], [i + 0.5], s=7, facecolors="none", edgecolors="tab:red", lw=0.6)
    for v in (sset, rise):
        axB.axvline(v, color="tab:red", lw=0.6, ls="--")
    axB.set_yticks(np.arange(n) + 0.5); axB.set_yticklabels([f"{labels[s]}  {pos[s]}" for s in rows], fontsize=5); axB.tick_params(axis="y", length=0)
    for i, s in enumerate(rows):                      # the weighted mean time at the right of each row
        axB.text(24.15, i + 0.5, f"{int(mh[s]):02d}:{int(60 * (mh[s] % 1)):02d}", fontsize=4.5, va="center", ha="left", color="0.3")
    axB.text(24.15, -0.4, "mean time", fontsize=4.5, va="bottom", ha="left", color="0.3")
    axB.set_xlim(0, 24); axB.set_xticks(range(0, 25, 3)); axB.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)]); axB.set_xlabel("hour of the day (local)")
    cax = fig.add_axes(rect(5.55, 7.05, B_TOP + B_H + 0.46, 0.07))   # horizontal, under the heatmap's right end (no room to the right of the canvas)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("share of the type's peak hourly activity", fontsize=BODY_PT); cb.ax.tick_params(labelsize=BODY_PT)
    page.label(axB, "B", "Each type's day, scaled to its peak hour, in order of mean time", x=-(2.15 - 1.45) / (7.05 - 2.15), y=1 + 0.05 / B_H)
    page.finish(fig, args.out)
    allp.assign(label=[labels[s] for s in allp.index]).to_csv(args.out + "_by_hour.csv")
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
