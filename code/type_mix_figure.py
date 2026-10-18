"""Figure 9: the mix of call types over the year, and where each type is heard (house style, paper_page.PaperPage).

  (A) singer-hours per RECORDED HOUR, by week, stacked by streak (the TOP most active streaks
      named, the rest gray), with the weekly mean of the nightly minimum air temperature dashed.
      Singer-hours per recorded hour is the mean number of singers tracked at once.
  (B) each streak's peak period and its detection by recorder. The peak weeks of a streak are
      the weeks in which its pooled activity (singer-hours per recorded hour over all recorders)
      is at least PEAK_FRAC of its maximum -- a property of the type, independent of site. For
      every recorder that recorded at least MIN_NIGHTS nights inside those weeks, the cell is the
      fraction of those nights on which the streak was detected (a singer-period of the streak
      on that recorder-night). A recorder with no data in the streak's peak weeks is left blank.
      A strip at the left shows each streak's peak weeks on the year axis of (A).

The denominator is recorded hours, counted from the detector's per-window files (one per 300 s
window recorded) in the unit directories, so partial nights count as what they were.

MBE (2026-10-06): "For each streak we know when we detect its calls (site independent) and when
those detections peak. We can then ask for each recorder whether we have data from that peak
period and if so, if and how often we detect it. If we don't have data for that time period, we
leave it blank."

Usage:
  python analysis/callers/type_mix_figure.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
      --roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --out analysis/callers/runs/figures_paper/F08_type_mix
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import textwrap

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from type_mix import TZ  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

TOP = 14
PEAK_FRAC = 0.5
PEAK_SMOOTH = 3          # weeks, centered running mean before the threshold
MIN_PEAK_WEEKS = 3
MIN_COVER = 0.25         # a week counts only if it has this share of the median week's recorded hours
MIN_NIGHTS = 5
WINDOW_H = 300 / 3600


def recorded_windows(roots, first, last, cache) -> pd.DataFrame:
    """One row per recorder-night: recorder, site, night, windows recorded (detector per-window files)."""
    if os.path.exists(cache):
        U = pd.read_csv(cache, parse_dates=["night"])
    else:
        rows = []
        for r in roots:
            for d in sorted(glob.glob(os.path.join(r, "BARLT_*_night"))):
                m = re.match(r"BARLT_(\d+)_([A-Za-z0-9]+)_(\d{8})_night", os.path.basename(d))
                if not m:
                    continue
                n = sum(1 for fn in os.listdir(d) if fn.endswith("_night.csv") and fn.startswith("BARLT_"))
                rows.append(dict(recorder="BARLT_" + m.group(1), site=m.group(2), night=pd.Timestamp(m.group(3)), windows=n))
        U = pd.DataFrame(rows); U.to_csv(cache, index=False)
    U = U[(U.night >= first) & (U.night <= last)].copy()
    U["hours"] = U.windows * WINDOW_H
    U["week"] = U.night.dt.to_period("W-SAT")
    return U


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--first", default="2024-11-01"); ap.add_argument("--last", default="2025-11-30")
    ap.add_argument("--top", type=int, default=TOP); ap.add_argument("--min-n", type=int, default=40); ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="nights", choices=["nights", "hours"],
                    help="cell of (B): 'nights' = fraction of recorded nights in the peak weeks with the streak; 'hours' = singer-hours per recorded hour in the peak weeks")
    ap.add_argument("--b-top", action="store_true", help="(B) shows only the --top most active streaks named in (A)")
    ap.add_argument("--by-site", action="store_true", help="(B) columns are sites (their recorders' nights pooled) instead of recorders")
    ap.add_argument("--number", default="11", help="figure number for the legend line")
    ap.add_argument("--chirp-root", default=None, help="a chirp_year.py directory (<year>/chirp): its streaks (ids C1..) join the main streaks")
    ap.add_argument("--sites", nargs="+", default=None,
                    help="(B) becomes one stacked-by-type panel per named site (same measure, colors and y scale as (A)) instead of the peak-week matrix (MBE, 2026-10-13)")
    args = ap.parse_args()
    root = args.root; first, last = pd.Timestamp(args.first), pd.Timestamp(args.last)
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[(S.harmonic_of < 0) if "harmonic_of" in S.columns else True]
    S = S[S.n >= args.min_n]
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv")).drop(columns=["streak"], errors="ignore").merge(A[["an", "streak"]], on="an", how="inner")
    P["hour"] = pd.to_datetime(P.hour, utc=True).dt.tz_convert(TZ)
    P["week"] = P.hour.dt.tz_localize(None).dt.to_period("W-SAT")
    P["night"] = pd.to_datetime(P.night)
    P = P[P.streak.isin(S.index)]
    ids = paper_ids.load(root, args.suffix)
    if args.chirp_root:                                  # the chirp-plane streaks: ids offset by CH_OFF, paper numbers C1..
        CH_OFF = 100000
        Ac = pd.read_csv(os.path.join(args.chirp_root, "animal_nights_streaks_c2.csv"))
        Sc = pd.read_csv(os.path.join(args.chirp_root, "streaks_coarse_c2.csv"), index_col=0)
        Sc = Sc[(Sc.harmonic_of < 0) if "harmonic_of" in Sc.columns else True]; Sc = Sc[Sc.n >= args.min_n]
        Pc = pd.read_csv(os.path.join(args.chirp_root, "streak_hourly_points_all.csv")).drop(columns=["streak"], errors="ignore").merge(Ac[["an", "streak"]], on="an", how="inner")
        Pc["hour"] = pd.to_datetime(Pc.hour, utc=True).dt.tz_convert(TZ); Pc["week"] = Pc.hour.dt.tz_localize(None).dt.to_period("W-SAT"); Pc["night"] = pd.to_datetime(Pc.night)
        Pc = Pc[Pc.streak.isin(Sc.index)]
        cid = pd.read_csv(os.path.join(args.chirp_root, "paper_ids_chirp.csv")).set_index("streak").paper_id
        Ac["streak"] = Ac.streak + CH_OFF; Pc["streak"] = Pc.streak + CH_OFF; Sc.index = Sc.index + CH_OFF
        ids.update({int(k) + CH_OFF: v for k, v in cid.items()})
        S = pd.concat([S, Sc]); A = pd.concat([A, Ac]); P = pd.concat([P, Pc], ignore_index=True)
        print(f"  + {len(Sc)} chirp streaks, {len(Pc):,} chirp singer-hours from {args.chirp_root}")
    weeks = pd.period_range(first, last, freq="W-SAT")
    U = recorded_windows(args.roots, first, last, os.path.join(root, "recorded_windows.csv"))
    rec_h = U.groupby("week").hours.sum().reindex(weeks).fillna(0)
    H = P.groupby(["streak", "week"]).size().unstack("week").reindex(columns=weeks).fillna(0)
    per_h = H.div(rec_h.replace(0, np.nan), axis=1)          # singer-hours per recorded hour
    hours_by_streak = P.groupby("streak").size().sort_values(ascending=False)
    top = list(hours_by_streak.index[:args.top])
    print(f"{len(P):,} singer-hours in {len(S)} streaks (>= {args.min_n} singer-periods); {len(U)} recorder-nights, {U.hours.sum():,.0f} recorded hours "
          f"(median {U.hours.median():.1f} h per recorder-night)")

    # (B) peak weeks per streak and detection per recorder inside them
    A["night"] = pd.to_datetime(A.night)
    An = A[A.streak.isin(S.index)]
    present = An.groupby(["streak", "site", "recorder", "night"]).size().reset_index()[["streak", "site", "recorder", "night"]]
    rec_order = (U.groupby(["site", "recorder"]).size().reset_index().sort_values(["site", "recorder"]))
    key = "site" if args.by_site else "recorder"          # the column unit of (B)
    if args.by_site:
        recs = list(dict.fromkeys(rec_order.site)); sites = recs
    else:
        sites = rec_order.site.tolist(); recs = rec_order.recorder.tolist()
    rows = []
    peak_weeks = {}
    covered = rec_h >= MIN_COVER * rec_h[rec_h > 0].median()
    for sid in S.index:
        h = per_h.loc[sid].fillna(0).where(covered, 0.0)
        hs = h.rolling(PEAK_SMOOTH, center=True, min_periods=1).mean()
        pk = set()
        for frac in (PEAK_FRAC, 0.35, 0.25):
            pk = set(hs.index[hs >= frac * hs.max()]) if hs.max() > 0 else set()
            if len(pk) >= MIN_PEAK_WEEKS:
                break
        if 0 < len(pk) < MIN_PEAK_WEEKS:
            pk = set(hs.sort_values(ascending=False).index[:MIN_PEAK_WEEKS])
        peak_weeks[sid] = pk
        Up = U[U.week.isin(pk)]
        n_rec = Up.groupby(key).size(); h_rec = Up.groupby(key).hours.sum()        # recorder-nights and recorded hours per column unit
        det = present[(present.streak == sid)].merge(Up[["recorder", "night"]], on=["recorder", "night"])
        n_det = det.groupby(key).size()
        sh = P[(P.streak == sid)].merge(Up[["recorder", "night"]], on=["recorder", "night"]).groupby(key).size()
        for r in recs:
            n = int(n_rec.get(r, 0)); ok = n >= MIN_NIGHTS
            rows.append(dict(streak=sid, unit=r, nights=n, detected=int(n_det.get(r, 0)), frac=(n_det.get(r, 0) / n) if ok else np.nan,
                             singer_hours=int(sh.get(r, 0)), recorded_hours=float(h_rec.get(r, 0.0)), hours_per_h=(sh.get(r, 0) / h_rec.get(r, np.nan)) if ok else np.nan))
    D = pd.DataFrame(rows)
    M = D.pivot(index="streak", columns="unit", values="frac" if args.metric == "nights" else "hours_per_h").reindex(index=S.index, columns=recs)
    # rows by peak week: the activity-weighted mean week of the peak weeks (MBE, 2026-10-06), earliest at the top
    def peak_center(sid):
        pk = peak_weeks[sid]
        if not pk:
            return np.inf
        w = np.array([wk_idx0[x] for x in pk], float); a = np.array([per_h.loc[sid].fillna(0)[x] for x in pk], float)
        return float((w * a).sum() / a.sum()) if a.sum() > 0 else w.mean()
    wk_idx0 = {w: i for i, w in enumerate(weeks)}
    order = sorted(S.index, key=peak_center)
    if args.b_top:
        order = [sid for sid in order if sid in top]
    M = M.loc[order]
    D["paper_id"] = D.streak.map(ids)
    D.to_csv(args.out + "_peak_detection.csv", index=False, float_format="%.3f")
    pk_tab = pd.DataFrame({sid: dict(n_peak_weeks=len(peak_weeks[sid]), first=min(peak_weeks[sid]).start_time.date() if peak_weeks[sid] else None,
                                     last=max(peak_weeks[sid]).end_time.date() if peak_weeks[sid] else None) for sid in order}).T
    print(pk_tab.to_string())
    print(f"cells with data: {M.notna().sum().sum()} of {M.size}; blank (fewer than {MIN_NIGHTS} recorded nights in the peak weeks): {M.isna().sum().sum()}")

    cmap = plt.get_cmap("tab20")
    colors = {s: cmap(i % 20) for i, s in enumerate(top)}; colors["other"] = (0.8, 0.8, 0.8, 1.0)
    C = pd.read_csv(os.path.join(root, f"type_catalog{args.suffix}.csv")).set_index("type") if os.path.exists(os.path.join(root, f"type_catalog{args.suffix}.csv")) else None

    def name_of(sid):
        if C is not None and sid in C.index and isinstance(C.candidate.get(sid), str) and C.candidate_status.get(sid) in ("recording", "position"):
            g = C.candidate[sid].split(); return f"{g[0][0]}. {g[-1]}"
        return ""
    def rate_txt(s_):
        return f"chirps {S.loc[s_, 'R25']:.1f} Hz" if str(ids.get(s_, "")).startswith("C") else f"{S.loc[s_, 'R25']:.0f} Hz"
    labels = {s: f"{ids.get(s, s)}  {S.loc[s, 'F25'] / 1000:.1f} kHz @ {rate_txt(s)}" for s in top}      # no species names before the names figure (MBE, 2026-10-13)
    labels["other"] = f"{len(S) - len(top)} other streaks"

    nrow_b = len(order)
    page = PaperPage(number=args.number, title="The mix of call types through the year and across the ranch")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top_, h):
        return [(x0 - fx0) / Wc, 1 - (top_ - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    X0, X1 = 1.45, 7.05
    A_TOP, A_H = 1.15, (2.10 if args.sites else 2.60)
    row_h = min(0.26, 4.2 / max(nrow_b, 1)); B_TOP = A_TOP + A_H + 0.75; B_H = row_h * nrow_b
    SITE_H, SITE_GAP = 1.45, 0.10                                  # (B) with --sites: one panel per site
    STRIP_W = 1.05; XB = 2.55                                      # (B) starts further right: its row labels sit in the space to the left

    # (A) year
    ax = fig.add_axes(rect(X0, X1, A_TOP, A_H))
    x = np.array([w.start_time + pd.Timedelta(days=3.5) for w in weeks])
    stack = [per_h.loc[s].fillna(0).to_numpy() for s in top] + [per_h.loc[~per_h.index.isin(top)].sum(axis=0).fillna(0).to_numpy()]
    ax.stackplot(x, *stack, labels=[labels[s] for s in top] + [labels["other"]], colors=[colors[s] for s in top] + [colors["other"]], linewidth=0)
    ax.set_ylabel("singers tracked at once\n(singer-hours per recorded hour)"); ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, None)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=5, frameon=False, ncol=2, handlelength=1.0, columnspacing=1.0, labelspacing=0.3)
    wpath = os.path.join(root, "weather_hourly_hrrr.csv")
    if not os.path.exists(wpath):
        wpath = os.path.join(args.roots[-1], "weather_hourly_hrrr.csv")
    if os.path.exists(wpath):
        W = pd.read_csv(wpath); W["t"] = pd.to_datetime(W.t, utc=True).dt.tz_convert(TZ)
        W = W[(W.t.dt.hour >= 19) | (W.t.dt.hour < 7)]
        W["night"] = (W.t - pd.Timedelta(hours=12)).dt.tz_localize(None).dt.normalize()
        tn = W.groupby("night").T_C.min(); wx = tn.groupby(tn.index.to_period("W-SAT")).mean().reindex(weeks)
        axb = ax.twinx(); axb.plot(x, wx.to_numpy(), color="k", lw=0.9, ls="--"); axb.axhline(0, color="k", lw=0.4, alpha=0.4)
        axb.set_ylabel("weekly mean nightly minimum\nair temperature (°C, dashed)")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    for lab in ax.get_xticklabels():
        pass
    page.label(ax, "A", "Singers tracked at once, by week and type", y=1 + 0.05 / A_H)

    if args.sites:
        # (B) the same measure at each named site: its recorders' singer-hours over its recorded hours, by week, stacked by
        # type in (A)'s colors; weeks with under MIN_COVER of the site's usual recording hatched
        site_axes = []; ymax = 0.0; stacks = {}
        for site in args.sites:
            rec_s = U[U.site == site].groupby("week").hours.sum().reindex(weeks).fillna(0)
            Hs = P[P.site == site].groupby(["streak", "week"]).size().unstack("week").reindex(columns=weeks).fillna(0)
            per_s = Hs.div(rec_s.replace(0, np.nan), axis=1)
            st = [per_s.loc[s].fillna(0).to_numpy() if s in per_s.index else np.zeros(len(weeks)) for s in top] + [per_s.loc[~per_s.index.isin(top)].sum(axis=0).fillna(0).to_numpy()]
            thin = rec_s < MIN_COVER * rec_s[rec_s > 0].median()
            stacks[site] = (st, thin, rec_s, int(Hs.values.sum()), U[U.site == site].recorder.nunique())
            ymax = max(ymax, float(np.sum(st, axis=0).max()))
        for k, site in enumerate(args.sites):
            st, thin, rec_s, n_sh, n_rec = stacks[site]
            top_k = B_TOP + k * (SITE_H + SITE_GAP)
            axk = fig.add_axes(rect(X0, X1, top_k, SITE_H)); site_axes.append(axk)
            axk.stackplot(x, *st, colors=[colors[s] for s in top] + [colors["other"]], linewidth=0)
            for w, t in zip(weeks, thin):
                if t:
                    axk.axvspan(w.start_time, w.end_time, color="white", alpha=0.6, lw=0, hatch="////", zorder=5)
            axk.set_xlim(x[0], x[-1]); axk.set_ylim(0, ymax * 1.04)
            axk.text(0.008, 0.95, site, transform=axk.transAxes, fontsize=BODY_PT + 1, fontweight="bold", va="top", ha="left")
            axk.xaxis.set_major_locator(matplotlib.dates.MonthLocator()); axk.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
            if k < len(args.sites) - 1:
                plt.setp(axk.get_xticklabels(), visible=False)
            print(f"  (B) {site}: {n_rec} recorders, {rec_s.sum():,.0f} recorded hours, {n_sh:,} singer-hours, peak {np.sum(st, axis=0).max():.2f} singers at once")
        site_axes[len(site_axes) // 2].set_ylabel("singers tracked at once\n(singer-hours per recorded hour)")
        page.label(site_axes[0], "B", "The same, site by site", y=1 + 0.05 / SITE_H)
        page.finish(fig, args.out)
        print(f"wrote {args.out}.pdf / .png")
        return

    # (B) left: peak-week strips; right: detection matrix
    axs = fig.add_axes(rect(XB, XB + STRIP_W, B_TOP, B_H)); axm = fig.add_axes(rect(XB + STRIP_W + 0.06, X1, B_TOP, B_H), sharey=axs)
    nrow = len(order)
    wk_idx = {w: i for i, w in enumerate(weeks)}
    for i, sid in enumerate(order):
        h_ = per_h.loc[sid].fillna(0).to_numpy(); hm = h_.max() if h_.max() > 0 else 1
        axs.fill_between(np.arange(len(weeks)) + 0.5, i + 0.9, i + 0.9 - 0.8 * h_ / hm, color="0.6", lw=0, step="mid")
        for w in peak_weeks[sid]:
            axs.axvspan(wk_idx[w], wk_idx[w] + 1, ymin=(nrow - i - 1) / nrow, ymax=(nrow - i) / nrow, color="tab:orange", alpha=0.35, lw=0)
    axs.set_xlim(0, len(weeks)); axs.set_ylim(nrow, 0)
    ticks = [i for i, w in enumerate(weeks) if w.start_time.day <= 7 and w.start_time.month in (11, 2, 5, 8)]
    axs.set_xticks(ticks); axs.set_xticklabels([weeks[i].start_time.strftime("%b %y") for i in ticks], fontsize=5, rotation=90)
    axs.set_yticks(np.arange(nrow) + 0.5); axs.set_yticklabels([labels.get(s_, f"{ids.get(s_, s_)}  {S.loc[s_, 'F25'] / 1000:.1f} kHz @ {rate_txt(s_)}") for s_ in order], fontsize=5)
    for lab, sid in zip(axs.get_yticklabels(), order):
        if sid in colors:
            lab.set_color(colors[sid]); lab.set_fontweight("bold")
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    axs.tick_params(axis="y", length=0)
    axs.set_title("weekly activity (gray), peak weeks (orange)", fontsize=BODY_PT, loc="left", pad=3)

    Mv = M.to_numpy(dtype=float)
    cm = plt.get_cmap("Blues").copy()
    vmax = 1.0 if args.metric == "nights" else float(np.nanpercentile(Mv, 99))
    im = axm.imshow(Mv, aspect="auto", cmap=cm, vmin=0, vmax=vmax, extent=[0, len(recs), nrow, 0], interpolation="nearest")
    for i in range(nrow):
        for j in range(len(recs)):
            if np.isnan(Mv[i, j]):
                axm.add_patch(plt.Rectangle((j, i), 1, 1, fc="white", ec="none", lw=0))
                axm.add_patch(plt.Rectangle((j, i), 1, 1, fc="none", ec="0.8", hatch="////", lw=0))
    axm.set_xlim(0, len(recs))
    axm.set_xticks(np.arange(len(recs)) + 0.5)
    wrap = lambda name: re.sub(r"(?<=[a-z])(?=[A-Z])", "\n", name) if len(name) > 6 else name     # EastCreek -> East\nCreek, WestSideJunction -> West\nSide\nJunction
    axm.set_xticklabels([wrap(r) for r in recs] if args.by_site else [r[-4:] for r in recs], fontsize=5 if args.by_site else 5, rotation=90 if not args.by_site else 0)
    plt.setp(axm.get_yticklabels(), visible=False); axm.tick_params(axis="y", length=0)
    if not args.by_site:
        bounds = [0] + [i + 1 for i in range(len(sites) - 1) if sites[i] != sites[i + 1]] + [len(sites)]
        for b in bounds[1:-1]:
            axm.axvline(b, color="k", lw=0.6)
        for a, b in zip(bounds[:-1], bounds[1:]):
            axm.annotate(sites[a], xy=((a + b) / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -26), textcoords="offset points",
                         ha="center", va="top", fontsize=5, rotation=90 if b - a < 3 else 0)
    else:
        for j in range(1, len(recs)):
            axm.axvline(j, color="w", lw=0.5)
    for i in range(1, nrow):
        axm.axhline(i, color="w", lw=0.4)
    cax = fig.add_axes(rect(X1 + 0.06, X1 + 0.14, B_TOP, B_H))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(("fraction of recorded nights in the peak weeks with the streak") if args.metric == "nights" else
                 f"singers of the streak tracked at once in its peak weeks\n(singer-hours per recorded hour, scale clipped at {vmax:.2f})", fontsize=BODY_PT)
    cb.ax.tick_params(labelsize=BODY_PT)
    axm.set_xlabel("site (its recorders' nights pooled)" if args.by_site else "recorder, grouped by site", labelpad=4 if args.by_site else 40)
    axs.set_title("weekly activity (gray), peak weeks (orange)", fontsize=BODY_PT, loc="left", pad=3)
    page.label(axs, "B", "Where each type is heard", x=-(XB - X0) / STRIP_W, y=1 + 0.17 / B_H)
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png and {args.out}_peak_detection.csv")


if __name__ == "__main__":
    main()
