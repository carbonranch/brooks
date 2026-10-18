"""The chirpers over the year: singer-hours whose windows carried a chirp rate.

v2 reports a chirper with R = syllable rate inside the chirp and chirp_rate = the
chirp repetition rate (chirp_channel.py); year_hours_figure carries both per hour
as Rc (median) and rc_frac (share of the hour's windows with one). A chirper hour
is one with rc_frac >= CHIRP_MIN.

  A. table per streak           year_chirpers<suffix>.csv
     chirper hours and share, chirp rate (median, IQR), syllable periods per
     chirp period (R / Rc), chirp-rate slope on HRRR temperature (%/degC) and
     the syllable-rate slope on the same hours, months with chirping
  B. the chirper plane          year_chirpers<suffix>.png, left
     chirp rate against syllable rate for every chirper hour, colored by
     temperature, the chirper streaks labeled: chirp types as clusters
  C. chirp rate vs temperature  right: per chirper streak (>= MIN_HOURS), with
     the fitted slope; Dolbear's snowy tree cricket line for reference
  D. chirp share by week        year_chirpers_weeks<suffix>.png
     for each chirper streak, weekly hours split into chirping and trilling:
     does the chirper fraction change with season (two species on one line)?

Usage:
    python analysis/callers/year_chirpers.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TZ = "America/Chicago"
CHIRP_MIN = 0.5      # share of the hour's windows with a chirp rate
MIN_HOURS = 100      # chirper hours for a streak to get its own panel / slope
SHARE_MIN = 0.10     # chirper share for a streak to count as a chirper streak


def dolbear(T_C):
    """Dolbear 1897: chirps per minute N = 40 + 4 (T_F - 50), snowy tree cricket; as Hz."""
    return (40 + 4 * (T_C * 9 / 5 + 32 - 50)) / 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--suffix", default="_v6")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[S.harmonic_of < 0] if "harmonic_of" in S.columns else S
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv")).drop(columns=["streak"], errors="ignore").merge(A[["an", "streak"]], on="an")
    P = P[P.streak.isin(S.index)].copy()
    P["hour"] = pd.to_datetime(P.hour, utc=True).dt.tz_convert(TZ)
    P["chirp"] = P.rc_frac >= CHIRP_MIN
    C = P[P.chirp & P.Rc.notna() & (P.Rc > 0)].copy()
    C["per_chirp"] = C.R / C.Rc
    print(f"{len(P):,} singer-hours; {len(C):,} chirper hours ({len(C) / len(P):.1%}); chirp rate median {C.Rc.median():.2f} Hz")

    # A. per streak
    rows = []
    for sid, g in P.groupby("streak"):
        c = g[g.chirp & g.Rc.notna()]
        if len(c) == 0:
            continue
        slope = syl_slope = np.nan
        if len(c) >= MIN_HOURS and c["T"].std() > 2:
            slope = 100 * np.polyfit(c["T"], np.log(c.Rc), 1)[0]
            syl_slope = 100 * np.polyfit(c["T"], np.log(c.R), 1)[0]
        months = ",".join(str(m) for m in sorted(c.hour.dt.month.unique()))
        rows.append(dict(streak=sid, F25=S.loc[sid, "F25"], R25=S.loc[sid, "R25"], hours=len(g), chirp_hours=len(c), chirp_share=len(c) / len(g),
                         Rc_med=c.Rc.median(), Rc_q25=c.Rc.quantile(.25), Rc_q75=c.Rc.quantile(.75), per_chirp_med=(c.R / c.Rc).median(),
                         Rc_slope_pct_per_C=slope, R_slope_pct_per_C=syl_slope, T_med=c["T"].median(), chirp_months=months))
    D = pd.DataFrame(rows).sort_values("chirp_hours", ascending=False)
    D.to_csv(os.path.join(root, f"year_chirpers{args.suffix}.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 240); pd.set_option("display.max_rows", 80)
    print(D[D.chirp_hours >= 20].round(2).to_string(index=False))
    chirpers = list(D[(D.chirp_share >= SHARE_MIN) & (D.chirp_hours >= MIN_HOURS)].streak)

    # B + C
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    sc = ax1.scatter(C.R, C.Rc, c=C["T"], cmap="jet", vmin=5, vmax=35, s=4, alpha=0.5, edgecolors="none")
    for sid in chirpers:
        g = C[C.streak == sid]
        ax1.annotate(f"{sid}\n{S.loc[sid, 'F25'] / 1000:.2f} kHz", (g.R.median(), g.Rc.median()), fontsize=7, ha="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.5", alpha=0.85))
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("syllable rate inside the chirp, R (Hz)"); ax1.set_ylabel("chirp rate, Rc (Hz)")
    ax1.set_title(f"The chirper plane: {len(C):,} chirper hours (>= {CHIRP_MIN:.0%} of windows with a chirp rate),\ncolored by temperature; "
                  f"labels = streaks with >= {SHARE_MIN:.0%} chirper share", fontsize=8.5, loc="left")
    for k in (5, 10, 20):
        xs = np.array([8, 120]); ax1.plot(xs, xs / k, color="0.7", lw=0.6, ls=":"); ax1.text(120, 120 / k, f"R/Rc = {k}", fontsize=6.5, color="0.4", va="bottom", ha="right")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.02); cb.set_label("HRRR air temperature (°C)", fontsize=8)
    ax1.grid(alpha=0.3, which="both", lw=0.4)
    cmap = plt.get_cmap("tab10")
    for i, sid in enumerate(chirpers):
        g = C[C.streak == sid]
        ax2.scatter(g["T"], g.Rc, s=5, alpha=0.35, color=cmap(i % 10), edgecolors="none")
        r = D.set_index("streak").loc[sid]
        if np.isfinite(r.Rc_slope_pct_per_C):
            b, a = np.polyfit(g["T"], np.log(g.Rc), 1)
            xs = np.linspace(g["T"].quantile(.02), g["T"].quantile(.98), 20)
            ax2.plot(xs, np.exp(a + b * xs), color=cmap(i % 10), lw=1.6,
                     label=f"{sid}: {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz, {len(g)} h, {100 * b:.1f} %/°C")
    xs = np.linspace(10, 32, 20)
    ax2.plot(xs, dolbear(xs), color="k", lw=1, ls="--", label="Dolbear 1897, snowy tree cricket (4.9 %/°C at 25 °C)")
    ax2.set_yscale("log"); ax2.set_xlabel("HRRR air temperature at the hour (°C)"); ax2.set_ylabel("chirp rate (Hz)")
    ax2.set_title("Chirp rate against temperature per chirper streak,\nwith the fitted slope", fontsize=8.5, loc="left")
    ax2.legend(fontsize=6.8, loc="upper left", frameon=False); ax2.grid(alpha=0.3, which="both", lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"year_chirpers{args.suffix}.png"), dpi=140); fig.savefig(os.path.join(root, f"year_chirpers{args.suffix}.pdf")); plt.close(fig)

    # D. chirp share by week per chirper streak
    P["week"] = P.hour.dt.tz_localize(None).dt.to_period("W-SAT")
    weeks = pd.period_range(P.week.min().start_time, P.week.max().start_time, freq="W-SAT")
    ncol = 2; nrow = int(np.ceil(len(chirpers) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2 * ncol, 2.6 * nrow), sharex=True, squeeze=False)
    x = np.array([w.start_time for w in weeks])
    for ax, sid in zip(axes.ravel(), chirpers):
        g = P[P.streak == sid]
        h_all = g.groupby("week").size().reindex(weeks).fillna(0)
        h_ch = g[g.chirp].groupby("week").size().reindex(weeks).fillna(0)
        ax.bar(x, h_all - h_ch, width=7, color="0.75", label="trilling hours", align="edge")
        ax.bar(x, h_ch, width=7, bottom=h_all - h_ch, color="tab:red", label="chirping hours", align="edge")
        ax2b = ax.twinx()
        share = (h_ch / h_all.replace(0, np.nan))
        ax2b.plot(x + pd.Timedelta(days=3.5), share, color="k", lw=0.9, marker=".", ms=3)
        ax2b.set_ylim(0, 1); ax2b.set_ylabel("chirp share", fontsize=7); ax2b.tick_params(labelsize=7)
        ax.set_title(f"{sid}  {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz: {int(h_ch.sum())} chirping of {int(h_all.sum())} hours", fontsize=8, loc="left")
        ax.set_ylabel("singer-hours / week", fontsize=7); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(chirpers):]:
        ax.set_axis_off()
    axes[0, 0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("Chirping and trilling hours by week within each chirper streak (bars), and the chirp share (black): a seasonal change in share is two species on one line",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(root, f"year_chirpers_weeks{args.suffix}.png"), dpi=140); fig.savefig(os.path.join(root, f"year_chirpers_weeks{args.suffix}.pdf")); plt.close(fig)
    print("wrote", os.path.join(root, f"year_chirpers{args.suffix}.png / _weeks.png / .csv"))


if __name__ == "__main__":
    main()
