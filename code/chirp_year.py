"""The chirpers' year on a second plane: carrier against CHIRP rate.

The year chain places a singer by its syllable rate, so a chirper whose syllables the time-domain reader could not
read (v3: a chirp rate but no syllable rate) is linked into a track but never reaches the streaks. Those tracks
form lines on carrier against chirp rate as clean as the trillers' on the main plane (explore/chirp_only_tracks_v3,
2026-10-11), and chirp rate is the rhythm the modulation map actually measured for them. Here they get their own
streak chain on that plane:

  1. chirp singer-hours: for every night track of >= MIN_H hours with a chirp rate and no syllable rate, one row
     per hour: carrier (F_fix, else F_c), chirp rate (median), HRRR temperature; written in the format of the main
     chain (R := chirp rate) into <year_root>/chirp/
  2. build_streaks.py on that directory with chirp-plane parameters (temperature slopes fitted here from the
     within-track drift; coupling and tolerances from the command line), then merge_streaks.py
  3. paper numbers C1..Cn by carrier (paper_ids_chirp.csv)
  4. the figure: (A) chirp singer-hours colored by temperature, (B) colored by chirp streak and labeled, (C) chirp
     rate against temperature per streak with the fitted slope

Usage:
    python analysis/callers/chirp_year.py analysis/callers/runs/v3/year_nov24_nov25 --roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 \
        --out analysis/callers/runs/figures_paper/F07b_chirp_plane
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_page import PaperPage, BODY_PT  # noqa: E402

TZ = "America/Chicago"
MIN_H = 2.0
CHIRP_FRAC = 0.5                # a track is a chirper when this share of its kept windows carries a chirp rate
MIN_N = 40
RCLO, RCHI = 1.5, 8.5                 # the detector caps chirp rate at 8 Hz (BOUT_MAX_HZ)
FLO, FHI = 1500.0, 20000.0
TLO, THI = 12.0, 32.0
PY = sys.executable


def chirp_hours(roots, first, last):
    rows = []
    wx = []
    for r in roots:
        p = os.path.join(r, "weather_hourly_hrrr.csv")
        if os.path.exists(p):
            W = pd.read_csv(p); W["t"] = pd.to_datetime(W.t, utc=True); wx.append(W.dropna(subset=["T_C"]))
    Wx = pd.concat(wx).drop_duplicates("t").sort_values("t")
    wt, wT = Wx.t.astype("int64").to_numpy(), Wx.T_C.to_numpy()
    for r in roots:
        for u in sorted(glob.glob(os.path.join(r, "BARLT_*_night"))):
            m = re.match(r"BARLT_(\d+)_([A-Za-z0-9]+)_(\d{8})_night", os.path.basename(u))
            if not m:
                continue
            night = pd.Timestamp(m.group(3))
            if not (first <= night <= last):
                continue
            ft = os.path.join(u, "night_night_global_tracks.csv"); fa = os.path.join(u, "night_night_global_window_animals.csv")
            if not (os.path.exists(ft) and os.path.exists(fa)):
                continue
            T = pd.read_csv(ft, usecols=["track", "span_h", "R", "Rc"])
            co = T[(T.span_h >= MIN_H) & T.Rc.notna()]
            if co.empty:
                continue
            A = pd.read_csv(fa, usecols=["t", "track", "F_c", "F_fix", "R", "chirp_rate", "depth", "keep"])
            A = A[A.track.isin(co.track) & A.keep.astype(bool)].copy()
            # a chirper: a chirp rate in at least CHIRP_FRAC of the track's kept windows (with or without a syllable rate)
            frac = A.groupby("track").chirp_rate.apply(lambda v: v.notna().mean())
            A = A[A.track.isin(frac.index[frac >= CHIRP_FRAC]) & A.chirp_rate.notna()]
            if A.empty:
                continue
            A["t"] = pd.to_datetime(A.t, utc=True); A["F"] = A.F_fix.fillna(A.F_c)
            A["hour"] = A.t.dt.floor("h")
            g = A.groupby(["track", "hour"]).agg(F=("F", "median"), Rc=("chirp_rate", "median"), R=("R", "median"), depth=("depth", "median"), n=("t", "size")).reset_index()
            g["tid"] = os.path.basename(u) + ":" + g.track.astype(str)
            g["recorder"] = "BARLT_" + m.group(1); g["site"] = m.group(2); g["night"] = night
            rows.append(g)
    H = pd.concat(rows, ignore_index=True)
    H["T"] = np.interp(H.hour.astype("int64"), wt, wT)
    return H, Wx


def within_track_slopes(H):
    """Pooled within-track slopes of log chirp rate and log carrier on temperature (fraction per degC), from tracks
    spanning >= 3 degC: the night's cooling read on each animal, as temperature_effects.py does for the main chain."""
    dr, df_, dT = [], [], []
    for tid, g in H.groupby("tid"):
        if len(g) >= 4 and g["T"].max() - g["T"].min() >= 3:
            dr.append(np.log(g.Rc) - np.log(g.Rc).mean()); df_.append(np.log(g.F) - np.log(g.F).mean()); dT.append(g["T"] - g["T"].mean())
    dr, df_, dT = (np.concatenate(x) for x in (dr, df_, dT))
    bR = float(np.polyfit(dT, dr, 1)[0]); bF = float(np.polyfit(dT, df_, 1)[0])
    return bR, bF, len(dT)


def plane_axes(ax):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(RCLO, RCHI); ax.set_ylim(FLO, FHI)
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6, 8])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000])); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_xlabel("chirp rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(True, alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("year_root"); ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--first", default="2024-11-01"); ap.add_argument("--last", default="2025-11-30")
    ap.add_argument("--coupling", type=float, default=0.35, help="d log F / d log Rc of the chirp lines")
    ap.add_argument("--along-tol", type=float, default=1.0); ap.add_argument("--perp-tol", type=float, default=0.10)
    ap.add_argument("--min-n", type=int, default=MIN_N); ap.add_argument("--number", default="9")
    ap.add_argument("--skip-build", action="store_true", help="reuse the chirp directory's streaks")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="chirp streak ids dropped from the typed set after clustering, by QC verdict (MBE 2026-10-13: streak 13, 15.3 kHz, 51 of 63 singer-nights at one site, short faint tracks, "
                         "the grid 'a mess'); their singer-nights become untyped (-1) in animal_nights_streaks_c2.csv, the unfiltered tables are kept as *_c2_all.csv and the reason in excluded_streaks.csv")
    ap.add_argument("--min-syl-share", type=float, default=0.5,
                    help="type only tracks whose syllable rate inside the chirps was read in at least this share of their hours; the rest (frogs, birds, raspers, "
                         "and chirpers whose syllables cannot be read) are written to bout_only_tracks.csv and not typed (MBE 2026-10-13: the mixed types were too messy to be reliable)")
    args = ap.parse_args()
    croot = os.path.join(args.year_root, "chirp"); os.makedirs(croot, exist_ok=True)
    first, last = pd.Timestamp(args.first), pd.Timestamp(args.last)

    if not args.skip_build:
        H, Wx = chirp_hours(args.roots, first, last)
        share = H.groupby("tid").R.apply(lambda v: v.notna().mean())
        H["syl_share"] = H.tid.map(share)
        bout = H[H.syl_share < args.min_syl_share]
        B = bout.groupby("tid").agg(recorder=("recorder", "first"), site=("site", "first"), night=("night", "first"), F=("F", "median"), Rc=("Rc", "median"),
                                    T=("T", "mean"), depth=("depth", "median"), n_hours=("hour", "size"), syl_share=("syl_share", "first")).reset_index()
        B.to_csv(os.path.join(croot, "bout_only_tracks.csv"), index=False, float_format="%.4f")
        bout.drop(columns=["syl_share"]).to_csv(os.path.join(croot, "bout_only_hours.csv"), index=False, float_format="%.4f")
        print(f"{bout.tid.nunique():,} tracks ({len(bout):,} hours) with a syllable rate in under {args.min_syl_share:.0%} of their hours set aside as bout-only (not typed)")
        H = H[H.syl_share >= args.min_syl_share].drop(columns=["syl_share"])
        bR, bF, n_pairs = within_track_slopes(H)
        print(f"{len(H):,} chirp singer-hours from {H.tid.nunique():,} chirping tracks (>= {MIN_H:g} h, a chirp rate in >= {CHIRP_FRAC:.0%} of windows), {H.recorder.nunique()} recorders; "
              f"within-track slopes: chirp rate {100 * bR:+.2f} %/degC, carrier {100 * bF:+.2f} %/degC ({n_pairs:,} hour pairs)")
        # the main chain's input format, R := chirp rate
        A = H.groupby("tid").agg(recorder=("recorder", "first"), site=("site", "first"), night=("night", "first"), F=("F", "median"), R=("Rc", "median"),
                                 T=("T", "mean"), depth=("depth", "median"), n_hours=("hour", "size")).reset_index().rename(columns={"tid": "an"})
        A["period"] = "night"; A["Rc"] = A.R; A["rc_frac"] = 1.0; A["type_corrected"] = -1
        A.to_csv(os.path.join(croot, "temperature_animal_nights.csv"), index=False, float_format="%.4f")
        pd.DataFrame(columns=["type", "R_perC", "F_perC"]).to_csv(os.path.join(croot, "temperature_season_by_type.csv"), index=False)
        P = H.rename(columns={"tid": "an", "R": "Rs"}); P["R"] = P.Rc; P["rc_frac"] = 1.0; P["n"] = 1; P["streak"] = -1
        P["hour"] = P.hour.dt.tz_convert(TZ)
        P[["hour", "R", "F", "depth", "T", "Rc", "Rs", "rc_frac", "an", "site", "recorder", "night", "n", "streak"]].to_csv(os.path.join(croot, "streak_hourly_points_all.csv"), index=False, float_format="%.4f")
        print(f"   {H.R.notna().sum():,} of the chirp singer-hours also carry a syllable rate ({H[H.R.notna()].tid.nunique():,} tracks)")
        Wx.to_csv(os.path.join(croot, "weather_hourly_hrrr.csv"), index=False)
        pd.Series(dict(bR=bR, bF=bF, coupling=args.coupling, along_tol=args.along_tol, perp_tol=args.perp_tol)).to_csv(os.path.join(croot, "chirp_params.csv"))
        subprocess.run([PY, f"{HERE}/build_streaks.py", croot, "--bR", f"{bR:.5f}", "--bF", f"{bF:.5f}", "--coupling", str(args.coupling),
                        "--along-tol", str(args.along_tol), "--perp-tol", str(args.perp_tol), "--suffix", "_c1"], check=True)
        subprocess.run([PY, f"{HERE}/merge_streaks.py", croot, "--in", "_c1", "--out", "_c2", "--coupling", str(args.coupling), "--coupling-hf", str(args.coupling),
                        "--slope", f"{100 * bR:.2f}"], check=True)
    # QC exclusions: applied to the tables every downstream script reads, so the excluded streak is not a type anywhere
    pa, ps = os.path.join(croot, "animal_nights_streaks_c2.csv"), os.path.join(croot, "streaks_coarse_c2.csv")
    pa_all, ps_all = pa.replace("_c2.csv", "_c2_all.csv"), ps.replace("_c2.csv", "_c2_all.csv")
    if args.exclude:
        if not os.path.exists(pa_all):
            os.replace(pa, pa_all); os.replace(ps, ps_all)
        A_all = pd.read_csv(pa_all); S_all = pd.read_csv(ps_all, index_col=0)
        exc = [s for s in args.exclude if s in S_all.index]
        A_all.loc[A_all.streak.isin(exc), "streak"] = -1
        A_all.to_csv(pa, index=False, float_format="%.4f"); S_all.drop(index=exc).to_csv(ps)
        pd.DataFrame({"streak": exc, "n": S_all.loc[exc, "n"].values, "F25": S_all.loc[exc, "F25"].round().values, "R25": S_all.loc[exc, "R25"].round(2).values,
                      "reason": ["QC (MBE 2026-10-13): heterogeneous grid, 51/63 singer-nights at one site, short faint tracks" for _ in exc]}).to_csv(os.path.join(croot, "excluded_streaks.csv"), index=False)
        print(f"excluded from the typed set by QC: {exc} ({int(S_all.loc[exc, 'n'].sum())} singer-nights)")
    elif os.path.exists(pa_all):
        os.replace(pa_all, pa); os.replace(ps_all, ps)
    A = pd.read_csv(pa)
    S = pd.read_csv(ps, index_col=0)
    P = pd.read_csv(os.path.join(croot, "streak_hourly_points_all.csv")).drop(columns=["streak"]).merge(A[["an", "streak"]], on="an", how="left")
    P["streak"] = P.streak.fillna(-1).astype(int)
    big = list(S[S.n >= args.min_n].sort_values("F25").index)
    ids = {sid: f"C{i + 1}" for i, sid in enumerate(big)}
    small = list(S[S.n < args.min_n].sort_values("F25").index)
    ids.update({sid: f"C{len(big) + i + 1}" for i, sid in enumerate(small)})
    # per-streak chirp-rate slope on temperature over its hours
    slopes = {}
    for sid in big:
        g = P[P.streak == sid]
        b, a, *_ = theilslopes(np.log(g.Rc), g["T"])
        slopes[sid] = (float(a), float(b))
    tab = pd.DataFrame({"streak": list(ids), "paper_id": list(ids.values())}).merge(S.reset_index().rename(columns={"index": "streak"}), on="streak", how="left")
    tab["Rc_slope_pct_perC"] = [100 * slopes[s][1] if s in slopes else np.nan for s in tab.streak]
    tab.to_csv(os.path.join(croot, "paper_ids_chirp.csv"), index=False, float_format="%.3f")
    print(f"{len(S)} chirp streaks, {len(big)} with >= {args.min_n} singer-nights:")
    print(tab[tab.streak.isin(big)][["paper_id", "streak", "n", "F25", "R25", "n_sites", "n_nights", "Rc_slope_pct_perC"]].round(2).to_string(index=False))

    # ---- the figure (MBE, 2026-10-12): A F-C by temperature, B F-R by temperature, C F-C by type, D F-R by type, E R-C by type ----
    page = PaperPage(number=args.number, title="The chirpers' year on the carrier against chirp-rate plane")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    rng = np.random.default_rng(0); Pp = P.iloc[rng.permutation(len(P))]
    has = Pp[Pp.Rs.notna()] if "Rs" in Pp.columns else Pp.iloc[0:0]
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20)); order = {sid: i for i, sid in enumerate(np.random.default_rng(1).permutation(big))}
    tnorm = Normalize(TLO, THI)
    H_ROW = 2.25; TOPS = (1.15, 3.95, 6.75); XL, XR = (1.35, 4.20), (4.70, 7.25)

    def syl_axes(ax):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(15, 120); ax.set_ylim(FLO, FHI)
        ax.xaxis.set_major_locator(FixedLocator([20, 30, 40, 50, 70, 100])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000])); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        ax.set_xlabel("syllable rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(True, alpha=0.25)

    def by_type(ax, D, x, y, label=True):
        oth = D[~D.streak.isin(big)]; ax.scatter(oth[x], oth[y], s=1.2, color="0.82", alpha=0.5, edgecolors="none", rasterized=True)
        for sid in big:
            g = D[D.streak == sid]
            if len(g):
                ax.scatter(g[x], g[y], s=1.5, color=cols[order[sid] % 20], alpha=0.6, edgecolors="none", rasterized=True)
                if label:
                    ax.annotate(ids[sid], (np.exp(np.log(g[x]).mean()), np.exp(np.log(g[y]).mean())), fontsize=5, ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cols[order[sid] % 20], lw=0.6, alpha=0.9))

    # (A) carrier against chirp rate, by temperature
    axA = fig.add_axes(rect(*XL, TOPS[0], H_ROW))
    sc = axA.scatter(Pp.Rc, Pp.F, s=1.5, c=Pp["T"], cmap="turbo", norm=tnorm, alpha=0.6, edgecolors="none", rasterized=True)
    plane_axes(axA)
    cax = axA.inset_axes([0.05, 0.92, 0.40, 0.03]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("air temperature (°C)", fontsize=BODY_PT); cb.ax.tick_params(labelsize=BODY_PT)
    page.label(axA, "A", "Chirpers by temperature", y=1 + 0.05 / H_ROW)
    # (B) carrier against syllable rate, by temperature (the hours that carry one)
    axB = fig.add_axes(rect(*XR, TOPS[0], H_ROW))
    axB.scatter(has.Rs, has.F, s=1.5, c=has["T"], cmap="turbo", norm=tnorm, alpha=0.6, edgecolors="none", rasterized=True)
    syl_axes(axB); axB.set_ylabel("")
    page.label(axB, "B", "Hours with a syllable rate", y=1 + 0.05 / H_ROW)
    # (C) carrier against chirp rate, by chirp type
    axC = fig.add_axes(rect(*XL, TOPS[1], H_ROW)); by_type(axC, Pp, "Rc", "F"); plane_axes(axC)
    page.label(axC, "C", "Chirpers by type", y=1 + 0.05 / H_ROW)
    # (D) carrier against syllable rate, by chirp type
    axD = fig.add_axes(rect(*XR, TOPS[1], H_ROW)); by_type(axD, has, "Rs", "F"); syl_axes(axD); axD.set_ylabel("")
    page.label(axD, "D", "Hours with a syllable rate, by type", y=1 + 0.05 / H_ROW)
    # (E) syllable rate against chirp rate, by chirp type: syllable periods per chirp period
    axE = fig.add_axes(rect(*XL, TOPS[2], H_ROW)); by_type(axE, has, "Rc", "Rs", label=False)
    for k in (5, 10, 20, 40):
        axE.plot([RCLO, RCHI], [RCLO * k, RCHI * k], "k:", lw=0.5); xk = min(RCHI * 0.95, 110 / k); axE.text(xk, xk * k, f"{k}x ", fontsize=5, ha="right", va="bottom", clip_on=True)
    axE.set_xscale("log"); axE.set_yscale("log"); axE.set_xlim(RCLO, RCHI); axE.set_ylim(15, 120)
    axE.xaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6, 8])); axE.xaxis.set_minor_locator(FixedLocator([])); axE.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    axE.yaxis.set_major_locator(FixedLocator([20, 30, 40, 50, 70, 100])); axE.yaxis.set_minor_locator(FixedLocator([])); axE.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    axE.set_xlabel("chirp rate (Hz)"); axE.set_ylabel("syllable rate (Hz)"); axE.grid(True, alpha=0.25)
    page.label(axE, "E", "Syllable rate against chirp rate", y=1 + 0.05 / H_ROW)
    page.finish(fig, args.out)
    # a second page: chirp rate against temperature per streak with the fitted slope
    page2 = PaperPage(number=f"S{args.number}", title="Chirp rate against temperature per chirp streak")
    fig2 = plt.figure(figsize=(Wc, Hc))
    n = len(big); ncol = 4; nrow = int(np.ceil(n / ncol)); cw, ch = 1.30, 0.90; x0, top0 = 1.35, 1.35
    for i, sid in enumerate(big):
        r, c = divmod(i, ncol)
        ax = fig2.add_axes(rect(x0 + c * (cw + 0.22), x0 + c * (cw + 0.22) + cw, top0 + r * (ch + 0.50), ch))
        g = P[P.streak == sid]
        ax.scatter(g["T"], g.Rc, s=1.2, color=cols[order[sid] % 20], alpha=0.5, lw=0, rasterized=True)
        a, b = slopes[sid]; tt = np.linspace(g["T"].quantile(0.02), g["T"].quantile(0.98), 10); ax.plot(tt, np.exp(a + b * tt), "k-", lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(RCLO, RCHI); ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 8])); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.set_xlim(-5, 35); ax.grid(alpha=0.25)
        ax.set_title(f"{ids[sid]}: {S.loc[sid, 'F25'] / 1000:.1f} kHz, {S.loc[sid, 'R25']:.1f} Hz at 25 °C\n{100 * b:+.1f} % per °C, {int(S.loc[sid, 'n'])} singer-nights", fontsize=5.5, loc="left", pad=2)
        if c == 0:
            ax.set_ylabel("chirp rate (Hz)")
        if r == nrow - 1:
            ax.set_xlabel("air temperature (°C)")
    page2.finish(fig2, args.out + "_slopes")
    print(f"wrote {args.out}.pdf / .png (+ _slopes); tables in {croot}")


if __name__ == "__main__":
    main()
