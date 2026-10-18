"""Is a streak one population? Four numbers per streak, before any name is given.

1. Co-occurrence spread. Two singers of one type on the same recorder in the same
   hour share a microclimate, so their positions along the line can differ only by
   what individuals of one species differ by. For every pair of singer-hours of the
   streak on one recorder in one hour: |delta along| in %. The null is one
   individual's own hour-to-hour scatter (consecutive hours of one singer-period).
   incoh_frac = fraction of co-occurring pairs separated by more than PAIR_TOL %
   (minus the null's fraction). Needs no thermometer and no shape.
2. Shared response. Residuals of along-line position about the streak's year-wide
   fit on HRRR temperature, paired between DIFFERENT recorders in the same hour:
   their correlation. One species tracking one temperature shares its residual
   property-wide (the month's species detector); a mixture dilutes it.
3. Descriptor unimodality. Pulse duration, sweep rate (glide / duration), bandwidth
   (pulse_shapes.py), chirp share (rc_frac of hours), across-line offset, within-track
   coupling (year_coupling_tracks): each tested for two components (dBIC >= 10,
   minor >= 20 %, >= 2 sd apart). n_split = how many descriptors split.
4. Seasonal stability. Largest monthly median of the along-line residual (%), from
   the same fit as (2).

Verdict: coherent if incoh_frac <= INCOH_MAX, shared >= SHARED_MIN, n_split == 0.
Otherwise listed with the numbers that failed.

Usage:
    python analysis/callers/streak_coherence.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

TZ = "America/Chicago"
HF_SPLIT, C_LO, C_HI = 10000.0, 0.55, 0.25
PAIR_TOL = 8.0        # % along the line: beyond one species' individual spread (between-individual sd ~3-4 %)
INCOH_MAX = 0.15      # calibrated on 5301 (O. texensis, reference recording on its line, one shape in the gallery): 0.13
SHARED_MIN = 0.3
SLOPE_MIN = 1.0       # %/degC along the line: below this the residuals are noise and the shared-response test does not apply
MIN_PAIRS = 30
SD_RATIO = 2.5        # two-component guard: wider component / narrower component above this is a tail, not a band


def frame(S, sid):
    c = C_LO if S.loc[sid, "F25"] < HF_SPLIT else C_HI
    return c


def along_across(R, F, c, lr0, lf0):
    lr, lf = np.log(R), np.log(F)
    along = ((lr - lr0) + c * (lf - lf0)) / np.hypot(1, c)
    across = ((lf - lf0) - c * (lr - lr0)) / np.hypot(1, c)
    return along, across


def two_component(x: np.ndarray) -> tuple[bool, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 40:
        return False, np.nan, np.nan
    X = x.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=0).fit(X); g2 = GaussianMixture(2, random_state=0, n_init=3).fit(X)
    dbic = g1.bic(X) - g2.bic(X)
    o = np.argsort(g2.means_.ravel()); mu = g2.means_.ravel()[o]; sd = np.sqrt(g2.covariances_.ravel())[o]
    sep = (mu[1] - mu[0]) / np.sqrt(0.5 * (sd[0] ** 2 + sd[1] ** 2))
    # a component more than SD_RATIO times wider than the other is the tail of a skewed distribution,
    # not a second band (pulse durations are long-tailed: 5001 runs smoothly from 3 to 14 ms and
    # 'split' into a narrow core and a wide tail) -- the same guard split_streaks_by_T uses
    tail = (sd.max() / max(sd.min(), 1e-9)) > SD_RATIO
    return bool(dbic >= 10 and g2.weights_.min() >= 0.2 and sep >= 2.0 and not tail), float(dbic), float(sep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--clear-only", action="store_true",
                    help="restrict the position tests to members with a measured pulse SNR >= 6 dB (pulse_shapes); weak tracks inflated the co-occurrence spread of 9101 from 4 %% to 20 %%")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[(S.harmonic_of < 0) & (S.n >= args.min_n)] if "harmonic_of" in S.columns else S[S.n >= args.min_n]
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv")).drop(columns=["streak"], errors="ignore").merge(A[["an", "streak"]], on="an")
    P["hour"] = pd.to_datetime(P.hour, utc=True).dt.tz_convert(TZ)
    P = P[P.streak.isin(S.index) & P.R.notna() & (P.R > 0)]
    pulses = pd.read_csv(os.path.join(root, f"pulse_shapes{args.suffix}.csv")) if os.path.exists(os.path.join(root, f"pulse_shapes{args.suffix}.csv")) else None
    tagc = ""
    if args.clear_only and pulses is not None:
        clear = set(pulses[(pulses.status == "ok") & (pulses.snr_db >= 6)].an)
        n0 = P.an.nunique(); P = P[P.an.isin(clear)]
        print(f"clear-only: {P.an.nunique()} of {n0} singer-periods have pulse SNR >= 6 dB", flush=True)
        tagc = "_clear"
    coup = pd.read_csv(os.path.join(root, f"year_coupling_tracks{args.suffix}.csv")) if os.path.exists(os.path.join(root, f"year_coupling_tracks{args.suffix}.csv")) else None
    mp = os.path.join(root, "streak_sina_matches.csv")
    cand = pd.read_csv(mp).query("rank == 1").set_index("streak").species.to_dict() if os.path.exists(mp) else {}

    rows = []
    for sid in S.sort_values("n", ascending=False).index:
        g = P[P.streak == sid].copy()
        if len(g) < 100:
            continue
        c = frame(S, sid)
        lr0, lf0 = np.log(g.R).median(), np.log(g.F).median()
        g["along"], g["across"] = along_across(g.R.to_numpy(), g.F.to_numpy(), c, lr0, lf0)
        # (1) co-occurrence: pairs of different singer-periods, same recorder, same hour
        pairs = []
        for (rec, h), gg in g.groupby(["recorder", "hour"]):
            if gg.an.nunique() < 2:
                continue
            al = gg.groupby("an").along.median().to_numpy()
            d = np.abs(al[:, None] - al[None, :])[np.triu_indices(len(al), 1)]
            pairs.append(100 * d)
        pairs = np.concatenate(pairs) if pairs else np.array([])
        # null: one singer-period's consecutive-hour steps
        g = g.sort_values(["an", "hour"])
        step = g.groupby("an").along.diff().abs().dropna().to_numpy() * 100
        null_frac = float((step > PAIR_TOL).mean()) if len(step) else np.nan
        pair_frac = float((pairs > PAIR_TOL).mean()) if len(pairs) >= MIN_PAIRS else np.nan
        incoh = pair_frac - null_frac if np.isfinite(pair_frac) else np.nan
        # (2) shared response: residual about the year fit on T, paired across recorders in the same hour
        ok = np.isfinite(g["T"])
        b, a = np.polyfit(g["T"][ok], g.along[ok], 1)
        g["res"] = g.along - (a + b * g["T"])
        hr = g.groupby(["hour", "recorder"]).res.median().reset_index()
        xs, ys = [], []
        for h, gg in hr.groupby("hour"):
            if len(gg) >= 2:
                r = gg.res.to_numpy()
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        xs.append(r[i]); ys.append(r[j])
        shared = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= MIN_PAIRS else np.nan
        # (4) seasonal stability
        g["month"] = g.hour.dt.to_period("M")
        mres = g.groupby("month").res.median() * 100
        mres = mres[g.groupby("month").size() >= 20]
        season = float(mres.abs().max()) if len(mres) else np.nan
        # (3) descriptors
        desc = {}
        an_streak = A[A.streak == sid]
        _, dbic_x, _ = two_component(an_streak.F.to_numpy() * 0 + along_across(an_streak.R.to_numpy(), an_streak.F.to_numpy(), c, lr0, lf0)[1] * 100)
        desc["across"] = two_component(along_across(an_streak.R.to_numpy(), an_streak.F.to_numpy(), c, lr0, lf0)[1] * 100)[0]
        hp = P[P.streak == sid].groupby("an").rc_frac.mean().to_numpy() if "rc_frac" in P.columns else np.array([])
        desc["chirp_share"] = bool(len(hp) >= 40 and (hp > 0.5).mean() >= 0.2 and (hp <= 0.5).mean() >= 0.2)
        if pulses is not None:
            pp = pulses[(pulses.status == "ok") & pulses.an.isin(an_streak.an) & (pulses.snr_db >= 6)]
            desc["duration"] = two_component(pp.dur_ms.to_numpy())[0]
            desc["sweep_rate"] = two_component((pp.glide_hz / pp.dur_ms).to_numpy())[0]
            # bandwidth is quantized to the row spacing (172/344/516 Hz) and splits trivially: not tested
        if coup is not None:
            cc = coup[coup.an.isin(an_streak.an) & coup.slope.notna()]
            desc["coupling"] = two_component(cc.slope.to_numpy())[0]
        n_split = int(sum(desc.values()))
        slope_pct = 100 * b
        shared_applies = abs(slope_pct) >= SLOPE_MIN
        # Verdict (MBE, 2026-10-05): a call type is one song form on one line. Parallel sub-lines of the
        # same song (across-line bimodality, co-occurrence spread) are structure within the type and are
        # REPORTED, not gated on; a type is 'mixed' only when a descriptor says a different song shares
        # the line: chirp share (chirper + triller), pulse duration or sweep rate.
        # 'mixed' needs the chirp-share split (a chirper and a triller on one line) or BOTH a duration and a
        # sweep-rate split; a duration split alone fired on smooth long-tailed distributions (5001: 3-14 ms,
        # no gap) and is reported but not acted on
        song_splits = [k for k, v in desc.items() if v and k in ("chirp_share", "duration", "sweep_rate")]
        mixed = desc.get("chirp_share", False) or (desc.get("duration", False) and desc.get("sweep_rate", False))
        verdict = "mixed song forms" if mixed else "one song form"
        pass_pos = np.isfinite(incoh) and incoh <= INCOH_MAX
        pass_shared = (not shared_applies) or (np.isfinite(shared) and shared >= SHARED_MIN)
        position_note = "single population" if (pass_pos and pass_shared) else ("too few pairs" if not np.isfinite(incoh) else "sub-structure")
        rows.append(dict(streak=sid, n=int(S.loc[sid, "n"]), F25=S.loc[sid, "F25"], R25=S.loc[sid, "R25"], candidate=cand.get(sid, ""),
                         pairs=len(pairs), pair_frac=pair_frac, null_frac=null_frac, incoh_frac=incoh, slope_pct_per_C=slope_pct, shared=shared,
                         shared_applies=shared_applies, n_shared_pairs=len(xs), season_max_pct=season, n_split=n_split,
                         splits=",".join(k for k, v in desc.items() if v), verdict=verdict, position=position_note, pairs_arr=pairs, null_arr=step))
    D = pd.DataFrame(rows)
    # the co-occurrence test, visible: pair separations against the one-singer null, largest streaks
    top = D.sort_values("n", ascending=False).head(16)
    ncol = 4; nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), sharex=True, squeeze=False)
    bins = np.linspace(0, 40, 41)
    for ax, r in zip(axes.ravel(), top.itertuples()):
        if len(r.null_arr):
            ax.hist(np.clip(r.null_arr, 0, 40), bins=bins, density=True, color="0.6", alpha=0.7, label="one singer, hour to hour (null)")
        if len(r.pairs_arr):
            ax.hist(np.clip(r.pairs_arr, 0, 40), bins=bins, density=True, histtype="step", color="tab:red", lw=1.4, label="two singers, same recorder & hour")
        ax.axvline(PAIR_TOL, color="k", lw=0.6, ls=":")
        ax.set_title(f"{r.streak}  {r.F25 / 1000:.2f} kHz @ {r.R25:.0f} Hz   {r.pairs} pairs, {100 * (r.pair_frac if np.isfinite(r.pair_frac) else 0):.0f} % > {PAIR_TOL:.0f} % (null {100 * r.null_frac:.0f} %)",
                     fontsize=7.5, loc="left", color="tab:red" if r.verdict == "mixed song forms" else "k")
        ax.tick_params(labelsize=7); ax.set_yticks([])
    for ax in axes.ravel()[len(top):]:
        ax.set_axis_off()
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("separation along the line (%)", fontsize=8)
    fig.suptitle("Co-occurrence test: two members of one streak on the same recorder in the same hour share a microclimate, so one species keeps them within "
                 "its individual spread; a mixture does not", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(root, f"streak_coherence_pairs{tagc}{args.suffix}.png"), dpi=140); plt.close(fig)
    D = D.drop(columns=["pairs_arr", "null_arr"])
    D.to_csv(os.path.join(root, f"streak_coherence{tagc}{args.suffix}.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 100)
    print(D.round(2).to_string(index=False))
    print(f"\n{(D.verdict == 'one song form').sum()} of {len(D)} streaks are one song form; {(D.position == 'single population').sum()} also pass the position tests")
    # figure: incoherence fraction vs shared response, labeled
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in D.itertuples():
        col = "tab:red" if r.verdict == "mixed song forms" else ("tab:green" if r.position == "single population" else "tab:orange")
        ax.scatter(r.incoh_frac, r.shared, s=np.clip(np.sqrt(r.n) * 1.5, 10, 120), color=col, alpha=0.8, edgecolors="k", linewidths=0.4)
        ax.annotate(f"{r.streak} {r.F25 / 1000:.1f}k@{r.R25:.0f}", (r.incoh_frac, r.shared), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.axvline(INCOH_MAX, color="k", lw=0.6, ls=":"); ax.axhline(SHARED_MIN, color="k", lw=0.6, ls=":")
    ax.set_xlabel(f"incoherence: fraction of same-recorder same-hour pairs > {PAIR_TOL:.0f} % apart along the line, minus the one-singer null")
    ax.set_ylabel("shared response: correlation of same-hour residuals between recorders")
    ax.set_title("Streak coherence: red = a different song shares the line (chirp share, pulse duration or sweep rate split); "
                 "green = one song form and a single population by position; orange = one song form with sub-structure", fontsize=9, loc="left")
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"streak_coherence{tagc}{args.suffix}.png"), dpi=140); plt.close(fig)
    print("wrote", os.path.join(root, f"streak_coherence{tagc}{args.suffix}.csv / .png"))


if __name__ == "__main__":
    main()
