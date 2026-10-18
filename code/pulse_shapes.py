"""Per-pulse shape from the raw spectrogram, no folding: for each singer-period, the
loudest 30 s of its deepest window is segmented into pulses on the band-limited power
series, and each pulse gives a glide (ridge frequency change from its start to its end,
Hz), a duration (ms) and a bandwidth. The singer-period's shape is the median over its
pulses; pulse-to-pulse spread is kept as a quality measure.

Why: the fold smears a glide when the rate is slightly off over 30 s, and the fold-based
glide descriptor (year_shapes.py) compresses a 600 Hz sweep into a few hundred Hz, so the
two shapes the eye sees in some streaks (9101, 6901; fixed-position galleries) did not
separate. Per-pulse measurement at a 1.45 ms hop avoids both (MBE, 2026-10-04: "worth a
try").

Pulses: peaks of the smoothed band power (band +/- one row) at least 0.6 / R apart with
prominence >= PROM_DB above the local floor; only pulses whose spacing to a neighbour is
within 25 % of the track's period are kept (the animal, not a neighbour), and at least
MIN_PULSES are needed. For each pulse the 'on' frames are those within +/- 0.45 period
whose band power is >= half the pulse peak; the ridge per frame is the power-weighted
frequency over the band +/- RIDGE_PAD; glide = slope of ridge on time x on-duration.

Writes pulse_shapes<suffix>.csv (per singer-period, resumable), pulse_shapes<suffix>.png
(per streak: glide vs duration, colored by T, two-component test on glide) and
pulse_shapes_streaks<suffix>.csv.

Usage:
    python analysis/callers/pulse_shapes.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --jobs 12
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_blobs as W  # noqa: E402
import streak_gallery as G  # noqa: E402

MIN_N, CAP = 100, 400
NFINE = 256              # frame for the pulse measurement: 5.8 ms (a 512-point frame, 11.6 ms, is longer than a 9 ms pulse and averages its sweep away)
HOP = 32                 # 0.73 ms hop
PROM_DB = 3.0
RIDGE_PAD = 300.0
MIN_PULSES = 15
DF = 44100 / 512


def measure(w: dict) -> dict:
    sig, sr, t0, _ = W.read_window(w["recorder"], w["site"], w["t"], 300.0)
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2; dt = 128 / sr
    r_lo = max(0, int(np.searchsorted(f, w["ref_lo"])) - 1); r_hi = min(len(f) - 1, int(np.searchsorted(f, w["ref_hi"])) + 1)
    try:
        cols = W.best_segment(P, r_lo, r_hi, dt, w["R"], 30.0)
    except ValueError:
        cols = slice(0, P.shape[1])
    R = w["R"]
    try:
        Rr = W.refine_rate(P, r_lo, r_hi, cols, dt, R)
        if 0.97 * R <= Rr <= 1.03 * R:
            R = Rr
    except ValueError:
        pass
    # fine spectrogram of the loudest 30 s
    s0, s1 = cols.start * 128, min(len(sig), cols.stop * 128 + 512)
    f2, _, Z2 = scipy.signal.stft(sig[s0:s1], fs=sr, nperseg=NFINE, noverlap=NFINE - HOP, window="hann", boundary=None, padded=False)
    P2 = np.abs(Z2) ** 2; dt2 = HOP / sr; df2 = sr / NFINE
    b_lo = max(0, int(np.searchsorted(f2, w["F_lo"])) - 1); b_hi = min(len(f2) - 1, int(np.searchsorted(f2, w["F_hi"])) + 1)
    band = P2[b_lo:b_hi + 1].sum(axis=0)
    lev = 10 * np.log10(band + 1e-20)
    lev_s = scipy.ndimage.uniform_filter1d(lev, 3)
    floor = scipy.ndimage.median_filter(lev_s, size=int(round(2.0 / R / dt2)) | 1)
    per = 1.0 / R
    peaks, props = scipy.signal.find_peaks(lev_s - floor, distance=max(1, int(0.6 * per / dt2)), prominence=PROM_DB)
    if len(peaks) < MIN_PULSES:
        return dict(status="too few pulses", n_pulses=int(len(peaks)), R=R)
    # keep pulses spaced like this animal
    gaps = np.diff(peaks) * dt2
    ok_gap = np.abs(gaps / per - 1) < 0.25
    keep = np.zeros(len(peaks), bool); keep[:-1] |= ok_gap; keep[1:] |= ok_gap
    peaks = peaks[keep]
    if len(peaks) < MIN_PULSES:
        return dict(status="pulses not periodic", n_pulses=int(len(peaks)), R=R)
    p_lo = max(0, int(np.searchsorted(f2, w["F_lo"] - RIDGE_PAD)) - 1); p_hi = min(len(f2) - 1, int(np.searchsorted(f2, w["F_hi"] + RIDGE_PAD)) + 1)
    fk = f2[p_lo:p_hi + 1]
    half = int(round(0.45 * per / dt2))
    glides, durs, bws, f0s, snrs = [], [], [], [], []
    for pk in peaks:
        a, b = max(0, pk - half), min(P2.shape[1], pk + half + 1)
        seg = band[a:b]
        on = seg >= 0.5 * seg.max()
        # contiguous on-run containing the peak
        i0 = pk - a
        lo = i0
        while lo > 0 and on[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(on) - 1 and on[hi + 1]:
            hi += 1
        frames = np.arange(a + lo, a + hi + 1)
        if len(frames) < 2:
            continue
        Q = P2[p_lo:p_hi + 1, frames]
        Q = Q - np.median(P2[p_lo:p_hi + 1, a:b], axis=1, keepdims=True)   # remove steady neighbours in the pad
        Q = np.clip(Q, 0, None)
        ridge = np.array([np.average(fk, weights=Q[:, j] + 1e-12) for j in range(Q.shape[1])])
        tt = np.arange(len(frames)) * dt2
        slope = np.polyfit(tt, ridge, 1)[0] if len(frames) > 2 else (ridge[-1] - ridge[0]) / max(tt[-1], dt2)
        glides.append(slope * (tt[-1] if len(frames) > 2 else tt[-1] + dt2))
        durs.append(len(frames) * dt2 * 1000)
        rp = Q.mean(axis=1); bws.append(float((rp >= 0.5 * rp.max()).sum() * df2))
        f0s.append(float(np.average(fk, weights=rp + 1e-12)))
        snrs.append(float((lev_s - floor)[pk]))
    if len(glides) < MIN_PULSES:
        return dict(status="too few measurable pulses", n_pulses=int(len(glides)), R=R)
    g = np.array(glides); d = np.array(durs)
    return dict(status="ok", n_pulses=int(len(g)), R=R, glide_hz=float(np.median(g)), glide_iqr=float(np.subtract(*np.percentile(g, [75, 25]))),
                dur_ms=float(np.median(d)), dur_iqr=float(np.subtract(*np.percentile(d, [75, 25]))), duty=float(np.median(d) / 1000 * R),
                bw_hz=float(np.median(bws)), f0=float(np.median(f0s)), snr_db=float(np.median(snrs)), F_fix=w["F_fix"], t=str(w["t"]))


def _one(args):
    an, unit_roots = args
    w = G.window_of(an, unit_roots)
    if w is None:
        return dict(an=an, status="no window")
    try:
        d = measure(w); d["an"] = an; return d
    except SystemExit as e:
        return dict(an=an, status=f"no audio: {str(e)[:40]}")
    except Exception as e:  # noqa: BLE001
        return dict(an=an, status=f"{type(e).__name__}: {str(e)[:40]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--cap", type=int, default=CAP); ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--streaks", type=int, nargs="*", default=None)
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[(S.harmonic_of < 0) & (S.n >= args.min_n)] if "harmonic_of" in S.columns else S[S.n >= args.min_n]
    if args.streaks:
        S = S.loc[[s for s in args.streaks if s in S.index]]
    out = os.path.join(root, f"pulse_shapes{args.suffix}.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["an"])
    todo = []
    for sid in S.index:
        g = A[A.streak == sid].sort_values("depth", ascending=False).head(args.cap)
        todo += [an for an in g.an if an not in set(done.an)]
    print(f"{len(S)} streaks, {len(todo)} singer-periods to measure ({len(done)} done)", flush=True)
    rows = done.to_dict("records")
    if todo:
        with Pool(args.jobs) as pool:
            for k, d in enumerate(pool.imap_unordered(_one, [(an, args.unit_roots) for an in todo], chunksize=4), 1):
                rows.append(d)
                if k % 200 == 0:
                    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.3f"); print(f"  {k}/{len(todo)}", flush=True)
    D = pd.DataFrame(rows); D.to_csv(out, index=False, float_format="%.3f")
    print("status counts:", D.status.value_counts().to_dict())
    D = D[D.status == "ok"].merge(A[["an", "streak", "T"]], on="an")
    from sklearn.mixture import GaussianMixture
    rows_s = []
    sids = [s for s in S.sort_values("n", ascending=False).index if (D.streak == s).sum() >= 30]
    ncol = 4; nrow = int(np.ceil(len(sids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, sid in zip(axes.ravel(), sids):
        g = D[(D.streak == sid) & (D.snr_db >= 6) & (D.n_pulses >= MIN_PULSES)]
        X = g.glide_hz.to_numpy().reshape(-1, 1)
        if len(X) < 6:                       # too few clear members for a mixture (a streak of mostly faint singers)
            ax.set_title(f"{sid}: {len(X)} clear members", fontsize=8); ax.set_axis_off(); continue
        g1 = GaussianMixture(1, random_state=0).fit(X); g2 = GaussianMixture(2, random_state=0, n_init=3).fit(X)
        dbic = g1.bic(X) - g2.bic(X)
        wts = g2.weights_; order = np.argsort(g2.means_.ravel()); mu = g2.means_.ravel()[order]; sd = np.sqrt(g2.covariances_.ravel())[order]
        sep = (mu[1] - mu[0]) / np.sqrt(0.5 * (sd[0] ** 2 + sd[1] ** 2))
        split = bool(dbic >= 10 and wts.min() >= 0.2 and sep >= 2.0)
        rows_s.append(dict(streak=sid, n=len(g), glide_med=g.glide_hz.median(), glide_iqr=g.glide_hz.quantile(.75) - g.glide_hz.quantile(.25),
                           dur_med=g.dur_ms.median(), duty_med=g.duty.median(), bw_med=g.bw_hz.median(), pulse_iqr_med=g.glide_iqr.median(),
                           dbic=dbic, mu_lo=mu[0], mu_hi=mu[1], w_min=wts.min(), sep_sd=sep, split=split))
        sc = ax.scatter(g.glide_hz, g.dur_ms, s=7, alpha=0.5, c=g["T"], cmap="jet", vmin=10, vmax=30, edgecolors="none")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{sid}  {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz  n {len(g)}" + (f"  SPLIT {mu[0]:.0f}/{mu[1]:.0f} Hz" if split else ""),
                     fontsize=8, loc="left", color="tab:red" if split else "k")
        ax.set_xlim(-1200, 600); ax.set_ylim(0, 40); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(sids):]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        ax.set_xlabel("per-pulse glide, median over pulses (Hz, signed)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("pulse duration (ms)", fontsize=8)
    fig.suptitle(f"Shape within streaks from the raw spectrogram: median per-pulse glide and pulse duration per singer-period (loudest 30 s, {1000 * NFINE / 44100:.1f} ms frames, {1000 * HOP / 44100:.2f} ms hop), "
                 "colored by temperature; SPLIT = glide is two-component (dBIC >= 10, minor >= 20 %, >= 2 sd apart)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(root, f"pulse_shapes{args.suffix}.png"), dpi=130); plt.close(fig)
    Rt = pd.DataFrame(rows_s); Rt.to_csv(os.path.join(root, f"pulse_shapes_streaks{args.suffix}.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 240)
    print(Rt.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
