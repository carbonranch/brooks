"""Sub-classification of a streak's matches by the shape of the fold (MBE, 2026-10-07).

For one streak, a random subsample of its firm singer-periods, one RANDOM 300 s window each
(not the deepest: a fair sample of what the type sounds like), the loudest 30 s for the band,
and a FINE fold: a 256-point STFT (5.8 ms frames, 0.73 ms hop, zero-padded to 2048 points) of
the band +/- PAD_HZ, period-averaged at the window's refined rate into NPH phase bins, in dB
above each row's quiet level over the period. Earlier shape work folded on the analysis grid
(11.6 ms frames), which smears a 6 ms pulse to twice its length; here the pulse is resolved.

Shape descriptors per window, from the fold:
  duty        share of the period the envelope is above half its peak
  rise, fall  phase from envelope onset to peak, peak to offset (fraction of the period)
  asym        (fall - rise) / (fall + rise): a pulse that builds slowly and stops dead is positive
  n_sub       sub-peaks in the envelope above 0.5 (a doubled pulse has 2)
  glide_hz    frequency centroid at the pulse's end minus its start (negative = falling)
  curve_hz    quadratic term of centroid against phase within the pulse (hooked vs straight)
  bw_hz       width of the pulse's spectrum at half its peak, at the envelope peak
  f_off       centroid at the envelope peak relative to the track's carrier (fraction)
  contrast    peak dB above quiet of the fold (signal quality, used to weight, not classify)
plus the window's context: carrier residual across the streak line (the sub-line coordinate),
position along it, temperature, site, month.

Then: the descriptors (standardized, weighted by contrast) go through PCA and a Gaussian
mixture with 1..MAX_K components compared by BIC; the aligned fold images (onset at phase 0,
frequency relative to the carrier) are clustered the same way. Each clustering is tested against
the context: does a cluster sit on a sub-line, at a site, in a season? The figure shows the mean
fold per cluster, the PCA plane colored by cluster and by the across-line residual, and the
descriptor distributions per cluster.

Usage:
  python analysis/callers/fold_subclass.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v9 \
      --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --streak 6901 --n 300 --jobs 8
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs as W  # noqa: E402
import specdisplay as SD  # noqa: E402

PAD_HZ = 700.0
NPH = 64
NPERSEG, HOP, NFFT = 256, 32, 2048
FOLD_S = 30.0
MAX_K = 4
MIN_CONTRAST = 6.0
MIN_WEIGHT = 0.08        # a mixture component holding less than this share of the windows is an outlier catcher, not a sub-type (5301: 3 noisy folds made a 'cluster')
TZ = "America/Chicago"
K_LINE = 0.55


def random_window(an: str, unit_roots, rng, min_depth_q=0.3):
    """A random window of the track among those with a syllable rate and depth above the track's min_depth_q quantile."""
    unit, track = an.rsplit(":", 1); track = int(track)
    period = "day" if unit.endswith("_day") else "night"
    for root in unit_roots:
        p = os.path.join(root, unit, f"night_{period}_global_window_animals.csv")
        if os.path.exists(p):
            Wd = pd.read_csv(p, usecols=["t", "track", "R", "F_fix", "F_lo", "F_hi", "depth", "chirp_rate"])
            g = Wd[(Wd.track == track) & Wd.F_fix.notna() & (Wd.R > 0) & Wd.chirp_rate.isna()]
            if g.empty:
                return None
            g = g[g.depth >= g.depth.quantile(min_depth_q)] if len(g) >= 4 else g
            b = g.iloc[rng.integers(len(g))]
            parts = unit.split("_")
            return dict(recorder=parts[0] + "_" + parts[1], site=parts[2], t=pd.Timestamp(b.t).tz_convert(TZ), R=float(b.R), F=float(b.F_fix), F_lo=float(b.F_lo), F_hi=float(b.F_hi))
    return None


def fine_fold(w: dict):
    sig, sr, _, _ = W.read_window(w["recorder"], w["site"], w["t"], 300.0)
    # loudest 30 s for the band and the refined rate, on the analysis grid
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2; dt = 128 / sr
    r_lo = max(0, int(np.searchsorted(f, w["F_lo"])) - 1); r_hi = min(len(f) - 1, int(np.searchsorted(f, w["F_hi"])) + 1)
    try:
        cols = W.best_segment(P, r_lo, r_hi, dt, w["R"], FOLD_S)
    except ValueError:
        cols = slice(0, P.shape[1])
    try:
        R = W.refine_rate(P, r_lo, r_hi, cols, dt, w["R"])
    except ValueError:
        R = w["R"]
    if not (0.97 * w["R"] <= R <= 1.03 * w["R"]):
        R = w["R"]
    a0 = cols.start * 128; a1 = min(len(sig), cols.stop * 128 + 512)
    seg = sig[a0:a1]
    ff, tf, Zf = scipy.signal.stft(seg, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP, nfft=NFFT, window="hann", boundary=None, padded=False)
    Pf = np.abs(Zf) ** 2
    rows = (ff >= w["F"] - PAD_HZ) & (ff <= w["F"] + PAD_HZ)
    Pf = Pf[rows]; fk = ff[rows]
    b = np.floor(((tf * R) % 1.0) * NPH).astype(int)
    F = np.zeros((Pf.shape[0], NPH)); cnt = np.bincount(b, minlength=NPH)
    for k in range(NPH):
        F[:, k] = Pf[:, b == k].sum(axis=1)
    F /= np.maximum(cnt, 1)
    Fdb = 10 * np.log10(F + 1e-20); Fdb -= np.percentile(Fdb, 10, axis=1, keepdims=True)
    return Fdb, fk, R


def describe(Fdb: np.ndarray, fk: np.ndarray, F_c: float, R: float) -> dict:
    band = (fk >= F_c - 400) & (fk <= F_c + 400)
    env = np.clip(Fdb[band], 0, None).sum(axis=0); env = env / (env.max() + 1e-12)
    # onset alignment: the rising half-crossing before the peak
    k_pk = int(np.argmax(env)); above = env >= 0.5
    k = k_pk
    while above[(k - 1) % NPH] and k > k_pk - NPH:
        k -= 1
    onset = k % NPH
    env_a = np.roll(env, -onset); F_a = np.roll(Fdb, -onset, axis=1)
    above_a = env_a >= 0.5
    duty = float(above_a.mean())
    k_pk_a = int(np.argmax(env_a))
    off = k_pk_a
    while off + 1 < NPH and above_a[off + 1]:
        off += 1
    rise = k_pk_a / NPH; fall = (off - k_pk_a + 1) / NPH
    asym = (fall - rise) / (fall + rise + 1e-9)
    sm = np.convolve(env_a, np.ones(3) / 3, "same")
    n_sub = int(((sm[1:-1] > sm[:-2]) & (sm[1:-1] >= sm[2:]) & (sm[1:-1] > 0.5)).sum())
    # spectral centroid per phase bin within the pulse, from power above quiet
    Pw = np.clip(F_a[band], 0, None); fb = fk[band]
    cen = np.array([(Pw[:, j] * fb).sum() / (Pw[:, j].sum() + 1e-12) if Pw[:, j].sum() > 0 else np.nan for j in range(NPH)])
    pulse = np.flatnonzero(above_a)
    if len(pulse) >= 3:
        ph = pulse / NPH; c = cen[pulse]; ok = np.isfinite(c)
        glide = float(c[ok][-1] - c[ok][0]) if ok.sum() >= 2 else np.nan
        curve = float(np.polyfit(ph[ok] - ph[ok].mean(), c[ok], 2)[0]) if ok.sum() >= 4 else np.nan
    else:
        glide = curve = np.nan
    spec = Pw[:, k_pk_a]; half = spec >= 0.5 * spec.max()
    bw = float(fb[half].max() - fb[half].min()) if half.any() else np.nan
    f_off = float(cen[k_pk_a] / F_c - 1) if np.isfinite(cen[k_pk_a]) else np.nan
    contrast = float(Fdb[band].max())
    return dict(duty=duty, rise=rise, fall=fall, asym=asym, n_sub=n_sub, glide_hz=glide, curve_hz=curve, bw_hz=bw, f_off=f_off, contrast=contrast,
                env=env_a, fold=F_a[band], fk=fb)


def _one(args):
    an, unit_roots, seed = args
    rng = np.random.default_rng(seed)
    try:
        w = random_window(an, unit_roots, rng)
        if w is None:
            return dict(an=an, status="no window")
        Fdb, fk, R = fine_fold(w)
        d = describe(Fdb, fk, w["F"], R)
        d.update(an=an, status="ok", t=str(w["t"]), R=R, F=w["F"], site=w["site"])
        return d
    except SystemExit as e:
        return dict(an=an, status=f"no audio: {str(e)[:40]}")
    except Exception as e:  # noqa: BLE001
        return dict(an=an, status=f"{type(e).__name__}: {str(e)[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--streak", type=int, required=True); ap.add_argument("--n", type=int, default=300); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detrend", action="store_true", help="regress each descriptor on the along-line position (which carries temperature) before clustering, so a shape that changes smoothly with rate is not read as sub-types")
    args = ap.parse_args()
    SD.paper_fonts()
    root = args.root; out_dir = os.path.join(root, "subclass"); os.makedirs(out_dir, exist_ok=True)
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    g = A[(A.streak == args.streak) & (A.form == "trill")] if "form" in A.columns else A[A.streak == args.streak]
    rng = np.random.default_rng(args.seed)
    pick = g.sample(min(args.n, len(g)), random_state=args.seed)
    print(f"streak {args.streak}: {len(g)} firm members, {len(pick)} sampled", flush=True)
    with Pool(args.jobs) as pool:
        res = pool.map(_one, [(an, args.unit_roots, int(rng.integers(1 << 31))) for an in pick.an], chunksize=4)
    ok = [r for r in res if r["status"] == "ok" and r["contrast"] >= MIN_CONTRAST]
    print(f"{len(ok)} windows with a fold of contrast >= {MIN_CONTRAST} dB ({sum(r['status'] != 'ok' for r in res)} failed)")
    D = pd.DataFrame([{k: v for k, v in r.items() if k not in ("env", "fold", "fk")} for r in ok]).merge(pick[["an", "F25", "R25", "T", "night"]], on="an")
    # context: across- and along-line coordinates relative to the streak's position
    lr0, lf0 = np.log(S.loc[args.streak, "R25"]), np.log(S.loc[args.streak, "F25"]); nk = np.hypot(1, K_LINE)
    D["across"] = ((np.log(D.F25) - lf0) - K_LINE * (np.log(D.R25) - lr0)) / nk
    D["along"] = ((np.log(D.R25) - lr0) + K_LINE * (np.log(D.F25) - lf0)) / nk
    D["month"] = pd.to_datetime(D.night).dt.month
    feats = ["duty", "rise", "fall", "asym", "n_sub", "glide_hz", "curve_hz", "bw_hz", "f_off"]
    X = D[feats].to_numpy(dtype=float); good = np.isfinite(X).all(axis=1); D = D[good].reset_index(drop=True); X = X[good]
    ok = [r for r, gg in zip(ok, good) if gg]
    if args.detrend:
        al = D.along.to_numpy(); Xd = X.copy()
        for j in range(X.shape[1]):
            b = np.polyfit(al, X[:, j], 1); Xd[:, j] = X[:, j] - np.polyval(b, al) + X[:, j].mean()
            print(f"  {feats[j]:9s} slope along the line {b[0]:+.3g} per log unit; variance explained {1 - np.var(Xd[:, j]) / np.var(X[:, j]):.0%}")
        X = Xd
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    pca = PCA(n_components=min(5, Xs.shape[1])).fit(Xs); Z = pca.transform(Xs)
    def best_mixture(Zk):
        bics, models = [], []
        for k in range(1, MAX_K + 1):
            gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=0).fit(Zk)
            bics.append(gm.bic(Zk) if gm.weights_.min() >= MIN_WEIGHT or k == 1 else np.inf); models.append(gm)
        kb = int(np.argmin(bics)) + 1
        return kb, models[kb - 1], bics
    kbest, gm, bics = best_mixture(Z[:, :3]); D["cluster"] = gm.predict(Z[:, :3])
    print("BIC by k (descriptors; inf = a component under the weight floor):", [round(b) if np.isfinite(b) else "x" for b in bics], "-> k =", kbest)
    # the fold images, aligned, on a common relative-frequency grid
    rel = np.linspace(-0.08, 0.08, 33)
    imgs = []
    for r in ok:
        fr = r["fk"] / r["F"] - 1
        img = np.stack([np.interp(rel, fr, np.clip(r["fold"][:, j], 0, None), left=0, right=0) for j in range(NPH)], axis=1)
        imgs.append(img / (img.max() + 1e-9))
    IM = np.stack(imgs)
    IMf = IM.reshape(len(IM), -1)
    if args.detrend:
        al = D.along.to_numpy(); A1 = np.column_stack([np.ones_like(al), al])
        beta, *_ = np.linalg.lstsq(A1, IMf, rcond=None); IMf = IMf - A1 @ beta + IMf.mean(axis=0)
    pca2 = PCA(n_components=6).fit(IMf); Z2 = pca2.transform(IMf)
    k2, gm2, bics2 = best_mixture(Z2[:, :4]); D["cluster_img"] = gm2.predict(Z2[:, :4])
    print("BIC by k (fold images):", [round(b) if np.isfinite(b) else "x" for b in bics2], "-> k =", k2)
    # association of clusters with context
    def report(col):
        print(f"\nclusters by {col}:")
        for c, gc in D.groupby(col):
            print(f"  {c}: n {len(gc):3d}; across {gc.across.mean():+.3f} (sd {gc.across.std():.3f}); along {gc.along.mean():+.2f}; T {gc['T'].mean():.1f}; months {sorted(gc.month.unique())}; "
                  f"sites {gc.site.value_counts().head(3).to_dict()}; duty {gc.duty.mean():.2f} glide {gc.glide_hz.mean():+.0f} curve {gc.curve_hz.mean():+.0f} bw {gc.bw_hz.mean():.0f}")
        if D[col].nunique() > 1:
            from scipy.stats import f_oneway, kruskal
            for v in ("across", "along", "T"):
                groups = [gc[v].to_numpy() for _, gc in D.groupby(col)]
                print(f"  Kruskal {v}: p = {kruskal(*groups).pvalue:.2g}")
    report("cluster"); report("cluster_img")
    D.to_csv(os.path.join(out_dir, f"fold_subclass_{args.streak}{'_detrend' if args.detrend else ''}.csv"), index=False, float_format="%.4f")
    np.save(os.path.join(out_dir, f"fold_subclass_{args.streak}_folds.npy"), IM)

    # figure
    kk = max(kbest, 1); cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(3, max(kk, 3), height_ratios=[1.2, 1, 1], hspace=0.5, wspace=0.3)
    for c in range(kk):
        ax = fig.add_subplot(gs[0, c]); sel = D.cluster.to_numpy() == c
        ax.imshow(IM[sel].mean(axis=0), origin="lower", aspect="auto", extent=[0, 1, rel[0] * 100, rel[-1] * 100], cmap="magma")
        ax.set_title(f"cluster {c}: {sel.sum()} windows; mean fold\nacross {D.across[sel].mean():+.3f}, T {D['T'][sel].mean():.0f} °C", fontsize=9, color=cmap(c))
        ax.set_xlabel("phase (period)"); ax.set_ylabel("% from carrier" if c == 0 else "")
    ax = fig.add_subplot(gs[1, 0]); ax.scatter(Z[:, 0], Z[:, 1], c=[cmap(c) for c in D.cluster], s=14, lw=0); ax.set_xlabel("PC1 (descriptors)"); ax.set_ylabel("PC2"); ax.set_title(f"descriptor PCA, GMM k = {kbest} (BIC)", fontsize=9)
    ax = fig.add_subplot(gs[1, 1]); sc = ax.scatter(Z[:, 0], Z[:, 1], c=D.across, cmap="coolwarm", vmin=-0.08, vmax=0.08, s=14, lw=0); ax.set_xlabel("PC1"); ax.set_title("colored by across-line residual (sub-line)", fontsize=9); fig.colorbar(sc, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[1, 2]); sc = ax.scatter(D.along, D.across, c=[cmap(c) for c in D.cluster], s=14, lw=0); ax.set_xlabel("along the line (log units)"); ax.set_ylabel("across"); ax.set_title("the sample on the streak, by cluster", fontsize=9)
    for i, v in enumerate(["duty", "glide_hz", "curve_hz", "bw_hz", "asym"][:max(kk, 3)]):
        ax = fig.add_subplot(gs[2, i])
        for c in range(kk):
            ax.hist(D[v][D.cluster == c], bins=20, alpha=0.6, color=cmap(c))
        ax.set_title(v, fontsize=9)
    fig.suptitle(textwrap.fill(f"streak {args.streak} ({S.loc[args.streak, 'F25'] / 1000:.2f} kHz @ {S.loc[args.streak, 'R25']:.0f} Hz): {len(D)} random windows of {len(pick)} sampled firm singer-periods, fine fold (5.8 ms frames) of the band at the window's rate over its loudest 30 s; "
                               f"shape descriptors{' (detrended on the along-line position)' if args.detrend else ''} -> PCA -> Gaussian mixture; BIC picks k = {kbest} on descriptors, {k2} on the fold images", 170), fontsize=10)
    tag = "_detrend" if args.detrend else ""
    fig.savefig(os.path.join(out_dir, f"fold_subclass_{args.streak}{tag}.png"), dpi=120, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, f"fold_subclass_{args.streak}{tag}.pdf"), bbox_inches="tight")
    print("wrote", os.path.join(out_dir, f"fold_subclass_{args.streak}{tag}.png"))


if __name__ == "__main__":
    main()
