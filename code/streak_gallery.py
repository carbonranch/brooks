"""Per streak: a sampling of singers along the line, as (1) 1-second spectrogram
excerpts on one common carrier axis and (2) their folded syllable profiles.

For each streak, N singer-periods are chosen at evenly spaced quantiles of the
along-line position (cold/slow end to warm/fast end), preferring deep tracks and
distinct recorder-nights. For each, the track's deepest window with a fixed
carrier is read from the archive (300 s), the loudest 30 s for the animal's band
is found exactly as the run does (window_blobs.best_segment), the 1 s with the
most in-band power inside it is drawn as a spectrogram -- every panel 0-22 kHz,
dB above the row's quiet level -- with the animal's band bracketed; and the band
+/- PAD_HZ is folded at the track's rate over the same 30 s (window_blobs.fold,
NPH phase bins) and drawn as modulated power about the row mean against time in
ms over two periods, so the unit's stretching with cooling is visible along the
row of panels while its shape is not supposed to change.

Writes <root>/gallery/streak_<id>_windows.png and _profiles.png, a combined
gallery<suffix>.pdf, and gallery_index<suffix>.csv (which singers were drawn).

Usage:
    python analysis/callers/streak_gallery.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 [--streaks 8901 6901] [--n 10] [--jobs 6]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import scipy.signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_blobs as W  # noqa: E402
import specdisplay as SD  # noqa: E402
SD.paper_fonts()

PAD_HZ = 600.0
ZOOM_HZ = 1000.0     # half-height of the zoomed band strip under each 1 s panel
NPH = 48
FOLD_S = 30.0
EXCERPT_S = 1.0
FMAX = 22050.0
HF_SPLIT, C_LO, C_HI = 10000.0, 0.55, 0.25
TZ = "America/Chicago"


def choose(A: pd.DataFrame, c: float, n: int, mode: str = "along") -> pd.DataFrame:
    """mode 'along': n singer-periods at even quantiles of along-line position, deep ones, distinct
    recorder-nights -- shows the line's extent but maximizes mixing if the line hosts a succession.
    mode 'fixed': the n deepest, longest singer-periods within a narrow slice (+/- 0.03 in log-rate
    units, ~3 %) about the streak's densest along-line position, distinct recorder-nights -- the test
    of whether the streak is one thing at one temperature (MBE, 2026-10-04)."""
    A = A.copy()
    lr, lf = np.log(A.R), np.log(A.F)
    A["along"] = ((lr - lr.median()) + c * (lf - lf.median())) / np.hypot(1, c)
    if mode == "fixed":
        hist, edges = np.histogram(A.along, bins=24)
        centre = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
        sl = A[(A.along - centre).abs() <= 0.03]
        q = sl.assign(score=sl.depth * np.sqrt(sl.n_hours if "n_hours" in sl else 1.0)).sort_values("score", ascending=False)
        used, picks = set(), []
        for _, r in q.iterrows():
            if (r.recorder, r.night) in used:
                continue
            used.add((r.recorder, r.night)); picks.append(r)
            if len(picks) >= n:
                break
        return pd.DataFrame(picks).sort_values("T")
    good = A[A.depth >= A.depth.quantile(0.4)] if len(A) >= 20 else A
    targets = np.quantile(A.along, np.linspace(0.03, 0.97, n))
    used, picks = set(), []
    for tq in targets:
        cand = good[~good.apply(lambda r: (r.recorder, r.night) in used, axis=1)] if used else good
        if cand.empty:
            cand = good
        i = (cand.along - tq).abs().idxmin()
        r = cand.loc[i]
        used.add((r.recorder, r.night)); picks.append(r)
    return pd.DataFrame(picks).sort_values("along")


OVERLAP_PAD = 0.10       # another kept track whose band overlaps this track's band +/- 10 % makes a window second choice


def window_of(an: str, unit_roots: list[str], form: str | None = None, avoid_overlap: bool = True):
    """The track's deepest window. form = 'chirp' prefers windows that carry a chirp rate, 'trill' those
    that do not: the deepest window of a chirper is often one in which it trilled continuously (depth
    favors continuous modulation), so a grid of a chirp type drawn from deepest windows shows trills
    (MBE, 2026-10-06: 'the plots under v7 have trillers and chirpers mixed')."""
    unit, track = an.split(":"); track = int(track)
    period = "day" if unit.endswith("_day") else "night"          # day units hold night_day_global_window_animals.csv
    for root in unit_roots:
        p = os.path.join(root, unit, f"night_{period}_global_window_animals.csv")
        if os.path.exists(p):
            Wd = pd.read_csv(p)
            g = Wd[(Wd.track == track) & Wd.F_fix.notna()].sort_values("depth", ascending=False)
            if form in ("chirp", "trill") and "chirp_rate" in g:
                gf = g[g.chirp_rate.notna()] if form == "chirp" else g[g.chirp_rate.isna()]
                if not gf.empty:
                    g = gf
            # the deepest window WITH a syllable rate: a chirper's deepest window may carry only the
            # chirp rate, and a fold at 2-4 Hz is not a syllable profile (it also stretches the shared axis)
            gr = g[g.R.notna() & (g.R > 0)]
            if not gr.empty:
                g = gr
            if avoid_overlap and "keep" in Wd.columns and len(g) > 1:
                # prefer a window in which no other kept track sings in this track's band: the loudest half second of a
                # window with a louder neighbor in the band shows the neighbor (MBE on 4801/6701: interlopers)
                others = Wd[(Wd.track != track) & (Wd.keep == True)]          # noqa: E712
                def clean(row):
                    o = others[others.t == row.t]
                    return not ((o.F_lo <= row.F_hi * (1 + OVERLAP_PAD)) & (o.F_hi >= row.F_lo * (1 - OVERLAP_PAD))).any()
                gc = g[[clean(r) for r in g.itertuples()]]
                if not gc.empty:
                    g = gc
            if g.empty:
                return None
            b = g.iloc[0]
            parts = unit.split("_")
            return dict(recorder=parts[0] + "_" + parts[1], site=parts[2], t=pd.Timestamp(b.t).tz_convert(TZ), R=float(b.R), F_fix=float(b.F_fix),
                        F_lo=float(b.F_lo), F_hi=float(b.F_hi), depth=float(b.depth), Rc=float(b.chirp_rate) if "chirp_rate" in g and pd.notna(b.chirp_rate) else np.nan,
                        ref_lo=float(b.ref_lo) if "ref_lo" in g and pd.notna(b.ref_lo) else float(b.F_lo),
                        ref_hi=float(b.ref_hi) if "ref_hi" in g and pd.notna(b.ref_hi) else float(b.F_hi))
    return None


def analyse(w: dict):
    """Read the window, spectrogram, loudest 30 s for the band, the 1 s excerpt and the fold."""
    if not np.isfinite(w["R"]):            # chirp-only track (no syllable rate): fold at the chirp rate instead
        if np.isfinite(w.get("Rc", np.nan)) and w["Rc"] > 0:
            w["R"] = w["Rc"]; w["fold_at_chirp"] = True
        else:
            raise ValueError("no rate to fold at")
    sig, sr, t0, _ = W.read_window(w["recorder"], w["site"], w["t"], 300.0)
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2
    dt = 128 / sr
    r_lo = max(0, int(np.searchsorted(f, w["ref_lo"])) - 1); r_hi = min(len(f) - 1, int(np.searchsorted(f, w["ref_hi"])) + 1)
    try:
        cols = W.best_segment(P, r_lo, r_hi, dt, w["R"], FOLD_S)
    except ValueError:
        cols = slice(0, P.shape[1])
    try:
        R = W.refine_rate(P, r_lo, r_hi, cols, dt, w["R"])
    except ValueError:                 # the FFT grid of a short segment has no bin within 0.6 Hz of R
        R = w["R"]
    if not (0.97 * w["R"] <= R <= 1.03 * w["R"]):
        R = w["R"]
    # the 1 s with the most in-band modulated power inside the loudest 30 s
    seg = P[r_lo:r_hi + 1, cols]
    lev = 10 * np.log10(seg.sum(axis=0) + 1e-20)
    n1 = int(round(EXCERPT_S / dt))
    c0 = cols.start
    if seg.shape[1] > n1:
        try:
            s = W.locked_series(P[r_lo:r_hi + 1, cols], dt, R).mean(axis=0)   # 0.5 s chunks
            if len(s) >= 2:
                k = int(np.argmax(np.convolve(s, np.ones(2), "valid")))
                c0 = cols.start + int(k * round(W.SLOW_S / dt))
        except ValueError:          # a segment too short for the chunking: keep its start
            c0 = cols.start
    c0 = max(0, min(c0, P.shape[1] - n1))
    ex = P[:, c0:c0 + n1]
    quiet = np.percentile(10 * np.log10(P + 1e-20), 10, axis=1, keepdims=True)
    ex_db = 10 * np.log10(ex + 1e-20) - quiet
    # fold of band +/- PAD over the loudest 30 s
    p_lo = max(0, int(np.searchsorted(f, w["F_fix"] - PAD_HZ)) - 1); p_hi = min(len(f) - 1, int(np.searchsorted(f, w["F_fix"] + PAD_HZ)) + 1)
    W.PHASE_BINS = NPH
    Fm = W.fold(P, p_lo, p_hi, cols, dt, R)
    # dB above each row's QUIET level over the period (its 10th percentile across phase bins), not about
    # its mean: about the mean, the silent part of the period in the animal's rows sat 6-10 dB below and
    # clipped to black, a "shadow" between pulses that is only the complement of the pulse (MBE, 2026-10-04)
    Fdb = 10 * np.log10(Fm + 1e-20); Fdb -= np.percentile(Fdb, 10, axis=1, keepdims=True)
    fk = f[p_lo:p_hi + 1]
    band = (fk >= w["F_lo"] - 86) & (fk <= w["F_hi"] + 86)
    Fmod = np.clip(Fm[band] - Fm[band].mean(axis=1, keepdims=True), 0, None) if band.any() else np.clip(Fm - Fm.mean(axis=1, keepdims=True), 0, None)
    env = Fmod.sum(axis=0); env = env / (env.max() + 1e-20)
    sig_ex = sig[c0 * 128: c0 * 128 + n1 * 128 + 512]              # the excerpt's audio, for fine display spectrograms
    return dict(ex_db=ex_db, f=f, dt=dt, R=R, Fdb=Fdb, fk=fk, env=env, t_ex=t0 + pd.Timedelta(seconds=c0 * dt), sig_ex=sig_ex, sr=sr, quiet=quiet.ravel())


def draw_streak(args):
    sid, picks, S_row, cand, unit_roots, out_dir, mode = args
    order_txt = ("cold/slow end (left) to warm/fast end" if mode == "along"
                 else "the deepest long tracks within one narrow slice of along-line position, ordered by temperature")
    results = []
    for _, r in picks.iterrows():
        w = window_of(r.an, unit_roots)
        if w is None:
            results.append((r, None, "no window")); continue
        try:
            x = analyse(w)
            results.append((r, dict(w, **x), ""))
        except SystemExit as e:
            results.append((r, w, f"no audio: {str(e)[:40]}"))
        except Exception as e:  # noqa: BLE001
            results.append((r, w, f"{type(e).__name__}: {str(e)[:40]}"))
    ncol = 5; nrow = int(np.ceil(len(results) / ncol))
    head = f"streak {sid}: {S_row['F25'] / 1000:.2f} kHz @ {S_row['R25']:.0f} Hz at 25 °C, {int(S_row['n'])} singer-periods" + (f"; SINA candidate by position: {cand}" if cand else "")
    # 1. windows: full 0-22 kHz panel with, under it, a zoomed strip of the animal's band +/- ZOOM_HZ
    fig = plt.figure(figsize=(3.4 * ncol, 5.0 * nrow))
    gs = fig.add_gridspec(2 * nrow, ncol, height_ratios=[3, 1.1] * nrow, hspace=0.55, wspace=0.18)
    top_axes = [[fig.add_subplot(gs[2 * i, j]) for j in range(ncol)] for i in range(nrow)]
    zoom_axes = [[fig.add_subplot(gs[2 * i + 1, j], sharex=top_axes[i][j]) for j in range(ncol)] for i in range(nrow)]
    for k, (r, w, err) in enumerate(results):
        ax, az = top_axes[k // ncol][k % ncol], zoom_axes[k // ncol][k % ncol]
        if w is None or err:
            ax.text(0.5, 0.5, err, ha="center", va="center", fontsize=8, transform=ax.transAxes); ax.set_axis_off(); az.set_axis_off(); continue
        ext = [0, w["ex_db"].shape[1] * w["dt"], 0, FMAX / 1000]
        ax.imshow(w["ex_db"], origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=30, interpolation="nearest")
        ax.plot([0.012, 0.012], [w["F_lo"] / 1000, w["F_hi"] / 1000], color="cyan", lw=2.5, solid_capstyle="butt")
        ax.plot([0.0, 0.03], [w["F_fix"] / 1000] * 2, color="cyan", lw=0.8)
        rc = f" · chirp {w['Rc']:.1f} Hz" if np.isfinite(w.get("Rc", np.nan)) else ""
        ax.set_title(f"{w['t']:%Y-%m-%d %H:%M} {w['recorder'][-4:]} {w['site']}\nT {r['T']:.1f} °C · R {w['R']:.1f} Hz · F {w['F_fix']:.0f} Hz{rc}", fontsize=7.5, loc="left")
        ax.set_ylim(0, FMAX / 1000); ax.set_xlim(0, EXCERPT_S); ax.tick_params(labelsize=7, labelbottom=False)
        az.imshow(w["ex_db"], origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=30, interpolation="nearest")
        az.set_ylim((w["F_fix"] - ZOOM_HZ) / 1000, (w["F_fix"] + ZOOM_HZ) / 1000)
        az.axhline(w["F_lo"] / 1000, color="cyan", lw=0.5, ls=":"); az.axhline(w["F_hi"] / 1000, color="cyan", lw=0.5, ls=":")
        az.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel("kHz", fontsize=8); az.set_ylabel("kHz (band)", fontsize=7.5)
        if k // ncol == nrow - 1:
            az.set_xlabel("s", fontsize=8)
    for k in range(len(results), nrow * ncol):
        top_axes[k // ncol][k % ncol].set_axis_off(); zoom_axes[k // ncol][k % ncol].set_axis_off()
    fig.suptitle(head + f"\n1 s excerpts from the loudest 30 s of each singer's deepest window, {order_txt}; color = dB above the row's quiet level (0–30);\n"
                 f"upper: 0–22 kHz with the animal's band and fixed-window carrier in cyan; lower: the band ± {ZOOM_HZ / 1000:.0f} kHz (dotted = band edges)", fontsize=9)
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.99)
    p1 = os.path.join(out_dir, f"streak_{sid}_windows.png"); fig.savefig(p1, dpi=130); plt.close(fig)
    # 2. profiles
    periods = [1000 / w["R"] for (_, w, err) in results if w is not None and not err and "R" in w and np.isfinite(w["R"]) and w["R"] > 0]
    xmax = 2 * max(periods) if periods else 100.0
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, (r, w, err) in zip(axes.ravel(), results):
        if w is None or err or "Fdb" not in w:
            ax.text(0.5, 0.5, err or "no fold", ha="center", va="center", fontsize=8, transform=ax.transAxes); ax.set_axis_off(); continue
        per = 1000 / w["R"]
        img = np.concatenate([w["Fdb"], w["Fdb"]], axis=1)
        y0, y1 = (w["fk"][0] - w["F_fix"]) / 1000, (w["fk"][-1] - w["F_fix"]) / 1000
        ax.imshow(img, origin="lower", aspect="auto", extent=[0, 2 * per, y0, y1], cmap="magma", vmin=0, vmax=15, interpolation="nearest")
        env = np.concatenate([w["env"], w["env"]])
        xe = np.linspace(0, 2 * per, len(env), endpoint=False) + per / NPH / 2
        ax.plot(xe, y0 + 0.9 * (y1 - y0) * env * 0.35, color="cyan", lw=0.9)
        ax.axvline(per, color="w", lw=0.5, alpha=0.6)
        what = "chirp period" if w.get("fold_at_chirp") else "period"
        ax.set_title(f"{w['t']:%Y-%m-%d} {w['recorder'][-4:]} · T {r['T']:.1f} °C\n{what} {per:.1f} ms (R {w['R']:.2f} Hz) · F {w['F_fix']:.0f} Hz", fontsize=7.5, loc="left")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(results):]:
        ax.set_axis_off()
    axes[0, 0].set_xlim(0, xmax); axes[0, 0].set_ylim(-PAD_HZ / 1000, PAD_HZ / 1000)
    for ax in axes[:, 0]:
        ax.set_ylabel("kHz from fixed-window carrier", fontsize=7.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("ms (two periods)", fontsize=8)
    fig.suptitle(head + f"\nsyllable profiles: the band ± {PAD_HZ:.0f} Hz folded at the track's rate over the loudest 30 s ({NPH} phase bins), dB above each row's quiet level over the period (0…15), two periods in ms; cyan = modulated-power envelope", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = os.path.join(out_dir, f"streak_{sid}_profiles.png"); fig.savefig(p2, dpi=130); plt.close(fig)
    idx = [dict(streak=sid, an=r.an, along=r.along, T=r["T"], depth=r.depth, status=err or "ok",
                t_excerpt=str(w["t_ex"]) if (w and "t_ex" in w) else "", R=w["R"] if w and "R" in w else np.nan, F_fix=w["F_fix"] if w else np.nan) for (r, w, err) in results]
    return sid, p1, p2, idx


def _safe_draw(args):
    try:
        return draw_streak(args)
    except Exception as e:  # noqa: BLE001
        plt.close("all")
        return f"streak {args[0]}: FAILED {type(e).__name__}: {str(e)[:100]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--suffix", default="_v6")
    ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--streaks", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", choices=["along", "fixed"], default="along",
                    help="along: even steps along the line; fixed: the deepest long tracks within one narrow slice of position (one-thing test)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--min-n", type=int, default=15)
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[S.harmonic_of < 0] if "harmonic_of" in S.columns else S
    cands = {}
    mp = os.path.join(root, "streak_sina_matches.csv")
    if os.path.exists(mp):
        M = pd.read_csv(mp); cands = M[M["rank"] == 1].set_index("streak").species.to_dict()
    out_dir = os.path.join(root, "gallery" if args.mode == "along" else f"gallery_{args.mode}"); os.makedirs(out_dir, exist_ok=True)
    tagm = "" if args.mode == "along" else f"_{args.mode}"
    sids = args.streaks or [s for s in S.sort_values("n", ascending=False).index if S.loc[s, "n"] >= args.min_n]
    jobs = []
    for sid in sids:
        g = A[A.streak == sid]
        if len(g) < 3:
            continue
        c = C_LO if S.loc[sid, "F25"] < HF_SPLIT else C_HI
        picks = choose(g, c, min(args.n, len(g)), args.mode)
        if picks.empty:
            continue
        jobs.append((sid, picks, S.loc[sid].to_dict(), cands.get(sid, ""), args.unit_roots, out_dir, args.mode))
    print(f"{len(jobs)} streaks, {args.n} singers each", flush=True)
    rows, pages = [], []
    with Pool(args.jobs) as pool:
        for res in pool.imap(_safe_draw, jobs):
            if isinstance(res, str):
                print("  " + res, flush=True); continue
            sid, p1, p2, idx = res
            rows += idx; pages += [p1, p2]
            print(f"  streak {sid}: {sum(1 for r in idx if r['status'] == 'ok')} of {len(idx)} drawn", flush=True)
    ip = os.path.join(root, f"gallery{tagm}_index{args.suffix}.csv")
    new = pd.DataFrame(rows)
    if args.streaks and os.path.exists(ip):          # a subset run updates its streaks' rows and leaves the rest
        old = pd.read_csv(ip)
        new = pd.concat([old[~old.streak.isin(new.streak.unique())], new], ignore_index=True).sort_values(["streak", "along"])
    new.to_csv(ip, index=False, float_format="%.3f")
    if not args.streaks:                              # the combined PDF only from a full run
        with PdfPages(os.path.join(root, f"gallery{tagm}{args.suffix}.pdf")) as pdf:
            for p in pages:
                img = plt.imread(p)
                fig = plt.figure(figsize=(img.shape[1] / 130, img.shape[0] / 130))
                fig.figimage(img, resize=True); pdf.savefig(fig); plt.close(fig)
        print("wrote", out_dir, "and", os.path.join(root, f"gallery{tagm}{args.suffix}.pdf"))
    else:
        print("wrote", out_dir, "(subset; PDF not rebuilt)")


if __name__ == "__main__":
    main()
