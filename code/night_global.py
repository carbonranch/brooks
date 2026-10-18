"""Night linker as a GLOBAL fit: common rate drift removed, then the whole
night's window-animals joined into tracks by one optimal assignment instead
of window-by-window greedy chaining.

1. Drift. Singers cool together: their rates fall 10-25 % over a night
   (2024-08-24 NorthSide -11 %, 2024-09-07 EastCreek -24 %). The common
   log-rate drift delta(t) is read from the data: for each pair of
   consecutive windows, every animal is matched to its nearest neighbour
   (within 150 Hz and 3 % rate) in the next window and the median log-rate
   step is taken; steps are integrated and smoothed. A singer's detrended
   rate y = log R - delta(t) is then nearly constant all night.
2. Candidate links. Detection i (window w) may continue to detection j in
   a later window w' <= w + MAX_GAP if their detrended rates agree within
   R_TOL (widened by GAP_TOL per skipped window) and their fundamental
   bands overlap within F_PAD. Link cost = 1 + (dy/SIGMA_R)^2 +
   (dF_c/SIGMA_F)^2 + GAP_PEN*(w'-w-1). Chirp-only detections (no syllable
   rate) link to each other on chirp rate within CHIRP_TOL.
3. Global solution. Tracks are paths in this DAG; the minimum-cost path
   cover is a minimum-weight bipartite matching between every detection's
   'out' port and every detection's 'in' port, with a dummy 'end' for each
   out and a dummy 'start' for each in at cost BREAK/2 each -- so a track
   break costs BREAK and a link is taken only if it is cheaper than
   breaking. Solved exactly with scipy's sparse LAPJV; every detection is on
   exactly one path, and a doubtful join in window 30 is weighed against
   everything windows 31-145 say.

0. Before any of this, postfilter.apply flags harmonics reported as their own
   animal (2x/3x carrier, same rate, same window) and detections on a
   recorder-month's fixed-rate line (fixed_rate_lines.csv in the run root, from
   fixed_rate_lines.py); only keep rows are linked, the rest are written back
   with track -1. --no-filter turns this off.

Usage:
    python analysis/callers/night_global.py <night_dir> [--tag _night] [--band F_HZ R_HZ] [--no-filter]
Writes <night_dir>/night<tag>_global_tracks.csv and _global_window_animals.csv
(same columns as night_animals.py's _tracks.csv where they exist: track,
n_windows, t0, t1, F_ref, R, R_sd, depth, span_h, fill, plus R_detr_sd_pct,
Rc, delta of the night as a separate _drift.csv).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices from chirp-only tracks
import pandas as pd
import scipy.sparse
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from night_animals import bands, load, reference_carrier, TZ  # noqa: E402
import postfilter  # noqa: E402

MAX_GAP = 6          # windows (30 min) a track may skip
R_TOL = 0.020        # detrended log-rate tolerance for a link (greedy-linked steps: 90 % < 1.4 %, 95 % < 1.9 %)
GAP_TOL = 0.004      # extra tolerance per skipped window
F_PAD = 150.0        # Hz slack on the overlap of the two fundamental bands
SIGMA_R = 0.010      # log-rate scale of the link cost
RATE_CAP = 6.0       # the rate term is capped: steps have heavy tails, a 2 % step must still be linkable
HARM_PEN = 2.0       # cost when the two detections overlap only through a non-fundamental band (harmonic flip); used only if ALLOW_HARM
ALLOW_HARM = False   # off: a 7 kHz animal whose attached 12 kHz harmonic overlapped a lone 12 kHz detection was linking to it (MBE: "not plausible")
SCALE_PEN = 5.0      # cost when they overlap only after scaling one band by 2 or 3 (fundamental in one window, harmonic in the next)
ALLOW_SCALED = False # off: a track stays in one band; a window where only the harmonic was found starts its own short track
                     # (on, the strip showed tracks leaping between a fundamental and its 2x band, F_sd 1-5 kHz within a track)
SIGMA_FC = 60.0      # Hz: carrier-centroid continuity between fundamentals (a neighbour 55 Hz away with the same rate is a different animal)
SIGMA_FC_REL = 0.03  # the carrier scale is max(SIGMA_FC, SIGMA_FC_REL * carrier): the band centroid of a real animal
                     # wanders ~2.5 % of its carrier between windows (month run, greedy tracks >= 3 h: F_c sd median
                     # 214 Hz, 55 Hz below 3 kHz to 158 Hz above 12 kHz), so a fixed 60 Hz scale shredded real tracks.
                     # Four test nights, 0 -> 0.03: tracks >= 3 h +50 %, detections in them 0.10-0.20 -> 0.15-0.31,
                     # F_c sd within them 56-86 -> 125-146 Hz (greedy: 0.8-2.2 kHz), detrended rate sd unchanged ~1.1-1.4 %.
                     # 0.05 gave a little more but F_sd 170-196 Hz and rate sd creeping up. 7 -> 12 kHz hops stay
                     # impossible: fundamentals must overlap (F_PAD) regardless of this scale.
FC_CAP = 12.0        # uncapped in practice (> BREAK): a carrier step beyond ~180 Hz is a break; a window with a poor centroid is skipped by a gap link
GAP_PEN = 0.6        # cost per skipped window
BREAK = 10.0         # cost of ending one track and starting another
CHIRP_TOL = 0.06
MIN_WINDOWS = 3


def common_drift(A: pd.DataFrame, windows: np.ndarray) -> np.ndarray:
    """Cumulative median log-rate step between consecutive windows (nearest
    neighbour within 150 Hz and 3 %), smoothed with a 3-window median."""
    lr = np.log(A.R.values.astype(float))
    steps = [0.0]
    for a, b in zip(windows[:-1], windows[1:]):
        ia = np.flatnonzero((A.t == a).values & np.isfinite(lr)); ib = np.flatnonzero((A.t == b).values & np.isfinite(lr))
        ds = []
        for i in ia:
            m = ib[(np.abs(A.F_c.values[ib] - A.F_c.values[i]) <= 150) & (np.abs(lr[ib] - lr[i]) < 0.03)]
            if len(m):
                j = m[np.argmin(np.abs(lr[m] - lr[i]) + np.abs(A.F_c.values[m] - A.F_c.values[i]) / 5000)]
                ds.append(lr[j] - lr[i])
        steps.append(float(np.median(ds)) if len(ds) >= 3 else 0.0)
    d = np.cumsum(steps)
    return pd.Series(d).rolling(3, center=True, min_periods=1).median().values


def global_link(A: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    A = A.copy().reset_index(drop=True)
    windows = np.array(sorted(A.t.unique()))
    widx = {t: k for k, t in enumerate(windows)}
    w = np.array([widx[t] for t in A.t])
    delta = common_drift(A, windows)
    has_r = A.R.notna().values
    rc = A.chirp_rate.values.astype(float) if "chirp_rate" in A.columns else np.full(len(A), np.nan)
    y = np.where(has_r, np.log(np.where(has_r, A.R.values.astype(float), 1.0)) - delta[w], np.nan)
    yc = np.where(np.isfinite(rc), np.log(np.where(np.isfinite(rc), rc, 1.0)), np.nan)
    bl = [bands(c) or [(A.F_lo.iloc[i], A.F_hi.iloc[i])] for i, c in enumerate(A.components)]
    fund = [b[0] for b in bl]
    lo = np.array([f[0] for f in fund]); hi = np.array([f[1] for f in fund])
    n = len(A)

    fc = A.F_c.values.astype(float)

    def band_cost(i, j):
        """Fundamentals overlap: the carrier-centroid step in units of SIGMA_FC, squared and
        capped (0 for the same animal, ~1 for a 60 Hz step, FC_CAP beyond ~120 Hz). HARM_PEN if
        only some other component band overlaps (one window found the harmonic, the other the
        fundamental), SCALE_PEN if only after scaling a band by 2 or 3, None if no overlap."""
        if lo[i] <= hi[j] + F_PAD and lo[j] <= hi[i] + F_PAD:
            s = max(SIGMA_FC, SIGMA_FC_REL * min(fc[i], fc[j]))
            return min(((fc[j] - fc[i]) / s) ** 2, FC_CAP)
        if ALLOW_HARM:
            for a0, a1 in bl[i]:
                for b0, b1 in bl[j]:
                    if a0 <= b1 + F_PAD and b0 <= a1 + F_PAD:
                        return HARM_PEN
        # fundamental in one window, its 2nd/3rd carrier harmonic alone in the other
        if ALLOW_SCALED:
            for m in (2.0, 3.0, 0.5, 1 / 3):
                if lo[i] * m <= hi[j] + F_PAD and lo[j] <= hi[i] * m + F_PAD:
                    return SCALE_PEN
        return None
    # candidate links
    rows, cols, costs = [], [], []
    order = np.argsort(w, kind="stable")
    by_w = {}
    for i in order:
        by_w.setdefault(w[i], []).append(i)
    for i in range(n):
        for g in range(1, MAX_GAP + 1):
            for j in by_w.get(w[i] + g, []):
                bc = band_cost(i, j)
                if bc is None:
                    continue
                if has_r[i] and has_r[j]:
                    dy = abs(y[j] - y[i])
                    if dy >= R_TOL + GAP_TOL * (g - 1):
                        continue
                    c = 1.0 + min((dy / SIGMA_R) ** 2, RATE_CAP) + bc + GAP_PEN * (g - 1)
                elif (not has_r[i]) and (not has_r[j]) and np.isfinite(yc[i]) and np.isfinite(yc[j]):
                    dc = abs(yc[j] - yc[i])
                    if dc >= CHIRP_TOL:
                        continue
                    c = 1.0 + (dc / (CHIRP_TOL / 2)) ** 2 + bc + GAP_PEN * (g - 1)
                else:
                    # syllable rate on one side only: link on chirp rate if both have one
                    if np.isfinite(yc[i]) and np.isfinite(yc[j]) and abs(yc[j] - yc[i]) < CHIRP_TOL:
                        c = 1.0 + (abs(yc[j] - yc[i]) / (CHIRP_TOL / 2)) ** 2 + bc + GAP_PEN * (g - 1) + 2.0
                    else:
                        continue
                if c < BREAK:
                    rows.append(i); cols.append(j); costs.append(c)
    # Minimum-cost path cover as a bipartite matching, solved per connected component of the
    # candidate graph (a band's detections; components are small, so a dense assignment is fast).
    # Rows = out ports (m) + dummy starts (m); cols = in ports (m) + dummy ends (m).
    # out_i -> in_j: link cost; out_i -> end_i: BREAK/2; start_j -> in_j: BREAK/2; start_k -> end_k: tiny;
    # start_j -> end_i (tiny) for every candidate (i, j), so a taken link leaves its freed dummies a
    # partner and any chain of links is a full matching.
    from scipy.optimize import linear_sum_assignment
    rows = np.array(rows, int); cols = np.array(cols, int); costs = np.array(costs, float)
    nxt = np.full(n, -1)
    if len(rows):
        und = scipy.sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        ncomp, comp = scipy.sparse.csgraph.connected_components(und, directed=False)
        for c_id in range(ncomp):
            nodes = np.flatnonzero(comp == c_id)
            m = len(nodes)
            if m < 2:
                continue
            pos = {int(v): k for k, v in enumerate(nodes)}
            sel = np.flatnonzero(comp[rows] == c_id)
            BIG = 1e6
            Cm = np.full((2 * m, 2 * m), BIG)
            r_loc = np.array([pos[int(r)] for r in rows[sel]]); c_loc = np.array([pos[int(c)] for c in cols[sel]])
            Cm[r_loc, c_loc] = costs[sel]                       # out_i -> in_j
            Cm[np.arange(m), m + np.arange(m)] = BREAK / 2       # out_i -> end_i
            Cm[m + np.arange(m), np.arange(m)] = BREAK / 2       # start_j -> in_j
            Cm[m + np.arange(m), m + np.arange(m)] = 1e-3        # start_k -> end_k
            Cm[m + c_loc, m + r_loc] = 1e-3                      # start_j -> end_i for each candidate (i, j)
            ri, ci = linear_sum_assignment(Cm)
            for r, c in zip(ri, ci):
                if r < m and c < m and Cm[r, c] < BIG:
                    nxt[nodes[r]] = nodes[c]
    print(f"[global_link] {n} detections, {len(rows)} candidate links, {int((nxt >= 0).sum())} links taken, "
          f"{int(has_r.sum())} with a syllable rate; drift end {100 * (np.exp(delta[-1]) - 1):+.1f} %", flush=True)
    has_prev = np.zeros(n, bool); has_prev[nxt[nxt >= 0]] = True
    track = np.full(n, -1); tid = 0
    for i in range(n):
        if has_prev[i]:
            continue
        j = i
        while j >= 0:
            track[j] = tid; j = nxt[j]
        tid += 1
    A["track"] = track
    A["y_detr"] = y
    rows_t = []
    for c, g in A.groupby("track"):
        nw = g.t.nunique()
        if nw < MIN_WINDOWS:
            continue
        span_h = (g.t.max() - g.t.min()).total_seconds() / 3600
        rows_t.append(dict(track=c, n_windows=nw, t0=g.t.min(), t1=g.t.max(), span_h=span_h, fill=nw / (round(span_h * 12) + 1),
                           F_ref=g.F_c.median(), F_fix=g.F_c.median(), F_sd=g.F_c.std(), R=g.R.median(), R_sd=g.R.std(),
                           R_detr_sd_pct=100 * float(np.std(g.y_detr.dropna())) if g.y_detr.notna().any() else np.nan,
                           Rc=g.chirp_rate.median() if "chirp_rate" in g else np.nan,
                           depth=g.depth.median(), n_comp_med=g.n_components.median()))
    cols_t = ["track", "n_windows", "t0", "t1", "span_h", "fill", "F_ref", "F_fix", "F_sd", "R", "R_sd", "R_detr_sd_pct", "Rc", "depth", "n_comp_med"]
    T = pd.DataFrame(rows_t, columns=cols_t)
    if len(T):
        T = T.sort_values("t0").reset_index(drop=True)
    return A, T, pd.Series(delta, index=windows)


def apply_sylv2_sidecar(night_dir: str, tag: str, A: pd.DataFrame, P: pd.DataFrame):
    """The time-domain chirp re-pass (chirp_repass.py, METHOD 19 AG) applied before linking, when the unit holds
    night<tag>_global_window_animals_sylv2.csv: every chirp-fundamental animal takes the re-read syllable rate
    (NaN where the reader rejected it: chirp-only, not tracked), and a twin -- a separate animal at that rate whose
    band overlaps, the chirper's own syllable peak detected again -- is folded into the chirper: its parts become the
    chirper's components, its row is dropped, the chirper's band widened. Enabled with BROOKS_SYLV2_LINK=1 (the v3 tree)."""
    p = os.path.join(night_dir, f"night{tag}_global_window_animals_sylv2.csv")
    if not os.path.exists(p):
        return A, P
    S = pd.read_csv(p)
    if S.empty:
        return A, P
    S["t"] = pd.to_datetime(S.t, utc=True).dt.tz_convert(TZ)
    A = A.copy(); P = P.copy()
    key = pd.MultiIndex.from_frame(A[["t", "animal"]])
    A.index = key
    for r in S.itertuples():
        k = (r.t, int(r.animal))
        if k not in A.index:
            continue
        A.loc[k, "R"] = float(r.R_new) if pd.notna(r.R_new) else np.nan
        if r.merged_twin >= 0:
            kt = (r.t, int(r.merged_twin))
            if kt in A.index:
                tw = A.loc[kt]
                A.loc[k, "F_lo"] = float(min(A.loc[k, "F_lo"], tw.F_lo)); A.loc[k, "F_hi"] = float(max(A.loc[k, "F_hi"], tw.F_hi))
                A.loc[k, "n_components"] = int(A.loc[k, "n_components"]) + int(tw.n_components)
                A.loc[k, "components"] = f"{A.loc[k, 'components']};{tw.components}"
                A = A.drop(index=kt)
                m = (P.t == r.t) & (P.animal == int(r.merged_twin))
                P.loc[m, "animal"] = int(r.animal)
    A = A.reset_index(drop=True).sort_values(["t", "depth"], ascending=[True, False])
    # the chirp-rate re-pass (chirp_rate_repass.py, METHOD 19 AJ): the iterated sub-harmonic check's chirp rate
    pc = os.path.join(night_dir, f"night{tag}_global_window_animals_crv2.csv")
    if os.path.exists(pc):
        C = pd.read_csv(pc)
        if not C.empty:
            C = C[(C.m > 1) & C.C_new.notna()]
            C["t"] = pd.to_datetime(C.t, utc=True).dt.tz_convert(TZ)
            key = pd.MultiIndex.from_frame(A[["t", "animal"]]); A.index = key
            for r in C.itertuples():
                k = (r.t, int(r.animal))
                if k in A.index:
                    A.loc[k, "chirp_rate"] = float(r.C_new)
            A = A.reset_index(drop=True)
    return A, P


def link_dir(night_dir: str, tag: str = "_night", quiet: bool = False, lines: pd.DataFrame | None = None, filt: bool = True):
    """Link one night directory and write night<tag>_global_{tracks,window_animals,drift}.csv.
    With filt, postfilter flags are added first (harmonic-as-own, fixed-rate line) and only
    keep rows are linked; the rest are written back with track -1. lines: the run's
    fixed_rate_lines.csv (postfilter.load_lines), else read from the parent of night_dir."""
    A, P = load(night_dir, tag)
    if A.empty:
        return None
    if os.environ.get("BROOKS_SYLV2_LINK", "0") == "1":
        A, P = apply_sylv2_sidecar(night_dir, tag, A, P)
    dropped = None
    if filt:
        if lines is None:
            lines = postfilter.load_lines(os.path.dirname(os.path.normpath(night_dir)))
        A = postfilter.apply(A, P, night_dir=night_dir, lines=lines)
        dropped = A[~A.keep]
        A = A[A.keep]
        if A.empty:
            return None
    if quiet:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            Wd, T, delta = global_link(A)
    else:
        Wd, T, delta = global_link(A)
    # fixed-window carrier per window (night_animals.reference_carrier): each track's reference
    # band is the component band present in most of its windows, F_fix the power-weighted
    # centroid over a fixed window about it -- stable against which rows passed threshold.
    # The track's F_fix is the median of these (was the median band centroid).
    linked = Wd.track >= 0
    if linked.any():
        Wr = reference_carrier(Wd[linked], P)
        for c in ("F_ref", "F_fix", "ref_lo", "ref_hi"):
            Wd[c] = np.nan
            Wd.loc[Wr.index, c] = Wr[c]
        if len(T):
            ff = Wd[linked].groupby("track").F_fix.agg(["median", "std"])
            T["F_fix"] = T.track.map(ff["median"]).fillna(T.F_fix)
            T["F_fix_sd"] = T.track.map(ff["std"])
    if dropped is not None and len(dropped):
        dropped = dropped.assign(track=-1)
        Wd = pd.concat([Wd, dropped], ignore_index=True).sort_values(["t", "depth"], ascending=[True, False])
    stem = os.path.join(night_dir, f"night{tag}_global")
    T.to_csv(stem + "_tracks.csv", index=False, float_format="%.3f")
    Wd.to_csv(stem + "_window_animals.csv", index=False, float_format="%.3f")
    delta.rename("delta_log_rate").to_csv(stem + "_drift.csv")
    return Wd, T, delta


def _link_one(args):
    d, tag, lines, filt = args
    try:
        r = link_dir(d, tag, quiet=True, lines=lines, filt=filt)
        return os.path.basename(d), (len(r[1]) if r else 0), ""
    except Exception as e:  # noqa: BLE001
        return os.path.basename(d), 0, str(e)[:120]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("night_dir", help="a night directory, or with --all a run root holding BARLT_* night directories")
    ap.add_argument("--tag", default="_night")
    ap.add_argument("--band", nargs=2, type=float, default=None, metavar=("F_HZ", "R_HZ"))
    ap.add_argument("--all", action="store_true", help="link every BARLT_* night under night_dir (skipping those already done)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-filter", action="store_true", help="skip postfilter (harmonic-as-own and fixed-rate-line flags)")
    args = ap.parse_args()
    filt = not args.no_filter
    if args.all:
        import glob
        from multiprocessing import Pool
        dirs = sorted(d for d in glob.glob(os.path.join(args.night_dir, "BARLT_*")) if os.path.exists(os.path.join(d, f"night{args.tag}_tracks.csv")))
        if not args.redo:
            dirs = [d for d in dirs if not os.path.exists(os.path.join(d, f"night{args.tag}_global_tracks.csv"))]
        lines = postfilter.load_lines(args.night_dir) if filt else None
        print(f"{len(dirs)} nights to link, {args.jobs} jobs, filter {'on' if filt else 'off'}"
              f"{'' if lines is None else f' ({len(lines)} fixed-rate lines)'}", flush=True)
        n_fail = 0
        with Pool(args.jobs) as pool:
            for k, (name, n, err) in enumerate(pool.imap_unordered(_link_one, [(d, args.tag, lines, filt) for d in dirs]), 1):
                if err:
                    n_fail += 1; print(f"[{k}/{len(dirs)}] {name}: FAILED {err}", flush=True)
                elif k % 50 == 0 or k == len(dirs):
                    print(f"[{k}/{len(dirs)}] {name}: {n} tracks", flush=True)
        print(f"done, {n_fail} failed", flush=True)
        return
    r = link_dir(args.night_dir, args.tag, filt=filt)
    if r is None:
        print("no window-animals"); return
    W, T, delta = r
    pd.set_option("display.width", 220)
    if filt and "keep" in W.columns:
        print(f"postfilter: {int((W.harm_of >= 0).sum())} harmonic-as-own, {int((W.sub_of >= 0).sum())} shadows, {int(W.line.sum())} on a fixed-rate line, {int(W.frag.sum())} frag (kept); "
              f"{int(W.keep.sum())} of {len(W)} linked")
        W = W[W.keep]
    print(f"{len(W)} window-animals -> {len(T)} tracks (>= {MIN_WINDOWS} windows); >= 1 h {int((T.span_h >= 1).sum())}, >= 3 h {int((T.span_h >= 3).sum())}, >= 6 h {int((T.span_h >= 6).sum())}; "
          f"drift over the night {100 * (np.exp(delta.iloc[-1]) - 1):+.1f} %, min {100 * (np.exp(delta.min()) - 1):+.1f} %")
    if args.band:
        F, R = args.band
        b = T[((T.F_ref - F).abs() <= 300) & (T.span_h >= 1)]
        q = b.R / R
        b = b[np.min(np.abs(q.values[:, None] / np.array([1, 2, 3, 0.5, 1 / 3])[None, :] - 1), axis=1) < 0.12]
        print(b[["t0", "t1", "span_h", "n_windows", "fill", "F_ref", "F_sd", "R", "R_sd", "R_detr_sd_pct", "depth", "n_comp_med"]].to_string(index=False))


if __name__ == "__main__":
    main()
