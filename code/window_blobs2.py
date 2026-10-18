"""One window: the frequency x AM map, the blobs in it, and the animals behind them.

VERSION 2 (2026-09-24), developed against MBE's 23 hand-annotated windows
(analysis/callers/runs/annotation, scored by annotation_scorecard.py) while
the corpus run still imports window_blobs.py. Differences from v1:
  1. the modulation map starts at AM_LO = 2 Hz, with a baseline that is a
     running median over LOG rate (the 1/f slope below 8 Hz made a linear
     15 Hz median useless there); slow chirpers and pulse trains appear at
     their true rate instead of at a harmonic;
  2. a low-rate peak (< BOUT_MAX_HZ) whose rows are shared with a faster
     peak is that singer's BOUT rhythm, recorded on the animal, not a singer;
  3. per peak, the rate is refined and the fold made over the singer's ON
     PERIODS when it sings less than ON_FRAC_CONT of the window: each run of
     song is folded with its own phase and the runs are aligned on their
     envelopes before summing (a chirper's phase resets at every chirp, so a
     single fold over 30 s smeared it);
  4. after refinement the fold is also tried at R/2 and R/3; if either is
     markedly deeper the rate is that sub-multiple (the map's peak was a
     harmonic of the rhythm);
  5. the structural harmonic fallback in link_parts now requires the lone
     part's on/off series to correlate with the animal's root (r > R_MIN):
     a chirper sitting at a harmonic position of a trill is no longer
     absorbed into it.
Tried and left off: a block-wise (30 s) modulation map for sporadic singers
(am_map_blocks; BLK_THRESH_DB = 0 disables it) -- no gain on the annotated
windows; and the chirp channel on top of v2 (+21 animals, no new coverage:
the 2 Hz floor with the wider baseline already covers its ground).
Scored against the 23 annotated windows (annotation_scorecard.py): singer
boxes covered 91 -> 104 of 112, hand rates agreeing 19 -> 21 of 23, misses
23 -> 9, non-insect boxes with a detection 4 -> 6 (one 99.6 Hz peak under an
owl and coyotes), harmonic boxes reported as their own singer 7 -> 10.

A stridulating insect repeats a short unit at rate R on a carrier band F. In a
map of carrier frequency against amplitude-modulation rate it is a compact blob,
and its harmonics -- carrier n*F and modulation m*R -- are further blobs. Two
animals at the same rate and neighbouring frequencies land in ONE blob. Neither
the lattice nor the merger can be resolved in (F, R) alone; both are resolved
by asking whether two pieces of the map switch on and off TOGETHER.

    blobs  --split by phase and on/off-->  parts  --link by co-variation-->  animals

The null for "co-varies" is measured in the same window from pairs of parts
whose rates are not in a small-integer ratio, which cannot be one animal.

Usage:
    python analysis/callers/window_blobs.py --recorder BARLT_00008662 \
        --site House --time "2024-09-12 20:40"
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import soundfile as sf

from chirp_channel import syllable_rate_in_rows  # v2: within-chirp syllable rate of a lone slow peak

AUDIO_ROOT = "/Volumes/8M2A/Data/Carbonranch/Ferguson/Audio"
AUDIO_ROOTS = [AUDIO_ROOT, "/Volumes/8M2C/Data/Carbonranch/Ferguson/Audio", "/Volumes/8M2D/Data/Carbonranch/Ferguson/Audio"]
# other archives (e.g. A2O under /Volumes/8M2C/Data/A2O/<Region>/Audio) and
# their local zone: BROOKS_AUDIO_ROOTS=path1:path2 and BROOKS_TZ=Australia/Adelaide
AUDIO_ROOTS = [p for p in os.environ.get("BROOKS_AUDIO_ROOTS", "").split(":") if p] + AUDIO_ROOTS
TZ = os.environ.get("BROOKS_TZ", "America/Chicago")
# v2 chirper handling (2026-10-08, METHOD 19 AF/AG): the concentration-tested syllable-rate reader and the merge of a
# chirp peak with its own syllable peak detected separately. OFF by default: the corpus run and the frozen figures
# use the first reader. Set BROOKS_SYL_V2=1 to turn it on (read at call time, so a script can toggle it).
SYL_V2 = os.environ.get("BROOKS_SYL_V2", "0") == "1"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NPERSEG, HOP = 512, 128          # 86 Hz bins, 11.6 ms window, 2.9 ms hop
AM_LO, AM_HI = 2.0, 100.0        # v2: down to 2 Hz with a log-rate baseline (v1: 8 Hz, linear 15 Hz median)
AM_LO = float(os.environ.get("BROOKS_AM_LO", AM_LO))
BASE_OCT = 1.4                   # v2: width of the running median over log rate, in octaves (0.7 and 1.0 cut the
                                 # broad, jittery peaks of 3-6 Hz pulse trains to <6 dB; 1.4 and 2.0 score the same)
BOUT_MAX_HZ = 8.0                # v2: a peak below this that shares rows with a faster peak may be that singer's bout rhythm
BOUT_RATIO_MAX = 0.4             # v2: ...if the fast modulation's envelope, folded at the slow rate, drops below this fraction of its peak
ON_FRAC_CONT = 0.75              # v2: singers on more than this fraction of the window are folded over the loudest 30 s as before
ON_MIN_S = 1.0                   # v2: a run of song shorter than this is not folded on its own
SUBHARM_GAIN = 1.25              # v2: the fold at R/m must be this much deeper (mean depth) to replace R
SUBHARM_ITER = 3                 # the sub-harmonic check is repeated up to this many times (m up to 27)
BLOCK_S = 30.0                   # v2: block length for the block-wise modulation map (sporadic singers)
BLK_THRESH_DB = 0.0              # v2: peak threshold in the block-max map; 0 = off. At 9 dB it added ~60 animals over the
                                 # 23 annotated windows, 3 more false positives (a train) and no new covered singer
FMIN_HZ = 300.0                  # carrier rows below this are not analysed (wind, rumble, hum)
WELCH_NPER = 2048                # 0.17 Hz AM resolution
BASE_HZ = 15.0                   # width of the per-row running median that sets the baseline
THRESH_DB = 6.0
MIN_ROWS = 2
FOLD_S = 30.0                    # rate drifts ~0.01 Hz over 90 s, enough to smear a longer fold
PHASE_BINS = 24
PAD_HZ = 300.0
MAX_SHOW = 12
R_TOL = 0.015                    # one animal's refined rate scatters ~0.1 Hz (0.5%) between windows
AM_GAP_BINS = 2                  # rows whose AM peaks differ by more than this are different blobs
PROF_CORR_MIN = 0.3              # adjacent rows of one animal have near-identical fold profiles
SLOW_S = 0.5                     # chunk for the on/off pattern (10-30 periods)
HP_S = 10.0                      # running median removed before correlating: slow common drift out
DEPTH_MIN = 0.15                 # a row below this (relative to its own power) does not seed a part
AMP_FLOOR_DB = 15.0              # a part extends into rows whose absolute in-phase amplitude is within this of its peak
DEPTH_EXT_MIN = 0.02             # ... provided the row is modulated at all
CV_MIN = 0.4                     # sd/mean of the raw on/off series; below this a row is continuous
ROW_CORR_MIN = 0.2               # a patterned row must follow its part's pattern at least this well
R_MIN, Z_MIN = 0.2, 4.5          # to link two parts: correlation and its shift-null z
N_SHIFTS = 60
MAX_HARM = 6
N_PERIODS = 5


def find_file(recorder: str, site: str, target_utc: pd.Timestamp):
    rx = re.compile(r"S(\d{8}T\d{6}\.\d+)\+0000_E(\d{8}T\d{6}\.\d+)\+0000")
    # the site directory is spelled NorthSide in one collection and Northside
    # in another, so match it case-insensitively; the audio spans three volumes
    paths = []
    for root in AUDIO_ROOTS:
        paths += glob.glob(os.path.join(root, "*", "*", f"{recorder}_*", recorder, "*", "S*.wav"))
    for p in sorted(paths):
        if p.split(os.sep)[-5].lower() != site.lower():
            continue
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        s = pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S.%f"), tz="UTC")
        e = pd.Timestamp(datetime.strptime(m.group(2), "%Y%m%dT%H%M%S.%f"), tz="UTC")
        if s <= target_utc < e:
            return p, s
    raise SystemExit(f"no file for {recorder}@{site} covering {target_utc}")


def read_window(recorder: str, site: str, target_local: pd.Timestamp, dur_s: float):
    target_utc = target_local.tz_convert("UTC")
    path, file_start = find_file(recorder, site, target_utc)
    info = sf.info(path)
    off = (target_utc - file_start).total_seconds() - dur_s / 2
    off = max(0.0, min(off, info.duration - dur_s))
    sig, sr = sf.read(path, start=int(off * info.samplerate),
                      frames=int(dur_s * info.samplerate), dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    return sig, sr, file_start + pd.Timedelta(seconds=off), path


def am_map(P: np.ndarray, dt: float):
    """Per carrier row: the modulation spectrum of its dB level, minus the row's
    own running-median baseline. A merely noisy row has a flat spectrum and
    scores zero everywhere; a modulated row scores at R and its harmonics."""
    L = 10 * np.log10(P + 1e-20)
    L -= L.mean(axis=1, keepdims=True)
    fr, S = scipy.signal.welch(L, fs=1 / dt, nperseg=WELCH_NPER, axis=1)
    return _excess(fr, S)


def _excess(fr: np.ndarray, S: np.ndarray):
    """Welch spectrum (per row) -> dB excess over the row's running median in
    LOG rate, cut to [AM_LO, AM_HI]."""
    # the baseline is computed over a wider range than is kept, so its running
    # median is not padded with edge values at 2 Hz and 100 Hz
    wide = (fr >= AM_LO / 1.6) & (fr <= min(AM_HI * 1.6, fr[-1]))
    fr, Sdb = fr[wide], 10 * np.log10(S[:, wide] + 1e-30)
    # v2 baseline: running median over LOG rate. Resample each row onto a log
    # grid, median-filter over BASE_OCT octaves, interpolate back. Below 8 Hz
    # the spectrum is a steep 1/f slope; a linear 15 Hz median there is wider
    # than the whole range and cannot follow it.
    lg = np.log2(fr)
    ng = int(np.ceil((lg[-1] - lg[0]) / (lg[1] - lg[0]) * 0.5))            # dense enough at the low end
    ng = max(ng, 400)
    grid = np.linspace(lg[0], lg[-1], ng)
    k = int(round(BASE_OCT / (grid[1] - grid[0]))) | 1
    base = np.empty_like(Sdb)
    for i in range(Sdb.shape[0]):
        on_grid = np.interp(grid, lg, Sdb[i])
        med = scipy.ndimage.median_filter(on_grid, size=k, mode="nearest")
        base[i] = np.interp(lg, grid, med)
    sel = (fr >= AM_LO) & (fr <= AM_HI)
    return fr[sel], (Sdb - base)[:, sel]


def am_map_blocks(P: np.ndarray, dt: float, block_s: float = None):
    """v2: the modulation map of each BLOCK_S block of the window, and the
    cell-wise maximum over blocks. A singer active for 8 s of 300 s is 1/40
    of the whole-window spectrum and never reaches threshold there; in its
    own 30 s block it is 1/4. The maximum over ~10 noisier estimates sits
    ~2 dB above the whole-window noise, hence the higher BLK_THRESH_DB."""
    block_s = block_s or BLOCK_S
    L = 10 * np.log10(P + 1e-20)
    n = int(round(block_s / dt))
    nb = max(1, P.shape[1] // n)
    ex_max, fr = None, None
    for b in range(nb):
        Lb = L[:, b * n:(b + 1) * n]
        Lb = Lb - Lb.mean(axis=1, keepdims=True)
        fr_b, S = scipy.signal.welch(Lb, fs=1 / dt, nperseg=WELCH_NPER, axis=1)
        fr, ex = _excess(fr_b, S)
        ex_max = ex if ex_max is None else np.maximum(ex_max, ex)
    return fr, ex_max


def find_blobs(excess: np.ndarray, f: np.ndarray, fr: np.ndarray, thresh: float = None) -> pd.DataFrame:
    """8-connected regions above threshold, then each region is divided along
    the AM axis: rows are grouped by the AM bin of their own peak, split where
    consecutive peak bins are more than AM_GAP_BINS apart. Two animals at 20.2
    and 21.0 Hz in neighbouring bands otherwise become one region carrying
    one rate label, and the fold at that rate smears the other animal."""
    thresh = THRESH_DB if thresh is None else thresh
    lab, n = scipy.ndimage.label(excess > thresh, structure=np.ones((3, 3)))
    rows = []
    for i in range(1, n + 1):
        r, c = np.nonzero(lab == i)
        if len(np.unique(r)) < MIN_ROWS:
            continue
        ur = np.unique(r)
        peak_col = np.array([c[r == rr][np.argmax(excess[rr, c[r == rr]])] for rr in ur])
        order = np.argsort(peak_col)
        groups, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if peak_col[b] - peak_col[a] > AM_GAP_BINS:
                groups.append(cur)
                cur = []
            cur.append(b)
        groups.append(cur)
        for g in groups:
            sel_rows = ur[g]
            if len(sel_rows) < MIN_ROWS:
                continue
            m = np.isin(r, sel_rows)
            rr, cc = r[m], c[m]
            w = excess[rr, cc]
            k = np.argmax(w)
            rows.append(dict(
                f_c=float(np.average(f[rr], weights=w)), f_lo=float(f[rr.min()]), f_hi=float(f[rr.max()]),
                R=float(fr[cc[k]]), am_lo=float(fr[cc.min()]), am_hi=float(fr[cc.max()]),
                peak_db=float(w[k]), n_cells=len(rr), n_rows=len(sel_rows),
                r_lo=int(rr.min()), r_hi=int(rr.max())))
    cols = ["f_c", "f_lo", "f_hi", "R", "am_lo", "am_hi", "peak_db", "n_cells", "n_rows", "r_lo", "r_hi"]
    B = pd.DataFrame(rows, columns=cols).sort_values("peak_db", ascending=False).reset_index(drop=True)
    B.index.name = "blob"
    return B


def refine_rate(P: np.ndarray, r_lo: int, r_hi: int, cols: slice, dt: float, R0: float) -> float:
    lev = 10 * np.log10(P[r_lo:r_hi + 1, cols].sum(axis=0) + 1e-20)
    lev = (lev - lev.mean()) * np.hanning(len(lev))
    nfft = 1 << (int(np.ceil(np.log2(len(lev)))) + 3)
    spec = np.abs(np.fft.rfft(lev, nfft))
    fq = np.fft.rfftfreq(nfft, dt)
    sel = np.flatnonzero(np.abs(fq - R0) < 0.6)
    if len(sel) == 0:            # a run too short for the FFT grid to have a bin within 0.6 Hz: keep the coarse rate
        return float(R0)
    k = sel[np.argmax(spec[sel])]
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return float(k * (fq[1] - fq[0]))


def best_segment(P: np.ndarray, r_lo: int, r_hi: int, dt: float, R0: float, fold_s: float) -> slice:
    """The fold_s-second stretch of the window in which this peak's rows are
    most strongly modulated at the coarse rate R0: the R-locked on/off series
    of the peak's rows (0.5 s chunks), averaged over rows, then a running mean
    of fold_s, and the maximum. Folding the central 30 s failed for callers
    that were silent there (no part seeded) or locked onto a neighbour that
    was singing; folding the loudest 30 s needs only the coarse rate, which
    the modulation map already gives (MJE, 2026-09-19)."""
    n = int(round(SLOW_S / dt))
    nch = P.shape[1] // n
    n_seg = max(1, int(round(fold_s / SLOW_S)))
    if nch <= n_seg:
        return slice(0, P.shape[1])
    s = locked_series(P[r_lo:r_hi + 1], dt, R0).mean(axis=0)
    c = np.concatenate([[0.0], np.cumsum(s)])
    run = c[n_seg:] - c[:-n_seg]                 # sum over each n_seg-chunk stretch
    k0 = int(np.argmax(run))
    c0 = k0 * n
    return slice(c0, min(P.shape[1], c0 + int(fold_s / dt)))


def fold(P: np.ndarray, r_lo: int, r_hi: int, cols: slice, dt: float, R: float) -> np.ndarray:
    Pw = P[r_lo:r_hi + 1, cols]
    t = np.arange(Pw.shape[1]) * dt
    b = np.floor(((t * R) % 1.0) * PHASE_BINS).astype(int)
    F = np.zeros((Pw.shape[0], PHASE_BINS))
    cnt = np.bincount(b, minlength=PHASE_BINS)
    for k in range(PHASE_BINS):
        F[:, k] = Pw[:, b == k].sum(axis=1)
    F /= np.maximum(cnt, 1)
    return F


def on_periods(P: np.ndarray, r_lo: int, r_hi: int, dt: float, R0: float):
    """Runs of song for a peak: the R0-locked series of its rows (SLOW_S
    chunks) averaged over rows; chunks above half the series' 95th percentile
    are ON; gaps of one chunk are bridged; runs shorter than ON_MIN_S are
    dropped. Returns a list of column slices and the on-fraction."""
    n = int(round(SLOW_S / dt))
    s = locked_series(P[r_lo:r_hi + 1], dt, R0).mean(axis=0)
    thr = 0.5 * np.percentile(s, 95)
    on = s > thr
    # bridge single-chunk gaps
    for i in range(1, len(on) - 1):
        if not on[i] and on[i - 1] and on[i + 1]:
            on[i] = True
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if (j - i) * SLOW_S >= ON_MIN_S:
                runs.append(slice(i * n, min(P.shape[1], j * n)))
            i = j
        else:
            i += 1
    return runs, float(on.mean())


def is_bout_rhythm(P: np.ndarray, r_lo: int, r_hi: int, dt: float, Rs: float, Rf: float) -> tuple[bool, float, float]:
    """Is the slow peak at Rs the BOUT rhythm of the fast singer at Rf in the
    same rows, or a separate slow singer on top of it? A bout rhythm switches
    the fast modulation on and off: the envelope of the level series' Rf
    component, folded at Rs, is deep and in phase with the level itself. A
    chirper riding on a continuous trill leaves the trill's modulation running
    through its own off phase (and, if anything, is anti-phased: the chirp
    fills the trill's gaps). Returns (bout?, min/max of the folded envelope,
    correlation of the two folds)."""
    L = 10 * np.log10(P[r_lo:r_hi + 1].sum(axis=0) + 1e-20)
    L = L - L.mean()
    n = len(L)
    X = np.fft.rfft(L * np.hanning(n))
    fq = np.fft.rfftfreq(n, dt)
    band = (fq > Rf * 0.75) & (fq < Rf * 1.25)
    Xa = np.where(band, X, 0)
    a = np.abs(np.fft.ifft(np.concatenate([2 * Xa, np.zeros(n - len(Xa))])))[:n]   # analytic envelope of the Rf component
    t = np.arange(n) * dt
    b = np.floor(((t * Rs) % 1.0) * PHASE_BINS).astype(int)
    cnt = np.maximum(np.bincount(b, minlength=PHASE_BINS), 1)
    fa = np.bincount(b, weights=a, minlength=PHASE_BINS) / cnt
    fl = np.bincount(b, weights=L, minlength=PHASE_BINS) / cnt
    k = PHASE_BINS // 3
    sa = np.sort(fa)
    ratio = float(sa[:k].mean() / max(sa[-k:].mean(), 1e-12))
    corr = float(np.corrcoef(fa, fl)[0, 1])
    return (ratio < BOUT_RATIO_MAX and corr > 0.3), ratio, corr


def fold_runs(P: np.ndarray, r_lo: int, r_hi: int, runs: list, dt: float, R: float) -> np.ndarray:
    """Fold each run of song separately (its own phase origin), align the
    runs on their in-band envelopes by circular cross-correlation (longest
    run is the reference, two passes), and average them weighted by length."""
    if len(runs) == 1:
        return fold(P, r_lo, r_hi, runs[0], dt, R)
    folds = [fold(P, r_lo, r_hi, sl, dt, R) for sl in runs]
    w = np.array([sl.stop - sl.start for sl in runs], float)
    order = np.argsort(-w)
    folds = [folds[i] for i in order]; w = w[order]
    def env(F):
        Fdb = 10 * np.log10(F + 1e-20)
        return np.clip(Fdb - np.median(Fdb, axis=1, keepdims=True), 0, None).sum(axis=0)
    envs = [env(F) for F in folds]
    ref = envs[0].copy(); shifts = [0] * len(folds)
    for _ in range(2):
        acc = np.zeros_like(ref); tot = 0.0
        for i, e in enumerate(envs):
            cc = np.fft.irfft(np.fft.rfft(e) * np.conj(np.fft.rfft(ref)), n=len(e))
            shifts[i] = int(np.argmax(cc))
            acc += w[i] * np.roll(e, -shifts[i]); tot += w[i]
        ref = acc / tot
    out = sum(w[i] * np.roll(folds[i], -shifts[i], axis=1) for i in range(len(folds))) / w.sum()
    return out


def refine_rate_runs(P: np.ndarray, r_lo: int, r_hi: int, runs: list, dt: float, R0: float) -> float:
    """Rate from the longest run of song (a single run keeps its phase, so
    the zero-padded FFT peak is sharp); falls back to refine_rate on it."""
    longest = max(runs, key=lambda sl: sl.stop - sl.start)
    return refine_rate(P, r_lo, r_hi, longest, dt, R0)


def subharmonic_rate(P: np.ndarray, r_lo: int, r_hi: int, runs: list, dt: float, R: float) -> tuple[float, int]:
    """Is the rhythm really at R/2 or R/3? The map's peak is at the strongest
    modulation harmonic, which for a chirp of few syllables or a slow pulse
    train is not the fundamental. Fold at R, R/2, R/3 over the same runs and
    take the sub-multiple whose mean first-harmonic depth over the peak's
    rows exceeds that at R by SUBHARM_GAIN. Returns (rate, m)."""
    def depth_at(Rm):
        F = fold_runs(P, r_lo, r_hi, runs, dt, Rm)
        d, _, _ = row_phase(F)
        return float(np.mean(d))
    # iterated (2026-10-11, METHOD 19 AJ): a slow rhythm's peak can sit on its 6th or 9th harmonic, and one division
    # left chirp rates at 2x or 3x their true value (a 2.2 Hz chirper read at 6.7 Hz from its 20 Hz peak); divide
    # again while a sub-multiple's fold is still SUBHARM_GAIN deeper, up to SUBHARM_ITER times
    best, bm = R, 1
    for _ in range(SUBHARM_ITER):
        d1 = depth_at(best)
        step, sm = best, 1
        for m in (2, 3):
            if best / m < AM_LO * 0.9:
                continue
            dm = depth_at(best / m)
            if dm > SUBHARM_GAIN * d1 and dm > (depth_at(step) if sm > 1 else d1):
                step, sm = best / m, m
        if sm == 1:
            break
        best, bm = step, bm * sm
    return best, bm


def row_phase(F: np.ndarray):
    """First Fourier harmonic of each row of the fold: modulation depth
    (2|c|/mean, relative to the row's own power), phase as a fraction of the
    period, and the ABSOLUTE in-phase amplitude 2|c|/n in linear power, which
    a steady neighbour sharing the row cannot dilute."""
    k = np.arange(F.shape[1])
    c = (F * np.exp(-2j * np.pi * k / F.shape[1])).sum(axis=1)
    depth = 2 * np.abs(c) / np.maximum(F.sum(axis=1), 1e-30)
    phase = (np.angle(c) / (2 * np.pi)) % 1.0
    amp = 2 * np.abs(c) / F.shape[1]
    return depth, phase, amp


def residual_profile(F: np.ndarray) -> np.ndarray:
    """Each row's in-phase residual (fold minus row mean) scaled to unit peak:
    the shape of the row's modulation independent of the steady power in it."""
    Lres = F - F.mean(axis=1, keepdims=True)
    return Lres / np.maximum(np.abs(Lres).max(axis=1, keepdims=True), 1e-30)


def locked_series(Pr: np.ndarray, dt: float, R: float) -> np.ndarray:
    """Per row, the amplitude (dB) of the level modulation AT R in each SLOW_S
    chunk: when this rate is present in this row. Unlike the raw level it is
    blind to a continuous neighbour at another rate sharing the bin."""
    n = int(round(SLOW_S / dt))
    nch = Pr.shape[1] // n
    L = 10 * np.log10(Pr[:, :nch * n] + 1e-20).reshape(Pr.shape[0], nch, n)
    L = L - L.mean(axis=2, keepdims=True)
    e = np.exp(-2j * np.pi * R * np.arange(n) * dt)
    return 2 * np.abs(L @ e) / n


def highpass(x: np.ndarray) -> np.ndarray:
    k = int(round(HP_S / SLOW_S)) | 1
    return x - scipy.ndimage.median_filter(x, size=(1, k) if x.ndim == 2 else k, mode="nearest")


def _cv(x: np.ndarray) -> float:
    """sd/mean of the on/off series after a 5 s median: bout structure survives
    the smoothing, the noise of a weak row does not."""
    k = int(round(5.0 / SLOW_S)) | 1
    s = scipy.ndimage.median_filter(x, size=k, mode="nearest")
    return float(s.std() / max(s.mean(), 1e-9))


def split_rows(depth, prof, raw, hp, r_lo, r_hi, pad_rows, corr_min, amp=None):
    """Walk the blob's rows from low to high frequency; start a new part where
    the fold profile changes abruptly from the previous row, or where a row
    with an on/off pattern does not follow the part's. One animal's adjacent
    rows have near-identical profiles even through an arched glide (a first-
    harmonic phase does not: the top of an arch sits ~0.2 period from its
    flanks and a phase-jump test flickers there). Parts under MIN_ROWS are
    discarded. Then each part is EXTENDED outward into the padded rows while
    the row's absolute in-phase amplitude stays within AMP_FLOOR_DB of the
    part's peak and its residual profile stays continuous: the blob's rows and
    the depth test are both relative to the row's own power, so a louder
    steady neighbour in a row hides the animal's energy there although the
    fold contains it (MJE, 2026-09-14). Returns a label per padded row."""
    n = len(depth)
    lab = np.full(n, -1)
    lo, hi = pad_rows, pad_rows + (r_hi - r_lo)
    g = -1
    prev = None
    for r in range(lo, hi + 1):
        if depth[r] < DEPTH_MIN:
            continue
        new = prev is None or r - prev > 2
        if not new:
            pc = np.corrcoef(prof[r], prof[prev])[0, 1]
            ref_raw = raw[lab == g].mean(axis=0)
            patterned = max(_cv(raw[r]), _cv(ref_raw)) > CV_MIN
            c = np.corrcoef(hp[r], hp[lab == g].mean(axis=0))[0, 1]
            new = pc < PROF_CORR_MIN or (patterned and c < corr_min)
        if new:
            g += 1
        lab[r] = g
        prev = r
    out = np.full(n, -1)
    k = 0
    for g in range(lab.max() + 1):
        rows = lab == g
        if rows.sum() >= MIN_ROWS:
            out[rows] = k
            k += 1
    if amp is not None and k > 0:
        floor = 10 ** (-AMP_FLOOR_DB / 10)
        for g in range(k):
            rows = np.flatnonzero(out == g)
            a_ref = amp[rows].max()
            for r0, step in ((rows.min(), -1), (rows.max(), +1)):
                r, nb = r0 + step, r0
                while 0 <= r < n and out[r] < 0 and amp[r] >= floor * a_ref and depth[r] >= DEPTH_EXT_MIN \
                        and np.corrcoef(prof[r], prof[nb])[0, 1] >= PROF_CORR_MIN:
                    out[r] = g
                    nb, r = r, r + step
    return out


def am_harmonic_of(parts: pd.DataFrame) -> np.ndarray:
    """A part whose rate is an integer multiple of a lower-rate part's rate, in
    overlapping rows, is that part's modulation harmonic: the same pulses seen
    at m*R. Structural, no co-variation needed; the fold at R already holds it.
    Returns the fundamental part's index, or -1."""
    out = np.full(len(parts), -1)
    for i, a in parts.iterrows():
        best = None
        for j, b in parts.iterrows():
            if b.R >= a.R * (1 - R_TOL):
                continue
            m = a.R / b.R
            overlap = a.f_lo <= b.f_hi + PAD_HZ and b.f_lo <= a.f_hi + PAD_HZ
            if overlap and abs(m - round(m)) < R_TOL * round(m) and 2 <= round(m) <= MAX_HARM:
                if best is None or b.R < parts.loc[best, "R"]:
                    best = j
        if best is not None:
            out[i] = best
    return out


def shift_z(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Correlation of two on/off series and its z against circular time shifts
    of one series -- the right null for the ALIGNMENT of two series, and one
    that calibrates itself to each pair's own noise level."""
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    gap = 2 * int(round(HP_S / SLOW_S))
    shifts = np.linspace(gap, n - gap, N_SHIFTS).astype(int)
    xs = (x - x.mean()) / (x.std() + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    rs = np.array([np.mean(xs * np.roll(ys, s)) for s in shifts])
    return r, float((r - rs.mean()) / (rs.std() + 1e-12))


def link_parts(parts: pd.DataFrame, hp_series: dict):
    """Fundamentals (parts that are nobody's AM harmonic) are taken in order of
    frequency. Each joins the existing animal with the same rate whose lowest
    part it co-varies with best -- r above R_MIN, z above Z_MIN against the
    shift null -- or founds a new animal. A star, not a chain: two parts join
    only through the animal's lowest part, so neighbours at the same rate
    cannot bridge two animals. AM harmonics follow their fundamental part.
    Pairs of fundamentals at different rates are the empirical null."""
    harm = am_harmonic_of(parts)
    animal = np.full(len(parts), -1)
    roots: list[int] = []
    links, null = [], []
    fund = [i for i in parts.index if harm[i] < 0]
    for i in sorted(fund, key=lambda k: parts.loc[k, "f_lo"]):
        best = None
        for a, root in enumerate(roots):
            r, z = shift_z(hp_series[i], hp_series[root])
            if abs(parts.loc[i, "R"] / parts.loc[root, "R"] - 1) >= R_TOL:
                null.append(r)
            elif r > R_MIN and z > Z_MIN and (best is None or z > best[2]):
                best = (a, r, z)
        if best is None:
            animal[i] = len(roots)
            roots.append(i)
        else:
            animal[i] = best[0]
            links.append((i, roots[best[0]], best[1], best[2]))
    # structural fallback for what co-variation could not carry (continuous
    # callers, faint harmonics): a lone fundamental at m*R whose band nests in
    # n x a component of an animal at R is that animal's (n*F, m*R) harmonic
    lone = [i for i in fund if list(animal).count(animal[i]) == 1 and harm[i] < 0]
    for i in lone:
        a_i, best = parts.loc[i], None
        for root_idx, root in enumerate(roots):
            R0 = parts.loc[root, "R"]
            m = a_i.R / R0
            if not (abs(m - round(m)) < R_TOL * round(m) and 1 <= round(m) <= MAX_HARM):
                continue
            # v2 step 5: the lone part must co-vary with the animal's root (a chirper that
            # happens to sit at a harmonic position of a trill does not) -- unless both are
            # continuous, where co-variation is untestable (the on/off series are noise) and
            # the harmonic of a continuous trill is the common case (month test 2026-09-25:
            # the gate alone left harmonics unattached and fragmented the fundamental's track)
            r_cov, _ = shift_z(hp_series[i], hp_series[root])
            both_cont = parts.loc[i, "on_frac"] >= ON_FRAC_CONT and parts.loc[root, "on_frac"] >= ON_FRAC_CONT
            if r_cov <= R_MIN and not both_cont:
                continue
            for j in parts.index[(animal == root_idx) & (harm < 0)]:
                if j == i:
                    continue
                c = parts.loc[j]
                for n in range(2, 5):
                    if a_i.f_lo >= n * c.f_lo - PAD_HZ and a_i.f_hi <= n * c.f_hi + PAD_HZ:
                        if best is None or R0 < parts.loc[roots[best], "R"]:
                            best = root_idx
        if best is not None and best != animal[i]:
            harm[i] = roots[best] if round(a_i.R / parts.loc[roots[best], "R"]) > 1 else harm[i]
            animal[i] = best
    for i in parts.index:
        if harm[i] >= 0:
            animal[i] = animal[harm[i]]
    return animal, harm, np.array(null), links


def carrier_estimates(fk: np.ndarray, F: np.ndarray, amp: np.ndarray, rr: np.ndarray):
    """Sub-bin carrier for a part. Rows are weighted by their absolute in-phase
    amplitude, so a neighbour's unmodulated energy in the same rows does not
    pull the estimate. Returns the centroid and the parabolically interpolated
    peak, both in Hz; the 86 Hz band edges cannot show a shift of tens of Hz,
    these can."""
    w = amp[rr]
    f_c = float(np.average(fk[rr], weights=w)) if w.sum() > 0 else float(fk[rr].mean())
    k = int(np.argmax(w))
    f_pk = float(fk[rr[k]])
    if 0 < k < len(rr) - 1 and rr[k + 1] - rr[k - 1] == 2:
        a, b, c = np.log(w[k - 1] + 1e-30), np.log(w[k] + 1e-30), np.log(w[k + 1] + 1e-30)
        den = a - 2 * b + c
        if den < 0:
            f_pk += 0.5 * (a - c) / den * (fk[1] - fk[0])
    return f_c, f_pk


def area_average_db(P: np.ndarray, ncol: int) -> np.ndarray:
    edges = np.linspace(0, P.shape[1], ncol + 1).astype(int)
    out = np.stack([P[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return 10 * np.log10(out + 1e-20)


def analyse_window(recorder: str, site: str, target: pd.Timestamp, dur: float, fmax: float = 22050.0) -> dict:
    """The whole per-window analysis: audio -> map -> blobs -> parts -> animals.
    Returns everything the figure and the downstream stages need."""
    sig, sr, t0, path = read_window(recorder, site, target, dur)
    out = analyse_signal(sig, sr, dur, fmax)
    out.update(path=path, t0=t0)
    return out


def analyse_signal(sig: np.ndarray, sr: int, dur: float, fmax: float = 22050.0, fold_s: float = FOLD_S) -> dict:
    """The analysis on a signal already in memory (a recorder window, or a
    reference recording of a known species). The fold covers the central
    fold_s seconds, or the whole clip if it is shorter."""
    if sr != 44100:
        from math import gcd
        # nothing above the source's own Nyquist is real: the resampler's
        # stop-band leakage mirrors strong low bands into the empty top half
        # (a 4.3 kHz singer at 22.05 kHz sampling reappears at 17.7 kHz), so
        # cap the analysed band at the original Nyquist
        fmax = min(fmax, sr / 2.0 * 0.98)
        g = gcd(int(sr), 44100)
        sig = scipy.signal.resample_poly(sig, 44100 // g, int(sr) // g).astype(np.float32)
        sr = 44100
    dur = min(dur, len(sig) / sr)
    fold_s = min(fold_s, dur)
    path, t0 = None, None
    f, t, Z = scipy.signal.stft(sig, fs=sr, window="hann", nperseg=NPERSEG,
                                noverlap=NPERSEG - HOP, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    dt = HOP / sr
    # no orthopteran carrier lies below FMIN_HZ; the bottom rows hold wind,
    # rumble and mains-type hum whose level modulation can pass the threshold
    keep = (f <= fmax) & (f >= FMIN_HZ)
    f, P = f[keep], P[keep]
    df = f[1] - f[0]

    fr, excess = am_map(P, dt)
    B = find_blobs(excess, f, fr)
    B["source"] = "win"
    # v2: peaks of the block-max map that the whole-window map does not have
    # (rows overlapping and rate within AM_GAP_BINS of an existing peak = same)
    if BLK_THRESH_DB > 0:
        _, excess_blk = am_map_blocks(P, dt)
        Bb = find_blobs(excess_blk, f, fr, thresh=BLK_THRESH_DB)
    else:
        Bb = B.iloc[0:0].copy()
    dfr = fr[1] - fr[0]
    add = []
    for b in Bb.itertuples():
        same = B[(B.r_lo <= b.r_hi) & (B.r_hi >= b.r_lo) & ((B.R - b.R).abs() <= AM_GAP_BINS * dfr)]
        if same.empty:
            add.append(b.Index)
    if add:
        Bb = Bb.loc[add].copy(); Bb["source"] = "blk"
        B = pd.concat([B, Bb], ignore_index=True)
        B.index.name = "blob"
    if B.empty:
        # a quiet or rain-swamped window: nothing modulated in the syllable band,
        # an empty inventory (the chirp pass may still find slow singers)
        A0 = pd.DataFrame(columns=["animal", "n_components", "n_am_harmonics", "F_lo", "F_hi", "F_c", "F_pk", "R", "chirp_rate",
                                   "onoff_cv", "on_frac", "depth", "components", "am_harmonics"])
        chirps = None
        if os.environ.get("BROOKS_CHIRP", "0") == "1":
            from chirp_channel import add_chirpers
            A0, chirps = add_chirpers(P, f, dt, A0, fold_s)
        return dict(path=path, t0=t0, dur=dur, sr=sr, f=f, df=df, dt=dt, P=P, fr=fr, excess=excess,
                    B=B, per_blob={}, parts=pd.DataFrame(columns=["blob", "part", "f_lo", "f_hi", "R", "animal", "am_harm_of"]),
                    A=A0, null=np.array([]), links=[], chirps=chirps)
    # per peak: the loudest fold_s stretch at the coarse rate, then the refined
    # rate on that stretch (was: the central fold_s for every peak)
    # v2 step 2: a slow peak sharing rows with a faster peak is that singer's bout rhythm
    B["bout_of"] = -1
    B["bout_ratio"] = np.nan
    slow = B.index[B.R < BOUT_MAX_HZ]
    for k in slow:
        b = B.loc[k]
        others = B[(B.R >= BOUT_MAX_HZ) & (B.r_lo <= b.r_hi) & (B.r_hi >= b.r_lo)].sort_values("peak_db", ascending=False)
        for o in others.itertuples():
            lo, hi = max(int(b.r_lo), int(o.r_lo)), min(int(b.r_hi), int(o.r_hi))
            bout, ratio, corr = is_bout_rhythm(P, lo, hi, dt, float(b.R), float(o.R))
            B.loc[k, "bout_ratio"] = ratio
            if bout:
                B.loc[k, "bout_of"] = int(o.Index)
                break
    keep = B.index[B.bout_of < 0]
    bout_rate = {int(B.loc[k, "bout_of"]): float(B.loc[k, "R"]) for k in B.index if B.loc[k, "bout_of"] >= 0}
    B = B.loc[keep].copy()

    # v2 step 3: on periods -> rate refinement and fold over the song, not the silence
    segs, on_fracs, R_fine, sub_m = [], [], [], []
    for b in B.itertuples():
        r_lo, r_hi = int(b.r_lo), int(b.r_hi)
        runs, on_frac = on_periods(P, r_lo, r_hi, dt, b.R)
        if on_frac >= ON_FRAC_CONT or not runs:
            runs = [best_segment(P, r_lo, r_hi, dt, b.R, fold_s)]
        else:
            # cap the total folded time at ~fold_s by keeping the longest runs
            runs = sorted(runs, key=lambda sl: -(sl.stop - sl.start))
            tot, kept = 0, []
            for sl in runs:
                kept.append(sl); tot += sl.stop - sl.start
                if tot * dt >= fold_s:
                    break
            runs = sorted(kept, key=lambda sl: sl.start)
        R1 = refine_rate_runs(P, r_lo, r_hi, runs, dt, b.R)
        R2, m = subharmonic_rate(P, r_lo, r_hi, runs, dt, R1)         # v2 step 4
        if m > 1:
            R2 = refine_rate(P, r_lo, r_hi, max(runs, key=lambda sl: sl.stop - sl.start), dt, R2) if R2 >= AM_LO else R2
        segs.append(runs); on_fracs.append(on_frac); R_fine.append(R2); sub_m.append(m)
    B["R_fine"] = R_fine
    B["on_frac_map"] = on_fracs
    B["sub_m"] = sub_m
    B["n_runs"] = [len(r) for r in segs]
    B["fold_t0"] = [r[0].start * dt for r in segs]
    B["bout_rate"] = [bout_rate.get(int(k), np.nan) for k in B.index]
    # a sub-harmonic-corrected peak that now has the rate of another peak in
    # overlapping rows is that peak's modulation harmonic seen twice: drop it
    # (the fold at R already holds the harmonic); keep the louder of the pair
    seg_of = dict(zip(B.index, segs))
    drop = set()
    idx = list(B.index)
    for i in idx:
        for j in idx:
            if i >= j or i in drop or j in drop:
                continue
            bi, bj = B.loc[i], B.loc[j]
            if (bi.sub_m > 1 or bj.sub_m > 1) and abs(bi.R_fine / bj.R_fine - 1) < R_TOL \
                    and bi.r_lo <= bj.r_hi and bj.r_lo <= bi.r_hi:
                drop.add(i if bi.peak_db < bj.peak_db else j)
    if drop:
        B = B.drop(index=list(drop))
        segs = [seg_of[k] for k in B.index]

    # ---- per blob: fold, phase, on/off series; then rows -> parts
    pad = int(round(PAD_HZ / df))
    per_blob = {}
    for (k, b), cols in zip(B.iterrows(), segs):
        r_lo, r_hi = int(b.r_lo), int(b.r_hi)
        p_lo, p_hi = max(0, r_lo - pad), min(len(f) - 1, r_hi + pad)
        F = fold_runs(P, p_lo, p_hi, cols, dt, b.R_fine)
        depth, phase, amp = row_phase(F)
        prof = residual_profile(F)
        raw = locked_series(P[p_lo:p_hi + 1], dt, b.R_fine)
        per_blob[k] = dict(p_lo=p_lo, p_hi=p_hi, F=F, depth=depth, phase=phase, amp=amp, prof=prof,
                           raw=raw, hp=highpass(raw), cols=cols)

    def make_parts(corr_min):
        rows, series = [], {}
        for k, b in B.iterrows():
            d = per_blob[k]
            lab = split_rows(d["depth"], d["prof"], d["raw"], d["hp"], int(b.r_lo), int(b.r_hi),
                             int(b.r_lo) - d["p_lo"], corr_min, amp=d["amp"])
            d["lab"] = lab
            for g in range(lab.max() + 1):
                rr = np.flatnonzero(lab == g)
                raw_m = d["raw"][rr].mean(axis=0)
                pid = len(rows)
                series[pid] = d["hp"][rr].mean(axis=0)
                f_c, f_pk = carrier_estimates(f[d["p_lo"]:d["p_hi"] + 1], d["F"], d["amp"], rr)
                # absolute R-locked amplitude per padded row, so a later stage
                # can measure the carrier over a FIXED frequency window
                lp = d["amp"]
                rows.append(dict(
                    blob=k, part=g, f_lo=f[d["p_lo"] + rr.min()], f_hi=f[d["p_lo"] + rr.max()],
                    f_c=f_c, f_pk=f_pk, n_rows=len(rr), R=b.R_fine,
                    prof_f0=f[d["p_lo"]], prof=" ".join(f"{v:.3g}" for v in lp),
                    phase=float(np.angle(np.exp(2j * np.pi * d["phase"][rr]).mean()) / (2 * np.pi) % 1.0),
                    # depth over the SEED rows (relative depth >= DEPTH_MIN), so the
                    # rows the part was extended into do not dilute it
                    depth=float(d["depth"][rr][d["depth"][rr] >= DEPTH_MIN].mean()
                                if (d["depth"][rr] >= DEPTH_MIN).any() else d["depth"][rr].mean()),
                    onoff_cv=_cv(raw_m),
                    on_frac=float((raw_m > 0.5 * raw_m.max()).mean())))
        return pd.DataFrame(rows), series

    parts, series = make_parts(corr_min=ROW_CORR_MIN)
    animal, harm, null, links = link_parts(parts, series)
    parts["animal"] = animal
    parts["am_harm_of"] = harm

    # ---- animals: one row each; fundamental = lowest-frequency component
    an_rows = []
    for a, g in parts.groupby("animal"):
        comps = g[g.am_harm_of < 0].sort_values("f_lo")
        if comps.empty:
            continue
        fund = comps.iloc[0]
        bb = B.loc[fund.blob]
        # v2: a chirper carries two rates. A fast singer with a bout rhythm in its rows
        # reports chirp_rate = that rhythm; a lone slow peak (< BOUT_MAX_HZ) IS a chirp
        # rate, and its syllable rate is read from the level spectrum inside the chirps
        # (chirp_channel.syllable_rate_in_rows) or left NaN. The night linker
        # (night_animals2.py) matches on either rate, so a singer seen at its syllable
        # rate in one window and at its chirp rate in the next stays one track.
        R_out, chirp_rate = float(fund.R), float(bb.bout_rate)
        if fund.R < BOUT_MAX_HZ:
            chirp_rate = float(fund.R)
            runs_b = seg_of.get(fund.blob) or [slice(0, P.shape[1])]
            longest = max(runs_b, key=lambda sl: sl.stop - sl.start)
            if SYL_V2:
                # the time-domain reader (METHOD 19 AG): the chirps found as events in the band's envelope, the
                # pulses counted inside each; a continuous neighbor in the rows adds no bursts and cannot win
                from chirp_channel import syllable_rate_from_chirps
                res = syllable_rate_from_chirps(sig, sr, float(fund.f_lo), float(fund.f_hi), longest.start * dt, longest.stop * dt)
                R_out = float(res["Rs"]) if np.isfinite(res["Rs"]) else np.nan
            else:
                Rs, ex_db, sb_db = syllable_rate_in_rows(P, int(bb.r_lo), int(bb.r_hi), longest, dt, float(fund.R))
                # accept the syllable rate as the chirp channel does: with sidebands at Rs +/- Rc
                # (pulses grouped into chirps), or clear and not owned by another peak sharing
                # these rows (a neighbouring triller's rate leaks into the row sum otherwise)
                owned = False
                if np.isfinite(Rs):
                    others = B[(B.index != fund.blob) & (B.r_lo <= bb.r_hi) & (B.r_hi >= bb.r_lo)]
                    owned = bool((np.abs(others.R_fine / Rs - 1) < 0.06).any())
                ok = np.isfinite(Rs) and ((np.isfinite(sb_db) and sb_db >= 2.0) or (ex_db >= 5.0 and not owned))
                R_out = float(Rs) if ok else np.nan
        an_rows.append(dict(
            animal=a, n_components=len(comps), n_am_harmonics=int((g.am_harm_of >= 0).sum()),
            F_lo=fund.f_lo, F_hi=fund.f_hi, F_c=fund.f_c, F_pk=fund.f_pk, R=R_out, chirp_rate=chirp_rate,
            onoff_cv=fund.onoff_cv, on_frac=fund.on_frac,
            depth=fund.depth,
            components=";".join(f"{r.f_lo:.0f}-{r.f_hi:.0f}" for r in comps.itertuples()),
            am_harmonics=";".join(f"{r.R / fund.R:.0f}x@{r.f_lo:.0f}-{r.f_hi:.0f}" for r in g[g.am_harm_of >= 0].itertuples()),
            bout_rate=float(bb.bout_rate), n_runs=int(bb.n_runs), sub_m=int(bb.sub_m), on_frac_map=float(bb.on_frac_map),
            source=str(bb.source)))
    A = pd.DataFrame(an_rows, columns=["animal", "n_components", "n_am_harmonics", "F_lo", "F_hi", "F_c", "F_pk", "R", "chirp_rate",
                                       "onoff_cv", "on_frac", "depth", "components", "am_harmonics", "bout_rate", "n_runs", "sub_m", "on_frac_map", "source"])
    A = A.sort_values(["depth"], ascending=False)
    if SYL_V2 and len(A):
        # v2 (METHOD 19 AG): a chirper whose syllable rate was read, and a separate animal at that rate (3 %)
        # whose fundamental band shares its rows, are one singer: the chirp peak and the syllable peak the bout
        # test failed to tie together. The twin's parts become the chirper's components.
        Ai = A.set_index("animal")
        merged = {}
        for a in Ai.index[(Ai.chirp_rate.notna()) & (Ai.R.notna()) & (Ai.bout_rate.isna())]:   # bout_rate NaN: the fundamental is the slow peak itself
            ra = Ai.loc[a]
            for b_ in Ai.index:
                if b_ == a or b_ in merged or pd.isna(Ai.loc[b_, "R"]):
                    continue
                rb = Ai.loc[b_]
                if rb.F_lo <= ra.F_hi and rb.F_hi >= ra.F_lo and abs(rb.R / ra.R - 1) < 0.03 and pd.isna(rb.chirp_rate):
                    merged[b_] = a
        if merged:
            for b_, a in merged.items():
                parts.loc[parts.animal == b_, "animal"] = a
                comps = parts[(parts.animal == a) & (parts.am_harm_of < 0)].sort_values("f_lo")
                Ai.loc[a, "n_components"] = len(comps)
                Ai.loc[a, "F_lo"] = float(min(Ai.loc[a, "F_lo"], Ai.loc[b_, "F_lo"])); Ai.loc[a, "F_hi"] = float(max(Ai.loc[a, "F_hi"], Ai.loc[b_, "F_hi"]))
                Ai.loc[a, "components"] = ";".join(f"{r.f_lo:.0f}-{r.f_hi:.0f}" for r in comps.itertuples())
            A = Ai.drop(index=list(merged)).reset_index()
    chirps = None
    if os.environ.get("BROOKS_CHIRP", "0") == "1":       # second pass for chirpers / frogs (chirp_channel.py)
        from chirp_channel import add_chirpers
        A, chirps = add_chirpers(P, f, dt, A, fold_s)
    return dict(path=path, t0=t0, dur=dur, sr=sr, f=f, df=df, dt=dt, P=P, fr=fr, excess=excess,
                B=B, per_blob=per_blob, parts=parts, A=A, null=null, links=links, chirps=chirps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--time", required=True, help="local time, e.g. '2024-09-12 20:40'")
    ap.add_argument("--dur", type=float, default=300.0,
                    help="window for blobs and on/off patterns; the fold uses each peak's loudest 30 s")
    ap.add_argument("--fmax", type=float, default=22050.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    target = pd.Timestamp(args.time, tz=TZ)
    W = analyse_window(args.recorder, args.site, target, args.dur, args.fmax)
    path, t0, f, df, dt, P, fr, excess, B, per_blob, parts, A, null, links = (
        W[k] for k in ("path", "t0", "f", "df", "dt", "P", "fr", "excess", "B", "per_blob", "parts", "A", "null", "links"))
    print(f"{path}\nwindow starts {t0.tz_convert(TZ)} local, {args.dur:.1f} s")

    stem = args.out or os.path.join(
        OUT_DIR, f"blobs_{args.recorder}_{args.site}_{target:%Y%m%d_%H%M}")
    B.to_csv(stem + ".csv", float_format="%.3f")
    parts.to_csv(stem + "_parts.csv", index=False, float_format="%.3f")
    A.to_csv(stem + "_animals.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    if args.no_plot or B.empty:
        print(f"wrote {stem}.csv / _parts.csv / _animals.csv: {len(B)} blobs -> {len(parts)} parts -> {len(A)} animals")
        return

    # ---- figure
    show = B.head(MAX_SHOW)
    per_row = 2
    nrow = int(np.ceil(len(show) / per_row))
    fig = plt.figure(figsize=(24, 9 + 3.4 * nrow))
    gs = fig.add_gridspec(2 + nrow, 3 * per_row, height_ratios=[4, 5] + [3] * nrow,
                          width_ratios=[3, 1, 4] * per_row, hspace=0.5, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    Sdb = area_average_db(P, 1200)
    ax.imshow(Sdb, origin="lower", aspect="auto", cmap="magma",
              extent=[0, args.dur, f[0], f[-1]],
              vmin=np.percentile(Sdb, 5), vmax=np.percentile(Sdb, 99.5))
    ax.set_ylabel("Hz")
    ax.set_xlabel("s")
    ax.set_title(f"{args.recorder} @ {args.site}  {t0.tz_convert(TZ):%Y-%m-%d %H:%M:%S} local, {args.dur:.0f} s")

    colors = plt.cm.tab20.colors
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(excess, origin="lower", aspect="auto", cmap="gray",
              extent=[fr[0], fr[-1], f[0], f[-1]], vmin=0, vmax=max(15, THRESH_DB * 2.5))
    for p in parts.itertuples():
        b = B.loc[p.blob]
        ax.add_patch(plt.Rectangle((b.am_lo - 0.3, p.f_lo - df / 2), b.am_hi - b.am_lo + 0.6,
                                   p.f_hi - p.f_lo + df, fill=False, lw=1.3, color=colors[p.animal % 20]))
        ax.text(b.am_hi + 0.5, (p.f_lo + p.f_hi) / 2, f"a{p.animal}", color=colors[p.animal % 20],
                fontsize=7, va="center")
    for k, b in B.iterrows():
        if k < MAX_SHOW:
            ax.text(b.am_lo - 0.6, b.f_hi + 60, f"#{k}", color="w", fontsize=8, ha="right")
    ax.set_xlabel("AM rate (Hz)")
    ax.set_ylabel("carrier (Hz)")
    ax.set_title(f"frequency x AM excess (dB); {len(B)} blobs > {THRESH_DB:.0f} dB -> {len(parts)} parts -> "
                 f"{len(A)} animals ({(A.n_components > 1).sum()} with >1 component, "
                 f"{int(parts.am_harm_of.ge(0).sum())} AM-harmonic parts absorbed); "
                 f"link needs r > {R_MIN} and shift-null z > {Z_MIN}; "
                 f"{len(null)} different-rate pairs: r p99 {np.percentile(null, 99):+.2f}")

    for k, b in show.iterrows():
        d = per_blob[k]
        p_lo, p_hi, F, depth, phase, raw, lab = (d[x] for x in ("p_lo", "p_hi", "F", "depth", "phase", "raw", "lab"))
        fk = f[p_lo:p_hi + 1]
        row, col0 = 2 + k // per_row, 3 * (k % per_row)
        per_ms = 1000 / b.R_fine
        bparts = parts[parts.blob == k].set_index("part")
        col_of = {g: colors[bparts.loc[g, "animal"] % 20] for g in bparts.index}

        ax = fig.add_subplot(gs[row, col0])
        Fdb = 10 * np.log10(F + 1e-20)
        Fdb -= Fdb.mean(axis=1, keepdims=True)
        Fn = np.concatenate([Fdb] * N_PERIODS, axis=1)
        ax.imshow(Fn, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, N_PERIODS * per_ms, fk[0], fk[-1]],
                  vmin=np.percentile(Fn, 2), vmax=np.percentile(Fn, 99.5))
        for g in bparts.index:
            rr = np.flatnonzero(lab == g)
            for n in range(N_PERIODS):
                ax.scatter((phase[rr] + n) * per_ms, fk[rr], s=7, color=col_of[g], edgecolors="none")
        ax.axhline(b.f_lo - df / 2, color="w", lw=0.6, ls=":")
        ax.axhline(b.f_hi + df / 2, color="w", lw=0.6, ls=":")
        who = ", ".join(f"a{bparts.loc[g, 'animal']}" for g in bparts.index)
        ax.set_title(f"#{k}  {b.f_lo:.0f}-{b.f_hi:.0f} Hz  R={b.R_fine:.2f} Hz  +{b.peak_db:.0f} dB"
                     f"  ->  {who or 'nothing modulated'}", fontsize=9, loc="left")
        ax.set_xlabel(f"ms ({N_PERIODS} periods, fold over the loudest {FOLD_S:.0f} s, from {b.fold_t0:.0f} s)")

        ax = fig.add_subplot(gs[row, col0 + 1])
        mean_db = 10 * np.log10(F.mean(axis=1) + 1e-20)
        ax.plot(mean_db, fk, color="k", lw=1)
        for g in bparts.index:
            rr = np.flatnonzero(lab == g)
            ax.plot(mean_db[rr], fk[rr], ".", color=col_of[g], ms=5)
        ax.set_ylim(fk[0], fk[-1])
        ax.set_yticklabels([])
        ax.set_xlabel("mean dB")
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[row, col0 + 2])
        v = np.percentile(raw, 99)
        ax.imshow(raw, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, args.dur, fk[0], fk[-1]], vmin=0, vmax=v)
        for g in bparts.index:
            rr = np.flatnonzero(lab == g)
            ax.axhline(fk[rr.min()] - df / 2, color="w", lw=0.8)
            ax.axhline(fk[rr.max()] + df / 2, color="w", lw=0.8)
            ax.text(-args.dur * 0.01, fk[rr].mean(), f"a{bparts.loc[g, 'animal']}", color=col_of[g],
                    fontsize=8, ha="right", va="center", clip_on=False)
        ax.set_yticklabels([])
        ax.set_xlabel(f"s  (modulation amplitude at R, {SLOW_S:.1f} s chunks, 0-{v:.1f} dB)")

    fig.savefig(stem + ".png", dpi=100, bbox_inches="tight")
    print(f"wrote {stem}.png / .csv / _parts.csv / _animals.csv")
    print(f"{len(null)} different-rate pairs: r median {np.median(null):+.2f}, p95 {np.percentile(null, 95):+.2f}, "
          f"p99 {np.percentile(null, 99):+.2f}; {len(links)} within-animal links")
    if links:
        L = pd.DataFrame(links, columns=["a", "b", "r", "z"])
        print(f"links: r min {L.r.min():.2f} median {L.r.median():.2f}; z min {L.z.min():.1f} median {L.z.median():.1f}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(A.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
