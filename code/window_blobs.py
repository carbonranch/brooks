"""One window: the frequency x AM map, the blobs in it, and the animals behind them.

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

AUDIO_ROOT = "/Volumes/8M2A/Data/Carbonranch/Ferguson/Audio"
AUDIO_ROOTS = [AUDIO_ROOT, "/Volumes/8M2C/Data/Carbonranch/Ferguson/Audio", "/Volumes/8M2D/Data/Carbonranch/Ferguson/Audio"]
# other archives (e.g. A2O under /Volumes/8M2C/Data/A2O/<Region>/Audio) and
# their local zone: BROOKS_AUDIO_ROOTS=path1:path2 and BROOKS_TZ=Australia/Adelaide
AUDIO_ROOTS = [p for p in os.environ.get("BROOKS_AUDIO_ROOTS", "").split(":") if p] + AUDIO_ROOTS
TZ = os.environ.get("BROOKS_TZ", "America/Chicago")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NPERSEG, HOP = 512, 128          # 86 Hz bins, 11.6 ms window, 2.9 ms hop
AM_LO, AM_HI = 8.0, 100.0        # below ~8 Hz the level spectrum is a 1/f slope, not pulses
AM_LO = float(os.environ.get("BROOKS_AM_LO", AM_LO))   # test override (2026-09-22: does 2 Hz find slow tinklers?)
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
    sel = (fr >= AM_LO) & (fr <= AM_HI)
    fr, Sdb = fr[sel], 10 * np.log10(S[:, sel] + 1e-30)
    k = int(round(BASE_HZ / (fr[1] - fr[0]))) | 1
    base = scipy.ndimage.median_filter(Sdb, size=(1, k), mode="nearest")
    return fr, Sdb - base


def find_blobs(excess: np.ndarray, f: np.ndarray, fr: np.ndarray) -> pd.DataFrame:
    """8-connected regions above threshold, then each region is divided along
    the AM axis: rows are grouped by the AM bin of their own peak, split where
    consecutive peak bins are more than AM_GAP_BINS apart. Two animals at 20.2
    and 21.0 Hz in neighbouring bands otherwise become one region carrying
    one rate label, and the fold at that rate smears the other animal."""
    lab, n = scipy.ndimage.label(excess > THRESH_DB, structure=np.ones((3, 3)))
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
    if B.empty:
        # a quiet or rain-swamped window: nothing modulated in the syllable band,
        # an empty inventory (the chirp pass may still find slow singers)
        A0 = pd.DataFrame(columns=["animal", "n_components", "n_am_harmonics", "F_lo", "F_hi", "F_c", "F_pk", "R",
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
    if os.environ.get("BROOKS_FOLD", "loudest") == "central":     # A/B switch for the stability test
        c0 = max(0, int((dur / 2 - fold_s / 2) / dt))
        segs = [slice(c0, min(P.shape[1], c0 + int(fold_s / dt)))] * len(B)
    else:
        segs = [best_segment(P, int(b.r_lo), int(b.r_hi), dt, b.R, fold_s) for b in B.itertuples()]
    B["R_fine"] = [refine_rate(P, int(b.r_lo), int(b.r_hi), sg, dt, b.R) for b, sg in zip(B.itertuples(), segs)]
    B["fold_t0"] = [sg.start * dt for sg in segs]          # seconds into the window

    # ---- per blob: fold, phase, on/off series; then rows -> parts
    pad = int(round(PAD_HZ / df))
    per_blob = {}
    for (k, b), cols in zip(B.iterrows(), segs):
        r_lo, r_hi = int(b.r_lo), int(b.r_hi)
        p_lo, p_hi = max(0, r_lo - pad), min(len(f) - 1, r_hi + pad)
        F = fold(P, p_lo, p_hi, cols, dt, b.R_fine)
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
        an_rows.append(dict(
            animal=a, n_components=len(comps), n_am_harmonics=int((g.am_harm_of >= 0).sum()),
            F_lo=fund.f_lo, F_hi=fund.f_hi, F_c=fund.f_c, F_pk=fund.f_pk, R=fund.R,
            onoff_cv=fund.onoff_cv, on_frac=fund.on_frac,
            depth=fund.depth,
            components=";".join(f"{r.f_lo:.0f}-{r.f_hi:.0f}" for r in comps.itertuples()),
            am_harmonics=";".join(f"{r.R / fund.R:.0f}x@{r.f_lo:.0f}-{r.f_hi:.0f}" for r in g[g.am_harm_of >= 0].itertuples())))
    A = pd.DataFrame(an_rows, columns=["animal", "n_components", "n_am_harmonics", "F_lo", "F_hi", "F_c", "F_pk", "R",
                                       "onoff_cv", "on_frac", "depth", "components", "am_harmonics"])
    A = A.sort_values(["depth"], ascending=False)
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
