"""The chirp channel: a second detection pass for singers whose strongest
rhythm is slower than the syllable band.

The main BROOKS pass finds singers by their syllable rate (8-100 Hz). A
chirping cricket -- a few syllables per chirp, chirps a few times a second --
has a weak, broadened syllable peak (split into sidebands +/- the chirp rate)
that often stays under the 6 dB threshold, while its chirp rate is a sharp,
strong peak below 8 Hz. Frog calls are the same shape: pulses inside notes,
notes repeated at 0.5-5 Hz. This module:

1. builds a second modulation map for CHIRP_LO..CHIRP_HI Hz with a finer
   Welch resolution and a baseline that follows the 1/f slope (running
   median on a log-rate axis), and finds 8-connected peaks >= THRESH_DB over
   >= 2 rows, divided along the rate axis as in the main pass;
2. for each chirp peak: refines the chirp rate Rc on the loudest 30 s,
   folds the rows at Rc with CHIRP_BINS phase bins, and reads from the fold
   the chirp duty and the syllable rate INSIDE the chirp (spacing of pulses
   in the on-portion by autocorrelation), with the number of pulses per
   chirp;
3. returns chirp singers as rows for the animal table with R = the
   within-chirp syllable rate (so they sit in the same carrier x rate plane)
   plus chirp_rate, pulses_per_chirp, chirp_duty, chirp_ms; a chirp peak whose
   band and syllable rate match an existing (trill-pass) singer within
   CHIRP_MATCH_R is attached to that singer instead of duplicating it.

Enabled from analyse_signal with BROOKS_CHIRP=1 (off by default while the
corpus run is in progress).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal

CHIRP_LO, CHIRP_HI = 1.5, 10.0    # chirp / note rates (Hz)
CHIRP_NPER = 8192                 # Welch segments: 0.042 Hz bins at 344.5 Hz frame rate
BASE_OCT = 0.6                    # running-median baseline width on the log-rate axis (octaves)
THRESH_DB = 6.0
MIN_ROWS = 2
GAP_HZ = 0.12                     # rows whose chirp-rate peaks differ by more than this are different singers
CHIRP_BINS = 96                   # phase bins in the chirp fold (a 100 ms chirp of 6 pulses needs >= 48)
MIN_PULSES = 2
CHIRP_MATCH_R = 0.06              # a chirp peak matches a trill-pass singer if its within-chirp rate agrees this well
SLOW_S = 0.5
FOLD_S = 30.0


def chirp_map(P: np.ndarray, dt: float):
    """Per carrier row: modulation spectrum of the dB level over CHIRP_LO..
    CHIRP_HI, minus a running median taken on a log-rate axis so the 1/f
    slope is followed rather than cut."""
    L = 10 * np.log10(P + 1e-20)
    L -= L.mean(axis=1, keepdims=True)
    fr, S = scipy.signal.welch(L, fs=1 / dt, nperseg=CHIRP_NPER, axis=1)
    sel = (fr >= CHIRP_LO * 0.5) & (fr <= CHIRP_HI * 2.0)          # margin for the baseline
    fr, Sdb = fr[sel], 10 * np.log10(S[:, sel] + 1e-30)
    lg = np.log2(fr)
    grid = np.arange(lg[0], lg[-1], 0.01)
    k = int(round(BASE_OCT / 0.01)) | 1
    base = np.empty_like(Sdb)
    for i in range(Sdb.shape[0]):
        on_grid = np.interp(grid, lg, Sdb[i])
        med = scipy.ndimage.median_filter(on_grid, size=k, mode="nearest")
        base[i] = np.interp(lg, grid, med)
    keep = (fr >= CHIRP_LO) & (fr <= CHIRP_HI)
    return fr[keep], (Sdb - base)[:, keep]


def find_chirp_peaks(excess: np.ndarray, f: np.ndarray, fr: np.ndarray) -> pd.DataFrame:
    lab, n = scipy.ndimage.label(excess > THRESH_DB, structure=np.ones((3, 3)))
    dfr = fr[1] - fr[0]
    gap_bins = max(1, int(round(GAP_HZ / dfr)))
    rows = []
    for i in range(1, n + 1):
        r, c = np.nonzero(lab == i)
        ur = np.unique(r)
        if len(ur) < MIN_ROWS:
            continue
        peak_col = np.array([c[r == rr][np.argmax(excess[rr, c[r == rr]])] for rr in ur])
        order = np.argsort(peak_col)
        groups, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if peak_col[b] - peak_col[a] > gap_bins:
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
            kk = int(np.argmax(w))
            rows.append(dict(f_c=float(np.average(f[rr], weights=w)), f_lo=float(f[rr.min()]), f_hi=float(f[rr.max()]),
                             Rc0=float(fr[cc[kk]]), peak_db=float(w[kk]), n_rows=len(sel_rows), r_lo=int(rr.min()), r_hi=int(rr.max())))
    return pd.DataFrame(rows, columns=["f_c", "f_lo", "f_hi", "Rc0", "peak_db", "n_rows", "r_lo", "r_hi"]).sort_values("peak_db", ascending=False).reset_index(drop=True)


def _locked(Pr: np.ndarray, dt: float, R: float) -> np.ndarray:
    n = int(round(SLOW_S / dt))
    nch = Pr.shape[1] // n
    L = 10 * np.log10(Pr[:, :nch * n] + 1e-20).reshape(Pr.shape[0], nch, n)
    L = L - L.mean(axis=2, keepdims=True)
    e = np.exp(-2j * np.pi * R * np.arange(n) * dt)
    return 2 * np.abs(L @ e) / n


def best_segment(P, r_lo, r_hi, dt, R0, fold_s):
    n = int(round(SLOW_S / dt))
    nch = P.shape[1] // n
    n_seg = max(1, int(round(fold_s / SLOW_S)))
    if nch <= n_seg:
        return slice(0, P.shape[1])
    s = _locked(P[r_lo:r_hi + 1], dt, R0).mean(axis=0)
    c = np.concatenate([[0.0], np.cumsum(s)])
    run = c[n_seg:] - c[:-n_seg]
    c0 = int(np.argmax(run)) * n
    return slice(c0, min(P.shape[1], c0 + int(fold_s / dt)))


def refine_rate(P, r_lo, r_hi, cols, dt, R0, half=0.25):
    lev = 10 * np.log10(P[r_lo:r_hi + 1, cols].sum(axis=0) + 1e-20)
    lev = (lev - lev.mean()) * np.hanning(len(lev))
    nfft = 1 << (int(np.ceil(np.log2(len(lev)))) + 3)
    spec = np.abs(np.fft.rfft(lev, nfft))
    fq = np.fft.rfftfreq(nfft, dt)
    sel = np.flatnonzero(np.abs(fq - R0) < half)
    k = sel[np.argmax(spec[sel])]
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        den = a - 2 * b + c
        if den < 0:
            k = k + 0.5 * (a - c) / den
    return float(k * (fq[1] - fq[0]))


def fold(P, r_lo, r_hi, cols, dt, R, nb=CHIRP_BINS):
    Pw = P[r_lo:r_hi + 1, cols]
    t = np.arange(Pw.shape[1]) * dt
    b = np.floor(((t * R) % 1.0) * nb).astype(int)
    F = np.zeros((Pw.shape[0], nb))
    cnt = np.bincount(b, minlength=nb)
    for k in range(nb):
        F[:, k] = Pw[:, b == k].sum(axis=1)
    return F / np.maximum(cnt, 1)


def chirp_shape(F: np.ndarray, Rc: float):
    """From the chirp fold (rows x CHIRP_BINS): envelope, duty, chirp
    duration, and the syllable rate inside the chirp from the pulse spacing
    of the on-portion (autocorrelation of the envelope restricted to the
    chirp, first peak beyond one bin). Returns dict or None."""
    Fm = np.clip(F - F.mean(axis=1, keepdims=True), 0, None)
    env = Fm.sum(axis=0)
    if env.max() <= 0:
        return None
    env = env / env.max()
    nb = len(env)
    on = env >= 0.5
    # rotate so the longest on-run starts at 0
    on2 = np.concatenate([on, on]).astype(int)
    best_len, best_start = 0, 0
    i = 0
    while i < nb:
        if on2[i]:
            j = i
            while j < 2 * nb and on2[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start = j - i, i
            i = j
        else:
            i += 1
    if best_len < 3:
        return None
    env_r = np.roll(env, -best_start)
    duty = min(1.0, best_len / nb)
    seg = env_r[:min(nb, best_len + 2)]
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    ac /= (ac[0] + 1e-12)
    # first local maximum beyond lag 1 with positive correlation
    lag = None
    for k in range(2, len(ac) - 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.1:
            lag = k
            break
    period_s = 1.0 / Rc
    chirp_ms = 1000 * duty * period_s
    # the autocorrelation estimate of the pulse spacing is kept for reference
    # only; chirp-to-chirp jitter smears pulses in the fold, so the pulse
    # rate is measured from the spectrum instead (syllable_rate_in_rows)
    Rs_ac = nb / (lag * period_s) if lag is not None else np.nan
    return dict(duty=duty, chirp_ms=chirp_ms, Rs_ac=float(Rs_ac), env=env_r)


SYL_LO, SYL_HI = 8.0, 100.0
SYL_MIN_DB = 3.0                  # the row-summed pulse peak must clear its 15 Hz running median by this much
SIDEBAND_MIN_DB = 2.0             # ... and show sidebands at Rs +/- Rc, the signature of pulses grouped into chirps
SYL_ALONE_DB = 5.0                # ... or be this clear with no trill-pass singer owning that rate in the band
MERGE_R = 0.05                    # chirp peaks in touching bands with rates this close are one singer


def syllable_rate_in_rows(P, r_lo, r_hi, cols, dt, Rc):
    """The pulse rate INSIDE the chirps, from the modulation spectrum of the
    level summed over the peak's rows on the loudest segment. Chirp-to-chirp
    jitter smears pulses in a fold at the chirp rate, but the pulse rate
    itself is a (broadened) spectral peak at Rs with sidebands at Rs +/- Rc.
    Returns (Rs, excess_db, sideband_db) or (nan, nan, nan)."""
    lev = 10 * np.log10(P[r_lo:r_hi + 1, cols].sum(axis=0) + 1e-20)
    lev = lev - lev.mean()
    fr, S = scipy.signal.welch(lev, fs=1 / dt, nperseg=2048)
    Sdb = 10 * np.log10(S + 1e-30)
    k = int(round(15.0 / (fr[1] - fr[0]))) | 1
    ex = Sdb - scipy.ndimage.median_filter(Sdb, size=k, mode="nearest")
    sel = (fr >= SYL_LO) & (fr <= SYL_HI)
    # ignore the chirp-rate harmonics themselves (m*Rc for m up to 100/Rc)
    harm = np.zeros(sel.sum(), bool)
    fs = fr[sel]
    for m in range(1, int(SYL_HI / Rc) + 2):
        harm |= np.abs(fs - m * Rc) < 0.6
    exs = np.where(harm, -np.inf, ex[sel])
    if not np.isfinite(exs).any():
        return np.nan, np.nan, np.nan
    j = int(np.argmax(exs))
    Rs, e = float(fs[j]), float(exs[j])
    if e < SYL_MIN_DB:
        return np.nan, e, np.nan
    # prefer the fundamental: if half the rate also carries a clear peak, the
    # maximum was the pulse rate's second harmonic
    if Rs / 2 >= SYL_LO:
        jh = int(np.argmin(np.abs(fs - Rs / 2)))
        if exs[jh] >= SYL_MIN_DB:
            Rs, e = float(fs[jh]), float(exs[jh])
    sb = [ex[sel][np.argmin(np.abs(fs - x))] for x in (Rs - Rc, Rs + Rc) if SYL_LO <= x <= SYL_HI]
    return Rs, e, float(np.mean(sb)) if sb else np.nan


SYL_CONC_MIN = 2.0                # v2 reader: the rhythm's power inside the chirps must be this many times its power between them
SYL_SMOOTH_S = 0.04               # v2 reader: smoothing that separates the chirp envelope from the syllables inside it


def syllable_rate_in_rows_v2(P, r_lo, r_hi, cols, dt, Rc, max_cand: int = 6):
    """The pulse rate inside the chirps, second version (2026-10-08, METHOD 19 AF/AG): the same row-summed level
    spectrum as syllable_rate_in_rows, but every candidate peak is tried in order of height and accepted only
    if its rhythm is CONCENTRATED IN THE CHIRPS -- its power at that rate inside the chirps (the smoothed level
    above its 70th percentile) is at least SYL_CONC_MIN times its power between them. A continuous triller
    leaking in from the rows below has the same power inside and between (ratio ~1) and is skipped, whichever
    of the two is taller; the chirper's own syllables, whether or not they were also detected as a separate
    fast peak in these rows, pass. The first reader took the single tallest peak and lost to the neighbor in
    37 of 49 chirpers of the focal night. Returns (Rs, excess_db, sideband_db, concentration) or nans."""
    lev = 10 * np.log10(P[r_lo:r_hi + 1, cols].sum(axis=0) + 1e-20)
    lev = lev - lev.mean()
    fr, S = scipy.signal.welch(lev, fs=1 / dt, nperseg=2048)
    Sdb = 10 * np.log10(S + 1e-30)
    k = int(round(15.0 / (fr[1] - fr[0]))) | 1
    ex = Sdb - scipy.ndimage.median_filter(Sdb, size=k, mode="nearest")
    sel = np.flatnonzero((fr >= SYL_LO) & (fr <= SYL_HI))
    fs, exs = fr[sel], ex[sel]
    harm = np.zeros(len(fs), bool)
    for m in range(1, int(SYL_HI / Rc) + 2):
        harm |= np.abs(fs - m * Rc) < 0.6
    ok = (exs >= SYL_MIN_DB) & ~harm
    peaks = [i for i in range(1, len(fs) - 1) if ok[i] and exs[i] >= exs[i - 1] and exs[i] >= exs[i + 1]]
    if not peaks:
        return np.nan, float(np.nanmax(np.where(harm, np.nan, exs))) if (~harm).any() else np.nan, np.nan, np.nan
    peaks = sorted(peaks, key=lambda i: -exs[i])[:max_cand]
    # the chirp gate on the same level series: slow envelope above its 70th percentile
    w = max(3, int(round(SYL_SMOOTH_S / dt)) | 1)
    sm = scipy.signal.savgol_filter(lev, w, 2)
    gate = sm > np.percentile(sm, 70)
    fast = lev - sm
    if gate.sum() < 8 or (~gate).sum() < 8:
        return np.nan, np.nan, np.nan, np.nan

    def power_at(R, mask):
        y = fast * mask; y = y - y.mean()
        frg, Sg = scipy.signal.welch(y, fs=1 / dt, nperseg=2048)
        j = int(np.argmin(np.abs(frg - R)))
        return float(Sg[max(0, j - 2):j + 3].max()) / max(1e-9, float(mask.mean()))

    def prefer_fundamental(i):
        if fs[i] / 2 >= SYL_LO:
            jh = int(np.argmin(np.abs(fs - fs[i] / 2)))
            if ok[jh]:
                return jh
        return i

    for i in peaks:
        i = prefer_fundamental(i)
        Rs, e = float(fs[i]), float(exs[i])
        conc = power_at(Rs, gate) / max(1e-12, power_at(Rs, ~gate))
        sb = [ex[sel][int(np.argmin(np.abs(fs - x)))] for x in (Rs - Rc, Rs + Rc) if SYL_LO <= x <= SYL_HI]
        sb_db = float(np.mean(sb)) if sb else np.nan
        if conc >= SYL_CONC_MIN:
            return Rs, e, sb_db, float(conc)
    return np.nan, float(exs[peaks[0]]), np.nan, np.nan


CHIRP_SMOOTH_S, PULSE_SMOOTH_S = 0.040, 0.003     # time-domain reader: the chirp envelope and the pulse envelope
CHIRP_MIN_S, CHIRP_GAP_S = 0.03, 0.02             # a chirp is >= 30 ms above threshold; gaps under 20 ms are bridged
PULSE_PROM, PULSE_MAX_HZ = 0.25, 120.0            # pulse peaks: prominence vs the chirp's range; at most 120 pulses per second
IQR_MAX_PCT, MIN_INTERVALS, MIN_CHIRPS = 35.0, 20, 5   # quality gate on the inter-pulse intervals


def syllable_rate_from_chirps(sig: np.ndarray, sr: int, F_lo: float, F_hi: float, t_a: float = 0.0, t_b: float | None = None, fs_env: float = 4000.0) -> dict:
    """The time-domain reader (MBE, 2026-10-08: "can we not recognize chirps directly and ID the syllable rate
    within"). Band-pass the audio to the chirper's band, take the envelope, and find the CHIRPS as events: the
    40 ms-smoothed envelope above a threshold between its quiet and loud levels, >= 30 ms long. Inside each chirp
    the pulses are the peaks of the 3 ms-smoothed envelope (prominence >= PULSE_PROM of the chirp's range, at most
    PULSE_MAX_HZ per second). The syllable rate is 1 / median inter-pulse interval over all chirps, with the
    interval spread (IQR as % of the median) as the quality: a continuous neighbor in the rows adds no bursts and
    cannot win, and a reading with a wide spread (pulses split or missed) is rejected. Returns a dict: Rs (nan when
    rejected), iqr_pct, pulses_per_chirp, chirp_ms, n_chirps, chirps_per_s, n_intervals."""
    a = int(max(0.0, t_a) * sr); b = len(sig) if t_b is None else int(min(len(sig) / sr, t_b) * sr)
    x = sig[a:b]
    if len(x) < sr:
        return dict(Rs=np.nan, iqr_pct=np.nan, pulses_per_chirp=np.nan, chirp_ms=np.nan, n_chirps=0, chirps_per_s=np.nan, n_intervals=0)
    lo_f, hi_f = max(100.0, F_lo), min(sr / 2 - 100.0, F_hi)
    if hi_f - lo_f < 50.0:                                       # a band at the top of the spectrum: nothing to filter
        return dict(Rs=np.nan, iqr_pct=np.nan, pulses_per_chirp=np.nan, chirp_ms=np.nan, n_chirps=0, chirps_per_s=np.nan, n_intervals=0)
    sos = scipy.signal.butter(6, [lo_f, hi_f], btype="band", fs=sr, output="sos")
    xb = scipy.signal.sosfiltfilt(sos, x)
    q = max(1, int(round(sr / fs_env))); env = scipy.signal.decimate(np.abs(scipy.signal.hilbert(xb)), q); fs = sr / q
    fast = scipy.signal.savgol_filter(env, int(PULSE_SMOOTH_S * fs) | 1, 2)
    slow = scipy.signal.savgol_filter(env, int(CHIRP_SMOOTH_S * fs) | 1, 2)
    lo_, hi_ = np.percentile(slow, 20), np.percentile(slow, 97)
    on = slow > lo_ + 0.4 * (hi_ - lo_)
    idx = np.flatnonzero(on)
    dur_s = len(x) / sr
    if not len(idx):
        return dict(Rs=np.nan, iqr_pct=np.nan, pulses_per_chirp=np.nan, chirp_ms=np.nan, n_chirps=0, chirps_per_s=0.0, n_intervals=0)
    cuts = np.flatnonzero(np.diff(idx) > int(CHIRP_GAP_S * fs))
    segs = [s_ for s_ in np.split(idx, cuts + 1) if (s_[-1] - s_[0]) / fs >= CHIRP_MIN_S]
    intervals, npulse, durs = [], [], []
    pad = int(0.01 * fs)
    for s_ in segs:
        i0, i1 = max(0, s_[0] - pad), min(len(fast), s_[-1] + pad)
        y = fast[i0:i1]
        pk, _ = scipy.signal.find_peaks(y, prominence=PULSE_PROM * (y.max() - y.min()), distance=max(1, int(fs / PULSE_MAX_HZ)))
        if len(pk) >= 2:
            intervals += list(np.diff(pk) / fs); npulse.append(len(pk)); durs.append((i1 - i0) / fs)
    out = dict(Rs=np.nan, iqr_pct=np.nan, pulses_per_chirp=float(np.median(npulse)) if npulse else np.nan, chirp_ms=1000 * float(np.median(durs)) if durs else np.nan,
               n_chirps=len(segs), chirps_per_s=len(segs) / dur_s, n_intervals=len(intervals))
    if len(intervals) >= MIN_INTERVALS and len(segs) >= MIN_CHIRPS:
        iv = np.array(intervals); med = float(np.median(iv))
        out["iqr_pct"] = 100.0 * float(np.percentile(iv, 75) - np.percentile(iv, 25)) / med
        if out["iqr_pct"] <= IQR_MAX_PCT:
            out["Rs"] = 1.0 / med
    return out


def add_chirpers(P: np.ndarray, f: np.ndarray, dt: float, A: pd.DataFrame, fold_s: float = FOLD_S):
    """Run the chirp pass on a window's spectrogram. Returns (A2, C): the
    animal table with chirp singers appended or attached, and the chirp
    table itself."""
    frc, exc = chirp_map(P, dt)
    B = find_chirp_peaks(exc, f, frc)
    crows = []
    A2 = A.copy()
    for col, default in (("chirp_rate", np.nan), ("pulses_per_chirp", np.nan), ("chirp_duty", np.nan), ("chirp_ms", np.nan), ("source", "trill")):
        if col not in A2.columns:
            A2[col] = default
    next_id = int(A2.animal.max()) + 1 if len(A2) else 0
    # chirp-rate harmonics: a peak at m x a lower peak's rate (m = 2, 3; 3 %)
    # in overlapping rows is the same chirper's harmonic, not a singer
    B = B.reset_index(drop=True)
    is_harm = np.zeros(len(B), bool)
    for i, a in B.iterrows():
        for j, c in B.iterrows():
            if j == i or c.Rc0 >= a.Rc0 * 0.97:
                continue
            m = a.Rc0 / c.Rc0
            if abs(m - round(m)) < 0.03 * round(m) and 2 <= round(m) <= 3 and a.f_lo <= c.f_hi + 100 and c.f_lo <= a.f_hi + 100:
                is_harm[i] = True
    B = B[~is_harm].reset_index(drop=True)
    # the same chirper spilling into neighbouring rows: peaks with bands within
    # 100 Hz and chirp rates within MERGE_R are one singer; keep the strongest
    drop = np.zeros(len(B), bool)
    for i, a in B.iterrows():
        for j, c in B.iterrows():
            if j <= i or drop[i] or drop[j]:
                continue
            if abs(a.Rc0 / c.Rc0 - 1) < MERGE_R and a.f_lo <= c.f_hi + 100 and c.f_lo <= a.f_hi + 100:
                drop[j if a.peak_db >= c.peak_db else i] = True
    B = B[~drop]
    for b in B.itertuples():
        r_lo, r_hi = int(b.r_lo), int(b.r_hi)
        cols = best_segment(P, r_lo, r_hi, dt, b.Rc0, fold_s)
        Rc = refine_rate(P, r_lo, r_hi, cols, dt, b.Rc0)
        F = fold(P, r_lo, r_hi, cols, dt, Rc)
        sh = chirp_shape(F, Rc)
        if sh is None:
            continue
        Rs, syl_db, sb_db = syllable_rate_in_rows(P, r_lo, r_hi, cols, dt, Rc)
        # the pulse peak belongs to THIS chirper if it carries sidebands at
        # Rs +/- Rc, or if it is clear (>= SYL_ALONE_DB) and no trill-pass singer
        # in the band already owns that rate; otherwise Rs is unknown
        if np.isfinite(Rs):
            has_sb = np.isfinite(sb_db) and sb_db >= SIDEBAND_MIN_DB
            owned = len(A2) and ((A2.F_lo <= b.f_hi + 100) & (A2.F_hi >= b.f_lo - 100) & (np.abs(A2.R / Rs - 1) < CHIRP_MATCH_R) & (A2.source == "trill")).any()
            if not has_sb and not (syl_db >= SYL_ALONE_DB and not owned):
                Rs = np.nan
        pulses = Rs * sh["chirp_ms"] / 1000.0 if np.isfinite(Rs) else np.nan
        # depth of the chirp modulation over the rows, first harmonic
        kk = np.arange(F.shape[1])
        c1 = (F * np.exp(-2j * np.pi * kk / F.shape[1])).sum(axis=1)
        depth = float(np.median(2 * np.abs(c1) / np.maximum(F.sum(axis=1), 1e-30)))
        row = dict(f_lo=b.f_lo, f_hi=b.f_hi, f_c=b.f_c, chirp_rate=Rc, chirp_db=b.peak_db, n_rows=b.n_rows,
                   Rs=Rs, syl_db=syl_db, sideband_db=sb_db, pulses_per_chirp=pulses, chirp_duty=sh["duty"], chirp_ms=sh["chirp_ms"], depth=depth,
                   fold_t0=cols.start * dt)
        sh["pulses"] = pulses
        # attach to an existing trill-pass singer with overlapping band and matching syllable rate
        matched = None
        if np.isfinite(Rs) and len(A2):
            cand = A2[(A2.F_lo <= b.f_hi + 100) & (A2.F_hi >= b.f_lo - 100) & (np.abs(A2.R / Rs - 1) < CHIRP_MATCH_R)]
            if len(cand):
                matched = int(cand.sort_values("depth", ascending=False).animal.iloc[0])
        if matched is not None:
            m = A2.animal == matched
            A2.loc[m, ["chirp_rate", "pulses_per_chirp", "chirp_duty", "chirp_ms"]] = [Rc, sh["pulses"], sh["duty"], sh["chirp_ms"]]
            row["animal"] = matched
            row["attached"] = True
        else:
            row["animal"] = next_id
            row["attached"] = False
            A2 = pd.concat([A2, pd.DataFrame([dict(
                animal=next_id, n_components=1, n_am_harmonics=0, F_lo=b.f_lo, F_hi=b.f_hi, F_c=b.f_c, F_pk=b.f_c,
                R=Rs if np.isfinite(Rs) else np.nan, onoff_cv=np.nan, on_frac=sh["duty"], depth=depth,
                components=f"{b.f_lo:.0f}-{b.f_hi:.0f}", am_harmonics="",
                chirp_rate=Rc, pulses_per_chirp=sh["pulses"], chirp_duty=sh["duty"], chirp_ms=sh["chirp_ms"], source="chirp")])],
                ignore_index=True)
            next_id += 1
        crows.append(row)
    C = pd.DataFrame(crows)
    return A2, C
