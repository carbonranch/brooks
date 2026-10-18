"""Re-measure the carrier band of every SINA reference clip from the clip's own long-term spectrum.

The window analysis sometimes hands a reference clip a band at the lower end of a glide or
at a sideband (E. melodius: 4.9 kHz measured against a 6.4 kHz page and a visible 6-7.5 kHz
song; O. nigricornis: one 2.2-7.7 kHz band over a clipped recording's harmonic comb). Here
the carrier is the peak of the clip's mean level above quiet across rows, searched within
0.5-2x the page's dominant frequency when the page gives one, and the band is the contiguous
rows within BAND_DB of that peak. The spectral peak has its own failure (a loud clip's 2x
harmonic, a katydid's full band), so the page's dominant frequency arbitrates: the peak replaces
the detector's carrier only when the detector is off the page by more than PAGE_TOL and the peak
is within it. Otherwise the detector's carrier stands and only its band is re-drawn as the
contiguous rows within BAND_DB of the carrier's row (O. nigricornis' 2.2-7.7 kHz band over a
clipped recording becomes the fundamental's). Writes <root>/sina/sina_reference_remeasured.csv
with F_use, f_lo_use, f_hi_use and `source` (detector / peak). The syllable rate is left as the
detector measured it.

Usage:
  python analysis/callers/sina_remeasure.py analysis/callers/runs/v2/year_nov24_nov25
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf

BAND_DB = 6.0
TOL = 0.10
PAGE_TOL = 0.15


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); args = ap.parse_args()
    sdir = os.path.join(args.root, "sina")
    M = pd.read_csv(os.path.join(sdir, "sina_reference_measurements.csv"))
    rows = []
    for m in M.itertuples():
        fn = os.path.join(sdir, "audio", m.wav)
        if not os.path.exists(fn):
            continue
        sig, sr = sf.read(fn, dtype="float32")
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        if sr != 44100:
            sig = scipy.signal.resample_poly(sig, 44100, sr).astype("float32"); sr = 44100
        f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
        L = 10 * np.log10(np.abs(Z) ** 2 + 1e-20)
        D = L - np.percentile(L, 10, axis=1, keepdims=True)
        prof = np.percentile(D, 90, axis=1)                       # the row's loud level above its quiet level
        prof = np.convolve(prof, np.ones(3) / 3, "same")
        lo, hi = (500.0 * m.sina_kHz, 2000.0 * m.sina_kHz) if pd.notna(m.sina_kHz) else (1500.0, 20000.0)
        sel = np.flatnonzero((f >= lo) & (f <= hi))
        if not len(sel):
            continue
        k = sel[np.argmax(prof[sel])]; pk = prof[k]
        a = k
        while a > 0 and prof[a - 1] >= pk - BAND_DB:
            a -= 1
        b = k
        while b < len(f) - 1 and prof[b + 1] >= pk - BAND_DB:
            b += 1
        F_peak = float(f[k]); f_lo2, f_hi2 = float(f[a]), float(f[b])
        off = float(np.log(F_peak / m.F_c)) if pd.notna(m.F_c) and m.F_c > 0 else np.nan
        page = 1000.0 * m.sina_kHz if pd.notna(m.sina_kHz) else np.nan
        det_off_page = abs(np.log(m.F_c / page)) if np.isfinite(page) and pd.notna(m.F_c) else 0.0
        peak_on_page = abs(np.log(F_peak / page)) <= PAGE_TOL if np.isfinite(page) else False
        if det_off_page > PAGE_TOL and peak_on_page:
            F_use, lo_use, hi_use, source = F_peak, f_lo2, f_hi2, "peak"
        else:
            # the detector's carrier stands; its band re-drawn around the carrier's row, within x/1.5
            kc = int(np.argmin(np.abs(f - m.F_c))); pc = prof[kc]
            a2 = kc
            while a2 > 0 and prof[a2 - 1] >= pc - BAND_DB and f[a2 - 1] >= m.F_c / 1.5:
                a2 -= 1
            b2 = kc
            while b2 < len(f) - 1 and prof[b2 + 1] >= pc - BAND_DB and f[b2 + 1] <= m.F_c * 1.5:
                b2 += 1
            F_use, lo_use, hi_use, source = float(m.F_c), float(f[a2]), float(f[b2]), "detector"
        rows.append(dict(wav=m.wav, name=m.name, T=m.T, sina_kHz=m.sina_kHz, F_c_det=m.F_c, f_lo_det=m.f_lo, f_hi_det=m.f_hi, R=m.R,
                         F_peak=F_peak, f_lo2=f_lo2, f_hi2=f_hi2, peak_db=float(pk), off_log=off, F_use=F_use, f_lo_use=lo_use, f_hi_use=hi_use, source=source))
    X = pd.DataFrame(rows)
    X.to_csv(os.path.join(sdir, "sina_reference_remeasured.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    rp = X[X.source == "peak"]
    print(f"{len(X)} clips; {len(rp)} take the spectral peak (detector off the page by > {PAGE_TOL:.0%}, peak within it):")
    print(rp[["name", "wav", "T", "sina_kHz", "F_c_det", "F_use", "f_lo_use", "f_hi_use", "R"]].round(2).to_string(index=False))
    nb = X[(X.source == "detector") & ((X.f_lo_use / X.f_lo_det > 1.15) | (X.f_hi_det / X.f_hi_use > 1.15))]
    print(f"{len(nb)} keep the detector's carrier with a narrower band, e.g.:")
    print(nb[["name", "wav", "F_c_det", "f_lo_det", "f_hi_det", "f_lo_use", "f_hi_use"]].round(0).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
