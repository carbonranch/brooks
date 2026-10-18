"""Chirp (bout) rate of every SINA reference clip, so the chirp types can be matched to references the way the trill
types are (MBE, 2026-10-13: "and no chirpers?").

The test is the detector's bout test as applied to clips in reference_vs_streak_grid.reference_cell: the clip's band
(f_lo..f_hi from sina_reference_measurements, remeasured values where they exist) summed to a dB envelope, smoothed over
two syllable periods, the slow trend (5 s running median) removed; its spectrum between CHIRP_LO and CHIRP_HI Hz has a
peak that stands `contrast` times above the median of its octave neighborhood. A clip is a chirper when contrast >
CHIRP_CONTRAST, and its chirp rate is the peak's frequency. The whole clip is used (not only the loudest half second);
duty is the share of the time the band is loud.

Writes <root>/sina/sina_reference_chirp.csv: wav, species, T, F_c, f_lo, f_hi, R (syllable rate as measured), chirp_rate,
contrast, duty, is_chirp.

Usage:
  python analysis/callers/sina_chirp_rates.py analysis/callers/runs/v3/year_nov24_nov25
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf

CHIRP_CONTRAST = 50.0
CHIRP_LO, CHIRP_HI = 1.5, 12.0


def bout_test(fn: str, lo: float, hi: float, R: float) -> dict:
    sig, sr = sf.read(fn, dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != 44100:
        sig = scipy.signal.resample_poly(sig, 44100, sr).astype("float32"); sr = 44100
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2; dt = 128 / sr
    rows = (f >= lo) & (f <= hi)
    env = 10 * np.log10(P[rows].sum(axis=0) + 1e-20)
    Ruse = R if np.isfinite(R) and R > 0 else 40.0
    w = max(3, int(round(2.0 / Ruse / dt))); env = np.convolve(env, np.ones(w) / w, "same")
    slow = scipy.signal.medfilt(env, 2 * int(5.0 / dt) + 1) if len(env) > 2 * int(5.0 / dt) + 1 else env.mean()
    x = env - slow; x = x - x.mean()
    F = np.fft.rfftfreq(len(x), dt); S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    band = (F >= CHIRP_LO) & (F <= CHIRP_HI) & (F < Ruse / 3)
    if band.any():
        f0 = float(F[band][np.argmax(S[band])]); nb = (F >= f0 / 2) & (F <= 2 * f0) & (np.abs(F / f0 - 1) > 0.15)
        contrast = float(S[band].max() / np.median(S[nb]))
    else:
        f0, contrast = np.nan, 0.0
    lo_db, hi_db = np.percentile(env, 10), np.percentile(env, 95)
    duty = float((env - lo_db > 0.5 * (hi_db - lo_db)).mean())
    return dict(chirp_rate=f0, contrast=contrast, duty=duty, is_chirp=bool(contrast > CHIRP_CONTRAST), dur_s=len(sig) / sr)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); args = ap.parse_args()
    sdir = os.path.join(args.root, "sina")
    M = pd.read_csv(os.path.join(sdir, "sina_reference_measurements.csv"))
    rm = os.path.join(sdir, "sina_reference_remeasured.csv")
    if os.path.exists(rm):
        X = pd.read_csv(rm).drop_duplicates("wav").set_index("wav")
        for a, b in (("F_c", "F_use"), ("f_lo", "f_lo_use"), ("f_hi", "f_hi_use")):
            M[a] = X[b].reindex(M.wav).to_numpy()
            M[a] = M[a].where(M[a].notna(), pd.read_csv(os.path.join(sdir, "sina_reference_measurements.csv"))[a])
    M["species"] = M.name.str.split("|").str[0].str.strip()
    rows = []
    for m in M.itertuples():
        fn = os.path.join(sdir, "audio", m.wav)
        if not os.path.exists(fn) or not np.isfinite(m.f_lo):
            rows.append(dict(wav=m.wav, species=m.species, T=m.T, F_c=m.F_c, f_lo=m.f_lo, f_hi=m.f_hi, R=m.R, status="no band" if os.path.exists(fn) else "no audio")); continue
        try:
            d = bout_test(fn, m.f_lo, m.f_hi, m.R)
            rows.append(dict(wav=m.wav, species=m.species, T=m.T, F_c=m.F_c, f_lo=m.f_lo, f_hi=m.f_hi, R=m.R, status="ok", **d))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(wav=m.wav, species=m.species, T=m.T, F_c=m.F_c, f_lo=m.f_lo, f_hi=m.f_hi, R=m.R, status=f"{type(e).__name__}: {str(e)[:50]}"))
    D = pd.DataFrame(rows).drop_duplicates("wav")
    D.to_csv(os.path.join(sdir, "sina_reference_chirp.csv"), index=False, float_format="%.3f")
    ok = D[D.status == "ok"]
    print(f"{len(D)} clips, {len(ok)} measured, {int(ok.is_chirp.sum())} chirpers (contrast > {CHIRP_CONTRAST:.0f}x)")
    pd.set_option("display.width", 200)
    print(ok[ok.is_chirp].sort_values("F_c")[["wav", "species", "T", "F_c", "R", "chirp_rate", "contrast", "duty"]].to_string(index=False))


if __name__ == "__main__":
    main()
