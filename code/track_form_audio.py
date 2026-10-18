"""Song form per singer-period, measured from the audio (the v9 refinement).

v8 took a singer-period's form from the detector's chirp channel, which never fires on some
audible chirpers and returns a stable but meaningless syllable rate for others (MBE,
2026-10-06: "that chirper vs triller thing needs to be refined for the next version"). Here each
singer-period is measured from its own audio in N_WIN windows of its track -- the deepest, and the
median-depth one, so a chirper that trilled in its loudest window is still caught -- with the test
that classifies all 17 checked SINA clips correctly (reference_vs_streak_grid.py):

  contrast   the band envelope (dB, smoothed over two syllable periods, slow trend removed) has a
             bout rhythm at CHIRP_LO-CHIRP_HI Hz whose spectral peak stands `contrast` times above its
             octave neighborhood -- hundreds for a chirper, 2-8 for a trill, up to ~30 for a wavering
             tree-cricket trill
  duty       share of the time the band is loud (above half its loud level)
  rhythm_hz  the bout rhythm's frequency
  harm2      the detector's syllable rate for the window is within 8 % of twice the bout rhythm (a
             chirp-rate harmonic, not a syllable rate)
  band_share power in the detected band / power in the octave around it during the loud moments:
             low for a broadband burst whose "carrier" is one slice of a wide spectrum
  locked_corr correlation, over the loudest 30 s, between the bout envelope and the strength of the
             band's modulation AT THE TRACK'S SYLLABLE RATE (the envelope band-passed around R, its
             amplitude smoothed over two periods). A chirper's syllables exist only inside its chirps,
             so the two rise and fall together (positive); a triller's syllable modulation is steady
             while a neighbor's bursts move the envelope (near zero or negative). This is what tells a
             chirper from a triller with a PERSISTENT chirping neighbor in its band, which the
             both-windows rule cannot (9951: 8102's trillers at House under the broadband bursts of
             8152, contrast 57-195 in both windows; MBE: "9951 still a mix of chirps and trills")

Per singer-period: chirp if EVERY measured window has contrast > CHIRP_CONTRAST AND locked_corr >
CHIRP_CORR; trill if every window has contrast < TRILL_CONTRAST; mixed otherwise. "Any window" was tried first and called 9 %
of O. texensis' and 11 % of 6901's singer-periods chirpers: a chirping neighbor in the same band
(4051 under 5301, 8151 under 6901) contaminates one window, rarely both. Both-windows: 5301
184 -> 29, 6901 514 -> 67, while 4001's missed chirpers hold (133 -> 117) and 4051 324 of 422. The duty and band-share tests that worked on the clean
SINA clips do not transfer to field audio (a faint trill's envelope fluctuates with the chorus;
the octave around a band holds other singers), so duty and band_share are recorded as information
only. harm2 if any window with a rhythm (contrast > TRILL_CONTRAST) has the detector's rate within
8 % of twice that rhythm. Test set (12 known singers, 2026-10-06): 8102 trillers 7-13, 7201 trillers
9-26, 4001 trillers 20 and 436 (the latter a chirper the chirp channel missed), chirpers 270-2350.

Resumable: appends to <root>/track_form_audio.csv and skips singer-periods already measured.

Usage:
  python analysis/callers/track_form_audio.py analysis/callers/runs/v2/year_nov24_nov25 \
      --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --jobs 10
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs as W  # noqa: E402

N_WIN = 2
CHIRP_LO, CHIRP_HI = 1.5, 12.0
CHIRP_CONTRAST, CHIRP_DUTY = 50.0, 0.50
CHIRP_CORR = 0.2         # test set: true chirpers +0.30 to +0.73, trillers under a persistent broadband burster -0.30 to -0.79
TRILL_CONTRAST, TRILL_DUTY = 20.0, 0.60
BROAD_SHARE = 0.5
TZ = "America/Chicago"


def windows_of(an: str, unit_roots, n: int):
    unit, track = an.rsplit(":", 1); track = int(track)
    period = "day" if unit.endswith("_day") else "night"                 # day units: night_day_global_window_animals.csv
    for root in unit_roots:
        p = os.path.join(root, unit, f"night_{period}_global_window_animals.csv")
        if os.path.exists(p):
            Wd = pd.read_csv(p, usecols=["t", "track", "R", "F_c", "F_lo", "F_hi", "F_fix", "depth", "chirp_rate"])
            g = Wd[(Wd.track == track) & Wd.F_fix.notna()].sort_values("depth", ascending=False)
            if g.empty:
                return None, None
            picks = [g.iloc[0]]
            if len(g) >= 3 and n >= 2:
                picks.append(g.iloc[len(g) // 2])
            parts = unit.split("_")
            return [dict(recorder=parts[0] + "_" + parts[1], site=parts[2], t=pd.Timestamp(b.t).tz_convert(TZ), R=float(b.R) if pd.notna(b.R) else np.nan,
                         F_lo=float(b.F_lo), F_hi=float(b.F_hi), F_c=float(b.F_c), Rc=float(b.chirp_rate) if pd.notna(b.chirp_rate) else np.nan) for b in picks], len(g)
    return None, None


def measure_window(w: dict) -> dict:
    sig, sr, _, _ = W.read_window(w["recorder"], w["site"], w["t"], 300.0)
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2; dt = 128 / sr
    rows = (f >= w["F_lo"]) & (f <= w["F_hi"])
    if not rows.any():
        rows = np.abs(f - w["F_c"]) < 100
    band = 10 * np.log10(P[rows].sum(axis=0) + 1e-20)
    # the loudest 30 s for the band
    n30 = int(30 / dt)
    if len(band) > n30:
        k = int(np.argmax(np.convolve(band, np.ones(n30) / n30, "valid"))); sl = slice(k, k + n30)
    else:
        sl = slice(0, len(band))
    env = band[sl]
    R = w["R"] if np.isfinite(w["R"]) and w["R"] > 0 else 30.0
    # smooth over ONE syllable period (two periods washed out a bout rhythm at R/2 -- a chirper whose detector
    # rate is twice its chirp rate, the 7201 impostor at 15 Hz / 7.9 chirps per second)
    wn = max(3, int(round(1.0 / R / dt))); env_s = np.convolve(env, np.ones(wn) / wn, "same")
    k5 = 2 * int(5.0 / dt) + 1
    slow = scipy.signal.medfilt(env_s, k5) if len(env_s) > k5 else np.full_like(env_s, env_s.mean())
    x = env_s - slow; x = x - x.mean()
    F = np.fft.rfftfreq(len(x), dt); S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    sel = (F >= CHIRP_LO) & (F <= CHIRP_HI) & (F < 0.7 * R)
    if sel.any():
        f0 = float(F[sel][np.argmax(S[sel])]); nb = (F >= f0 / 2) & (F <= 2 * f0) & (np.abs(F / f0 - 1) > 0.15)
        contrast = float(S[sel].max() / np.median(S[nb]))
    else:
        f0, contrast = np.nan, 0.0
    lo_db, hi_db = np.percentile(env_s, 10), np.percentile(env_s, 95)
    duty = float((env_s - lo_db > 0.5 * (hi_db - lo_db)).mean())
    harm2 = bool(np.isfinite(f0) and np.isfinite(w["R"]) and contrast > TRILL_CONTRAST and abs(w["R"] / (2 * f0) - 1) < 0.08)
    # syllable-modulation strength over time: the raw band envelope band-passed around R (+/- 15 %), rectified, smoothed over two periods
    locked_corr = np.nan
    if np.isfinite(w["R"]) and w["R"] > 0 and R < 0.45 / dt:
        env0 = env - env.mean()
        sos = scipy.signal.butter(2, [0.85 * R, min(1.15 * R, 0.49 / dt)], btype="band", fs=1 / dt, output="sos")
        bp = scipy.signal.sosfiltfilt(sos, env0)
        w2 = max(3, int(round(2.0 / R / dt))); strength = np.convolve(np.abs(bp), np.ones(w2) / w2, "same")
        if strength.std() > 0 and x.std() > 0:
            locked_corr = float(np.corrcoef(strength, x)[0, 1])
    # band share during the loud moments: band power over the octave around the band
    loud = np.flatnonzero(env_s - lo_db > 0.5 * (hi_db - lo_db)) + sl.start
    oct_rows = (f >= w["F_lo"] / 1.41) & (f <= w["F_hi"] * 1.41)
    if len(loud):
        Pl = P[:, loud]
        quiet = np.percentile(P, 10, axis=1, keepdims=True)
        above = np.clip(Pl - quiet, 0, None)
        share = float(above[rows].sum() / (above[oct_rows].sum() + 1e-20))
    else:
        share = np.nan
    return dict(contrast=contrast, duty=duty, rhythm_hz=f0, harm2=harm2, band_share=share, locked_corr=locked_corr, R_det=w["R"], Rc_det=w["Rc"])


def _one(args):
    an, unit_roots = args
    try:
        ws, n_win = windows_of(an, unit_roots, N_WIN)
        if not ws:
            return dict(an=an, status="no window")
        ms = [measure_window(w) for w in ws]
        out = dict(an=an, status="ok", n_win_track=n_win, n_measured=len(ms),
                   contrast_max=max(m["contrast"] for m in ms), contrast_min=min(m["contrast"] for m in ms),
                   duty_min=min(m["duty"] for m in ms), duty_max=max(m["duty"] for m in ms),
                   rhythm_hz=float(np.nanmedian([m["rhythm_hz"] for m in ms])), harm2=any(m["harm2"] for m in ms),
                   band_share=float(np.nanmedian([m["band_share"] for m in ms])),
                   corr_min=float(np.nanmin([m["locked_corr"] for m in ms])) if any(np.isfinite(m["locked_corr"]) for m in ms) else np.nan)
        chirp = out["contrast_min"] > CHIRP_CONTRAST and (not np.isfinite(out["corr_min"]) or out["corr_min"] > CHIRP_CORR)
        trill = out["contrast_max"] < TRILL_CONTRAST
        out["form_audio"] = "chirp" if chirp else ("trill" if trill else "mixed")
        out["broad"] = bool(np.isfinite(out["band_share"]) and out["band_share"] < BROAD_SHARE)
        return out
    except SystemExit as e:
        return dict(an=an, status=f"no audio: {str(e)[:40]}")
    except Exception as e:  # noqa: BLE001
        return dict(an=an, status=f"{type(e).__name__}: {str(e)[:40]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--unit-roots", nargs="+", required=True); ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=None, help="singer-period ids to measure (test)")
    ap.add_argument("--relabel", action="store_true", help="re-derive form_audio from the stored contrasts without re-reading audio")
    ap.add_argument("--recheck-chirps", action="store_true", help="re-measure only the singer-periods currently labeled chirp (adds locked_corr), then relabel")
    ap.add_argument("--ids-from", default=None, help="table whose 'an' column lists the singer-periods (default temperature_animal_nights.csv)")
    ap.add_argument("--out", default=None, help="output table (default <root>/track_form_audio.csv)")
    args = ap.parse_args()
    A = pd.read_csv(args.ids_from or os.path.join(args.root, "temperature_animal_nights.csv"), usecols=["an"]).drop_duplicates()
    out = args.out or os.path.join(args.root, "track_form_audio.csv")
    if args.relabel:
        D = pd.read_csv(out); ok = D.status == "ok"
        corr_ok = ~(D.loc[ok, "corr_min"] <= CHIRP_CORR) if "corr_min" in D.columns else True
        D.loc[ok, "form_audio"] = np.where((D.loc[ok, "contrast_min"] > CHIRP_CONTRAST) & corr_ok, "chirp", np.where(D.loc[ok, "contrast_max"] < TRILL_CONTRAST, "trill", "mixed"))
        D.to_csv(out, index=False, float_format="%.4f"); print("relabeled:", D.loc[ok, "form_audio"].value_counts().to_dict()); return
    done = pd.read_csv(out) if os.path.exists(out) and args.only is None else pd.DataFrame(columns=["an"])
    if args.recheck_chirps:
        redo = set(done[(done.status == "ok") & (done.form_audio == "chirp")].an)
        done = done[~done.an.isin(redo)]; todo = sorted(redo)
        print(f"re-measuring {len(todo):,} chirp-labeled singer-periods for locked_corr", flush=True)
    else:
        todo = args.only or [a for a in A.an if a not in set(done.an)]
    print(f"{len(todo):,} singer-periods to measure ({len(done):,} done)", flush=True)
    rows = done.to_dict("records") if args.only is None else []
    with Pool(args.jobs) as pool:
        for k, d in enumerate(pool.imap_unordered(_one, [(an, args.unit_roots) for an in todo], chunksize=4), 1):
            rows.append(d)
            if k % 500 == 0:
                pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f"); print(f"  {k}/{len(todo)}", flush=True)
    D = pd.DataFrame(rows)
    if args.recheck_chirps:
        ok = D.status == "ok"
        corr_ok = ~(D.loc[ok, "corr_min"] <= CHIRP_CORR)
        D.loc[ok, "form_audio"] = np.where((D.loc[ok, "contrast_min"] > CHIRP_CONTRAST) & corr_ok, "chirp", np.where(D.loc[ok, "contrast_max"] < TRILL_CONTRAST, "trill", "mixed"))
    if args.only is None:
        D.to_csv(out, index=False, float_format="%.4f")
    ok = D[D.status == "ok"]
    print(f"{len(ok):,} measured; form {ok.form_audio.value_counts().to_dict()}; broad {int(ok.broad.sum())}; harm2 {int(ok.harm2.sum())}")
    if args.only:
        pd.set_option("display.width", 220); print(D[["an", "status", "contrast_max", "contrast_min", "corr_min", "rhythm_hz", "harm2", "form_audio"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
