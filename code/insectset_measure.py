"""Run the window analysis on the InsectSet459 clips of North American Orthoptera (insectset_fetch.py), the way the SINA
reference clips were measured (sina_reference.analyse): one measurement per clip of carrier, band, syllable rate, depth,
chirp rate (detector and bout test), with the species label and the recordist's temperature where given.

The fundamental is the deepest non-harmonic component at or above 1.5 kHz, preferring components within 0.35 log units of
the species' SINA page frequency where SINA states one. Clips shorter than 3 s are skipped; longer than 120 s are cut.

Writes <root>/measurements.csv.

Usage:
  python analysis/callers/insectset_measure.py analysis/callers/runs/insectset459 --jobs 8
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

os.environ.setdefault("BROOKS_CHIRP", "1")      # the chirp pass: syllable rate inside the chirps + chirp rate (chirp_channel.add_chirpers)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs2 as W  # noqa: E402
from sina_chirp_rates import bout_test  # noqa: E402

SONGS = "analysis/callers/runs/v3/year_nov24_nov25/sina/sina_songs.csv"
MIN_S, MAX_S = 3.0, 120.0


def _one(args):
    fn, page_khz = args
    out = dict(file_name=os.path.basename(fn), status="ok")
    try:
        info = sf.info(fn)
        frames = int(min(info.duration, MAX_S) * info.samplerate)
        sig, sr = sf.read(fn, frames=frames, dtype="float32")
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        dur = len(sig) / sr
        out.update(dur_s=round(dur, 1), sr=sr)
        if dur < MIN_S:
            out["status"] = "short"; return out
        X = W.analyse_signal(sig, sr, dur, fmax=min(22050.0, sr / 2), fold_s=min(30.0, dur))
        parts, A = X["parts"], X["A"]
        out.update(n_blobs=len(X["B"]), n_parts=len(parts), n_animals=len(A))
        if A.empty:
            out["status"] = "no animal"; return out
        cand = parts[(parts.am_harm_of < 0) & (parts.f_c >= 1500)].copy()
        if cand.empty:
            out["status"] = "no component >= 1.5 kHz"; return out
        pool = cand
        if page_khz is not None and np.isfinite(page_khz):
            cand["dk"] = np.abs(np.log(cand.f_c / (1000 * page_khz)))
            near = cand[cand.dk < 0.35]
            if len(near):
                pool = near
        fund = pool.sort_values("depth", ascending=False).iloc[0]
        a = A[A.animal == fund.animal].iloc[0]
        # a chirper's strongest peak is its chirp rate (< 8 Hz): the syllable rate is what the chirp pass read inside the chirps (A.R), the
        # chirp rate is the animal's chirp_rate; a trill keeps the part's rate
        R_part = float(fund.R); R_syl = float(a.R) if ("R" in A.columns and pd.notna(a.R) and a.R >= 8) else (R_part if R_part >= 8 else np.nan)
        out.update(F_c=float(fund.f_c), f_lo=float(fund.f_lo), f_hi=float(fund.f_hi), R=R_syl, R_part=R_part, depth=float(fund.depth), on_frac=float(fund.on_frac),
                   det_chirp_rate=float(a.chirp_rate) if "chirp_rate" in A.columns and pd.notna(a.chirp_rate) else (R_part if R_part < 8 else np.nan),
                   bout_rate=float(a.bout_rate) if "bout_rate" in A.columns and pd.notna(a.bout_rate) else np.nan,
                   n_components=int(a.n_components) if "n_components" in A.columns else np.nan)
        try:
            b = bout_test(fn, float(fund.f_lo), float(fund.f_hi), R_syl if np.isfinite(R_syl) else 40.0)
            out.update(chirp_rate=b["chirp_rate"], bout_contrast=b["contrast"], bout_duty=b["duty"], is_chirp=b["is_chirp"])
        except Exception as e:  # noqa: BLE001
            out["bout_error"] = f"{type(e).__name__}"
        return out
    except Exception as e:  # noqa: BLE001
        out["status"] = f"{type(e).__name__}: {str(e)[:60]}"; return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sel = pd.read_csv(os.path.join(args.root, "selection.csv"))
    songs = pd.read_csv(SONGS).set_index("page").kHz_first
    adir = os.path.join(args.root, "raw", "audio")
    sel = sel[sel.file_name.map(lambda f: os.path.exists(os.path.join(adir, f)))]
    if args.limit:
        sel = sel.head(args.limit)
    jobs = [(os.path.join(adir, r.file_name), float(songs.get(r.page, np.nan)) if isinstance(r.page, str) else None) for r in sel.itertuples()]
    print(f"{len(jobs)} clips", flush=True)
    with Pool(args.jobs) as pool:
        res = pool.map(_one, jobs, chunksize=4)
    M = pd.DataFrame(res).merge(sel[["file_name", "species", "page", "n_AR", "n_near", "temperature", "subset", "license", "observation"]], on="file_name", how="left")
    M.to_csv(os.path.join(args.root, "measurements.csv"), index=False, float_format="%.4f")
    ok = M[M.status == "ok"]
    print(f"{len(ok)} of {len(M)} clips measured; with a syllable rate {ok.R.notna().sum()}; chirpers by bout test {int(ok.is_chirp.fillna(False).sum())}; "
          f"with temperature {ok.temperature.notna().sum()}")
    print(M.status.value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
