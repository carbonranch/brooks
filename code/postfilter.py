"""Post-detection flags on window-animals, applied between detection and
linking. Nothing is deleted: each flag is a column, and `keep` says whether a
detection goes to the linker. The dropped rows are written back into the
linker's window-animals output with track = -1 so the record stays complete.

Flags
  harm_of, harm_m : the detection sits at 2x or 3x (HARM_F_TOL) the carrier
                    of another detection in the same window with the same
                    rate (HARM_R_TOL on the syllable rate, or CHIRP_TOL on the
                    chirp rate when neither has a syllable rate): a harmonic
                    reported as its own animal. harm_of is that animal's id
                    (the lowest-carrier root if the chain is longer), else -1.
                    Over the sampled v2 November nights 18 % of detections
                    above 12 kHz had such a partner (old detector 7 %).
  sub_of          : the detection sits at 1/2 or 1/3 of the carrier of a same-rate
                    detection in the same window and is >= HARM_DB weaker than it:
                    the louder animal's shadow in a lower band (flag_harmonics).
  line            : the syllable rate lies on a fixed-rate line of this
                    recorder-month (fixed_rate_lines.py): a 0.25 Hz rate bin
                    holding an outsized share of the recorder's detections
                    across many windows and many carrier bands, with no
                    temperature drift. Recorder 8604 at House ran at 59.25 Hz
                    over 2-7 kHz from November 2024 to January 2025 (60 % of
                    its January detections), with weaker lines at 19.0 and
                    37.9 Hz; Copse showed the same 59.25 Hz faintly in
                    December. Machines, not animals.
  frag            : informational only (not used in keep). Deep, brief,
                    fragmented modulation -- on_frac < FRAG_ON, n_runs >=
                    FRAG_RUNS and on_frac_map < FRAG_MAP, or depth > FRAG_DEPTH.
                    On Nov-2024 night detections that persist into a
                    neighbouring window the rule fires on 4 %; on isolated
                    Feb-2025 daytime detections on 36 %. Too weak to drop on,
                    so the non-insect residual is handled by persistence
                    (tracks of >= MIN_WINDOWS) rather than per-window features.
  keep            : harm_of < 0 and sub_of < 0 and not line.

Usage
  import postfilter
  A, P = night_animals.load(night_dir, "_night")
  A = postfilter.apply(A, P, night_dir=night_dir, lines=postfilter.load_lines(run_root))
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

HARM_F_TOL = 0.08     # relative tolerance on F_c = m * F_c(root), m in (2, 3)
HARM_DB = -6.0        # the harmonic (or shadow) must be at least this much weaker than the fundamental
HARM_R_TOL = 0.03     # relative syllable-rate agreement
CHIRP_TOL = 0.06      # relative chirp-rate agreement (chirp-only pairs)
LINE_TOL = 0.50       # Hz around a fixed-rate line (the House motor reads 59.2 +/- 0.2; at 0.2 its 58.9 Hz edge survived as two sparse tracks)
FRAG_ON, FRAG_RUNS, FRAG_MAP, FRAG_DEPTH = 0.10, 4, 0.25, 1.2

UNIT_RE = re.compile(r"BARLT_(\d+)_(.+)_(\d{8})_(night|day)$")


def animal_power(P: pd.DataFrame) -> pd.Series:
    """Band power per (t, animal) from the parts files' power profiles."""
    pw = P.prof.map(lambda s: float(np.sum(np.array(str(s).split(), dtype=float))))
    keys = ["t", "animal"] if "t" in P.columns else ["animal"]
    return pw.groupby([P[k] for k in keys]).sum()


def _same_rate(Ri, Ci, Rj, Cj) -> bool:
    if np.isfinite(Rj) and np.isfinite(Ri):
        return abs(Rj - Ri) <= HARM_R_TOL * Ri
    if not np.isfinite(Rj) and not np.isfinite(Ri) and np.isfinite(Cj) and np.isfinite(Ci):
        return abs(Cj - Ci) <= CHIRP_TOL * Ci
    return False


def flag_harmonics(A: pd.DataFrame, P: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add harm_of / harm_m and sub_of. Works on one window or many (grouped by t).

    Two same-window detections at carriers in a 2:1 or 3:1 ratio with the same rate
    are one animal. The louder is the fundamental: on 2024-11-01 NorthSide 8051 the
    true harmonics (3.8 -> 7.9 and 12.7 kHz) were 30-50 dB below their root, while
    a 2.1 kHz detection at the rate of the deep 4.2 kHz triller was 10-35 dB weaker
    than it -- the triller's shadow in the half-frequency band, not a fundamental.
    So the upper member is harm_of the lower only when it is >= HARM_DB weaker, and
    the lower member is sub_of the upper when it is the weaker one. Without P
    (no power) only the upper member is flagged, on ratio and rate alone."""
    A = A.copy()
    groups = A.groupby("t", sort=False).indices.values() if "t" in A.columns else [np.arange(len(A))]
    F_all = A.F_c.values.astype(float)
    R_all = A.R.values.astype(float)
    C_all = A.chirp_rate.values.astype(float) if "chirp_rate" in A.columns else np.full(len(A), np.nan)
    ids = A.animal.values.astype(int) if "animal" in A.columns else np.arange(len(A))
    pw_all = np.full(len(A), np.nan)
    if P is not None and len(P):
        pw = animal_power(P)
        if "t" in A.columns:   # iterate the Series, not .values: numpy datetime64 drops the tz and misses the index
            pw_all = np.array([pw.get((t, int(a)), np.nan) for t, a in zip(A.t, ids)], dtype=float)
        else:
            pw_all = np.array([pw.get(a, np.nan) for a in ids], dtype=float)
    ratio_max = 10 ** (HARM_DB / 10)
    harm_of = np.full(len(A), -1, dtype=int)
    harm_m = np.zeros(len(A), int)
    sub_of = np.full(len(A), -1, dtype=int)
    for idx in groups:
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        order = idx[np.argsort(F_all[idx])]           # ascending carrier
        root = {i: i for i in order}
        for j in order:
            Fj, Rj, Cj, pj = F_all[j], R_all[j], C_all[j], pw_all[j]
            for i in order:
                if F_all[i] >= Fj:
                    break
                m = Fj / F_all[i]
                mm = 2 if abs(m - 2) < abs(m - 3) else 3
                if abs(m / mm - 1) > HARM_F_TOL or not _same_rate(R_all[i], C_all[i], Rj, Cj):
                    continue
                pi = pw_all[i]
                if np.isfinite(pi) and np.isfinite(pj) and pj > 0 and pi > 0:
                    if pj <= pi * ratio_max:          # j is the weak upper member: a harmonic
                        pass
                    elif pi <= pj * ratio_max:        # i is the weak lower member: a shadow
                        if sub_of[i] < 0 and harm_of[i] < 0:
                            sub_of[i] = ids[j]
                        continue
                    else:                             # comparable levels: leave both
                        continue
                r = root[i]
                root[j] = r
                harm_of[j] = ids[r]
                harm_m[j] = mm if r == i else 0       # 0: harmonic of a harmonic; root recorded
                break                                 # lowest-carrier match first
    A["harm_of"] = harm_of
    A["harm_m"] = harm_m
    A["sub_of"] = sub_of
    return A


def load_lines(run_root: str) -> pd.DataFrame | None:
    p = os.path.join(run_root, "fixed_rate_lines.csv")
    if not os.path.exists(p):
        return None
    L = pd.read_csv(p, dtype={"rec": str, "month": str})
    return L if len(L) else None


def flag_lines(A: pd.DataFrame, rec: str, month: str, lines: pd.DataFrame | None) -> pd.Series:
    line = pd.Series(False, index=A.index)
    if lines is None:
        return line
    L = lines[(lines.rec == rec) & (lines.month == month)]
    if L.empty:
        return line
    R = A.R.values.astype(float)
    hit = np.zeros(len(A), bool)
    for r0 in L.R.values.astype(float):
        hit |= np.abs(R - r0) <= LINE_TOL
    return pd.Series(hit, index=A.index)


def flag_frag(A: pd.DataFrame) -> pd.Series:
    need = {"on_frac", "n_runs", "on_frac_map", "depth"}
    if not need <= set(A.columns):
        return pd.Series(False, index=A.index)
    f = ((A.on_frac.astype(float) < FRAG_ON) & (A.n_runs.astype(float) >= FRAG_RUNS) & (A.on_frac_map.astype(float) < FRAG_MAP)) \
        | (A.depth.astype(float) > FRAG_DEPTH)
    return f.fillna(False)


def unit_rec_month(night_dir: str) -> tuple[str, str]:
    m = UNIT_RE.match(os.path.basename(os.path.normpath(night_dir)))
    return (m.group(1), m.group(3)[:6]) if m else ("", "")


def apply(A: pd.DataFrame, P: pd.DataFrame | None = None, night_dir: str | None = None, lines: pd.DataFrame | None = None,
          rec: str | None = None, month: str | None = None) -> pd.DataFrame:
    """All flags + keep. P: the parts table (night_animals.load) for band power.
    rec/month default to the night_dir name."""
    if night_dir is not None and (rec is None or month is None):
        rec, month = unit_rec_month(night_dir)
    A = flag_harmonics(A, P)
    A["line"] = flag_lines(A, rec or "", month or "", lines)
    A["frag"] = flag_frag(A)
    A["keep"] = (A.harm_of < 0) & (A.sub_of < 0) & ~A.line
    return A
