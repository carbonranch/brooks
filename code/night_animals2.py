"""Night-level animals from a sweep of per-window inventories.

A window's animal is a rate plus a set of component bands. Across windows the
same animal keeps its rate to ~1% and keeps at least one of its bands, while
the band that happens to be lowest, and the number of faint components, come
and go. So a night-animal is defined by rate AND the union of its component
bands, and a window's animal joins it if the rate matches and any band
overlaps. A night-animal may be missed for a few windows and picked up again.

The carrier trajectory is read from ONE reference band per night-animal (the
band most often present), so a window in which the 6.7 kHz component is faint
and the 13 kHz harmonic is the lowest visible does not register as a 6 kHz
jump in carrier.

Usage:
    python analysis/callers/night_animals.py analysis/callers/runs/night_8662 --tag _night [--min-windows 6]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R_TOL = 0.015        # rate match between windows a few minutes apart
F_PAD = 100.0        # Hz slack on band overlap
MAX_GAP = 4          # windows an animal may be missed before its track closes
STITCH_H = 0.75      # hours; tracks ending/starting within this are joined if rate and band agree
STITCH_R_TOL = 0.03
BAND_MEMORY = 6      # a track's bands are the union over its last this-many windows (30 min), not its whole life
MAX_COMP_BANDS = 6   # a window-animal with more components than this is a broadband burst; only its lowest band is remembered
DF = 44100 / 512     # row spacing of the stored profiles
FIX_PAD = 150.0      # Hz beyond the reference band for the fixed-window carrier
TZ = os.environ.get("BROOKS_TZ", "America/Chicago")


def bands(s: str) -> list[tuple[float, float]]:
    out = []
    for c in str(s).split(";"):
        if "-" in c:
            lo, hi = c.split("-")
            out.append((float(lo), float(hi)))
    return out


def overlap(b1, b2) -> bool:
    return any(lo1 <= hi2 + F_PAD and lo2 <= hi1 + F_PAD for lo1, hi1 in b1 for lo2, hi2 in b2)


def load(run_dir: str, tag: str):
    an, pa = [], []
    for p in sorted(glob.glob(os.path.join(run_dir, f"*_????????_????{tag}_animals.csv"))):
        m = re.search(r"_(\d{8})_(\d{4})", os.path.basename(p))
        t = pd.Timestamp(f"{m.group(1)} {m.group(2)[:2]}:{m.group(2)[2:]}", tz=TZ)
        A = pd.read_csv(p)
        if A.empty:
            continue
        A["t"] = t
        an.append(A)
        P = pd.read_csv(p.replace("_animals.csv", "_parts.csv"))
        P["t"] = t
        pa.append(P)
    if not an:  # a silent period (a frozen January day): every window empty
        return pd.DataFrame(), pd.DataFrame()
    A = pd.concat(an, ignore_index=True).sort_values(["t", "depth"], ascending=[True, False])
    return A, pd.concat(pa, ignore_index=True)


def link(A: pd.DataFrame) -> pd.DataFrame:
    """Window by window, assign each window-animal to an open track: rate
    within R_TOL of the track's latest rate and any band overlapping the
    track's union of bands. Closest rate first, so two same-band animals a few
    percent apart in rate do not swap. A track carries its union of bands."""
    A = A.copy()
    A["track"] = -1
    tracks: dict[int, dict] = {}
    windows = sorted(A.t.unique())
    next_id = 0
    for wi, t in enumerate(windows):
        cur = A[A.t == t]
        cands = []
        for i, a in cur.iterrows():
            ab = bands(a.components)
            a_rc = float(a.chirp_rate) if "chirp_rate" in A.columns and pd.notna(a.chirp_rate) else np.nan
            for tid, tr in tracks.items():
                if wi - tr["last_w"] > MAX_GAP:
                    continue
                dr = _rate_match(a.R, a_rc, tr["R"], tr["Rc"])
                if dr is not None and overlap(ab, tr["bands"]):
                    # when several singers share a rate, prefer the one whose
                    # LOWEST band continues the track's recent lowest bands; a
                    # window in which only a harmonic was found still joins,
                    # but only if no better continuation is present
                    pen = 0.0 if (ab and overlap(ab[:1], tr["fund"])) else R_TOL
                    cands.append((dr + pen, -a.depth, i, tid))
        used_i, used_t = set(), set()
        for _, _, i, tid in sorted(cands):
            if i in used_i or tid in used_t:
                continue
            A.loc[i, "track"] = tid
            tr = tracks[tid]
            tr["last_w"] = wi
            if pd.notna(A.loc[i, "R"]):
                tr["R"] = A.loc[i, "R"]
            if "chirp_rate" in A.columns and pd.notna(A.loc[i, "chirp_rate"]):
                tr["Rc"] = A.loc[i, "chirp_rate"]
            # the track's bands are the union over its most recent BAND_MEMORY
            # windows, not over its whole life: an all-time union only grows,
            # and one broadband window (a burst spanning 6-22 kHz linked at the
            # track's rate) would afterwards let any singer at that rate join.
            # A window-animal with more than MAX_COMP_BANDS components is such
            # a burst and contributes only its lowest band.
            mb = _memory_bands(A.loc[i, "components"])
            tr["hist"] = (tr["hist"] + [mb])[-BAND_MEMORY:]
            tr["bands"] = _union(tr["hist"])
            tr["fund"] = _union([[h[0]] for h in tr["hist"] if h])
            used_i.add(i)
            used_t.add(tid)
        for i in cur.index:
            if A.loc[i, "track"] < 0:
                A.loc[i, "track"] = next_id
                h = [_memory_bands(A.loc[i, "components"])]
                rc = A.loc[i, "chirp_rate"] if "chirp_rate" in A.columns else np.nan
                tracks[next_id] = dict(last_w=wi, R=A.loc[i, "R"], Rc=rc, bands=_union(h), hist=h, fund=_union([[x[0]] for x in h if x]))
                next_id += 1
    return A


CHIRP_TOL = 0.06     # chirp rates jitter more than syllable rates


def _rate_match(R_a, Rc_a, R_t, Rc_t):
    """Relative rate difference between a window-singer and a track. A
    chirper carries two rates -- syllable rate R and chirp (bout) rate Rc --
    and the window analysis may read either one in a given window (the map
    peak is at the syllable rate in one window, at the chirp rate in the
    next). So the match is on EITHER: the syllable rates agree within R_TOL,
    or both sides have a chirp rate and those agree within CHIRP_TOL (MBE,
    2026-09-25: "two variables to match on if it's a chirper"). The returned
    difference ranks candidates; a syllable-rate match ranks ahead of a
    chirp-rate-only match by CHIRP_TOL. None = no match.
    (night_animals.py, the production linker, matches on the syllable rate
    when both have one and falls back to the chirp rate only when one side
    lacks a syllable rate.)"""
    dr = abs(R_a / R_t - 1) if pd.notna(R_a) and pd.notna(R_t) else np.nan
    if dr < R_TOL:
        return dr
    if pd.notna(Rc_a) and pd.notna(Rc_t):
        dc = abs(Rc_a / Rc_t - 1)
        # a chirp-rate match may carry a window whose syllable rate is unread or
        # read a few percent off (jittery chirps), but not two chirpers whose
        # syllable rates plainly differ (3.06 kHz Gryllus chorus, 47-55 Hz, was
        # chained into one track by chirp rate alone)
        if dc < CHIRP_TOL and (np.isnan(dr) or dr < CHIRP_SYL_TOL):
            return dc + CHIRP_TOL
    return None


CHIRP_SYL_TOL = 0.045   # syllable rates may differ this much when the chirp rates carry the match (3 x R_TOL)


def _memory_bands(components: str) -> list[tuple[float, float]]:
    bb = bands(components)
    return bb if len(bb) <= MAX_COMP_BANDS else bb[:1]


def _union(hist: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for bb in hist:
        out += [b for b in bb if b not in out]
    return out


def stitch(A: pd.DataFrame) -> pd.DataFrame:
    """Join tracks end-to-start: a track that ends within STITCH_H of another's
    start, with rate within STITCH_R_TOL and any band overlapping, is the same
    animal missed for longer than MAX_GAP (a bout caller's silence, or a few
    windows in which it was over-split). Longest tracks first."""
    A = A.copy()
    info = {}
    for tid, g in A.groupby("track"):
        bb = []
        for c in g.components:
            bb += [b for b in bands(c) if b not in bb]
        g = g.sort_values("t")
        rc = g.chirp_rate.median() if "chirp_rate" in g.columns else np.nan
        info[tid] = dict(t0=g.t.iloc[0], t1=g.t.iloc[-1], R0=g.R.iloc[0], R1=g.R.iloc[-1], Rc=rc, bands=bb, n=len(g))
    order = sorted(info, key=lambda k: -info[k]["n"])
    alias = {k: k for k in info}
    changed = True
    while changed:
        changed = False
        for a in order:
            ia = info[alias[a]]
            best = None
            for b in order:
                if alias[b] == alias[a]:
                    continue
                ib = info[alias[b]]
                gap = (ib["t0"] - ia["t1"]).total_seconds() / 3600
                rate_ok = (pd.notna(ib["R0"]) and pd.notna(ia["R1"]) and abs(ib["R0"] / ia["R1"] - 1) < STITCH_R_TOL) or \
                          (pd.isna(ib["R0"]) or pd.isna(ia["R1"])) and pd.notna(ib["Rc"]) and pd.notna(ia["Rc"]) and abs(ib["Rc"] / ia["Rc"] - 1) < CHIRP_TOL
                if 0 < gap <= STITCH_H and rate_ok and overlap(ia["bands"], ib["bands"]):
                    if best is None or gap < best[0]:
                        best = (gap, alias[b])
            if best is not None:
                src, dst = best[1], alias[a]
                for k in alias:
                    if alias[k] == src:
                        alias[k] = dst
                info[dst] = dict(t0=info[dst]["t0"], t1=info[src]["t1"], R0=info[dst]["R0"], R1=info[src]["R1"],
                                 Rc=info[dst]["Rc"] if pd.notna(info[dst]["Rc"]) else info[src]["Rc"],
                                 bands=info[dst]["bands"] + [x for x in info[src]["bands"] if x not in info[dst]["bands"]],
                                 n=info[dst]["n"] + info[src]["n"])
                changed = True
                break
    A["track"] = A.track.map(alias)
    return A


def reference_carrier(A: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Per track: the component band present in most of its windows (lowest
    on ties) is the reference; per window, the carrier centroid of the part
    overlapping it. NaN when that component is absent in a window."""
    A = A.copy()
    A["F_ref"] = np.nan
    A["F_fix"] = np.nan
    A["ref_lo"] = np.nan
    A["ref_hi"] = np.nan
    has_prof = "prof" in P.columns
    key = P.set_index(["t", "animal"]).sort_index()
    for tid, g in A.groupby("track"):
        seen: dict[tuple, int] = {}
        for comps in g.components:
            bb = bands(comps)
            if len(bb) > MAX_COMP_BANDS:          # a broadband burst window: not a reference candidate
                continue
            for b in bb:
                if b[1] > 1.5 * b[0]:            # nor is a band wider than half its own lower edge
                    continue
                seen[b] = seen.get(b, 0) + 1
        if not seen:
            for comps in g.components:
                for b in bands(comps):
                    seen[b] = seen.get(b, 0) + 1
        if not seen:
            continue
        ref = sorted(seen, key=lambda b: (-seen[b], b[0]))[0]
        # merge bands that touch the reference so a one-row shift still counts
        for i, a in g.iterrows():
            try:
                parts = key.loc[(a.t, a.animal)]
            except KeyError:
                continue
            parts = parts if isinstance(parts, pd.DataFrame) else parts.to_frame().T
            comp = parts[(parts.am_harm_of < 0) & (parts.f_lo <= ref[1] + F_PAD) & (parts.f_hi >= ref[0] - F_PAD)]
            if len(comp):
                best = comp.iloc[np.argmax(comp.depth.to_numpy())]
                A.loc[i, "F_ref"] = best.f_c
                if has_prof and isinstance(best.prof, str):
                    # centroid of R-locked power over a FIXED window about the
                    # reference band, independent of which rows passed threshold
                    w = np.array(best.prof.split(), dtype=float)
                    fk = best.prof_f0 + DF * np.arange(len(w))
                    half = (ref[1] - ref[0]) / 2 + FIX_PAD
                    sel = np.abs(fk - (ref[0] + ref[1]) / 2) <= half
                    if sel.sum() >= 3 and w[sel].sum() > 0:
                        A.loc[i, "F_fix"] = float(np.average(fk[sel], weights=w[sel]))
        A.loc[g.index, "ref_lo"] = ref[0]
        A.loc[g.index, "ref_hi"] = ref[1]
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-windows", type=int, default=6)
    args = ap.parse_args()

    A0, P = load(args.run_dir, args.tag)
    stem = os.path.join(args.run_dir, f"night{args.tag}")
    if A0.empty:
        pd.DataFrame(columns=["t", "animal", "track", "R", "depth", "F_ref", "F_fix"]).to_csv(stem + "_window_animals.csv", index=False)
        pd.DataFrame(columns=["track", "n_windows", "span_h"]).set_index("track").to_csv(stem + "_tracks.csv")
        print("0 windows; 0 window-animals -> 0 night-animals (silent period)")
        return
    A1 = link(A0)
    n_raw = A1.track.nunique()
    A = reference_carrier(stitch(A1), P)
    print(f"linking: {n_raw} tracks; after stitching: {A.track.nunique()}")
    n_w = A.t.nunique()
    T = A.groupby("track").agg(
        n_windows=("t", "size"), t0=("t", "min"), t1=("t", "max"),
        ref_lo=("ref_lo", "first"), ref_hi=("ref_hi", "first"),
        F_ref=("F_ref", "median"), F_ref_sd=("F_ref", "std"), n_F_ref=("F_ref", "count"),
        F_fix=("F_fix", "median"), F_fix_sd=("F_fix", "std"),
        R=("R", "median"), R_sd=("R", "std"), depth=("depth", "median"), on_frac=("on_frac", "median"),
        n_comp_med=("n_components", "median"), n_comp_max=("n_components", "max")).sort_values("n_windows", ascending=False)
    T["span_h"] = (T.t1 - T.t0).dt.total_seconds() / 3600
    T["fill"] = T.n_windows / (T.span_h * 12 + 1)
    A.to_csv(stem + "_window_animals.csv", index=False, float_format="%.3f")
    T.to_csv(stem + "_tracks.csv", float_format="%.3f")

    print(f"{n_w} windows {A.t.min():%H:%M}-{A.t.max():%H:%M}; {len(A)} window-animals -> {len(T)} night-animals")
    print(f"{(T.n_windows == 1).sum()} seen once; {(T.n_windows >= args.min_windows).sum()} in >= {args.min_windows} windows; "
          f"{(T.span_h >= 1).sum()} spanning >= 1 h, {(T.span_h >= 3).sum()} >= 3 h, {(T.span_h >= 6).sum()} >= 6 h")
    long = T[T.n_windows >= args.min_windows]
    print(f"window-animals accounted for by night-animals with >= {args.min_windows} windows: "
          f"{A.track.isin(long.index).mean():.0%}; with depth >= 0.4: "
          f"{A[A.depth >= 0.4].track.isin(long.index).mean():.0%}")
    pd.set_option("display.width", 230)
    cols = ["n_windows", "span_h", "fill", "t0", "t1", "ref_lo", "ref_hi", "F_ref", "F_ref_sd", "F_fix", "F_fix_sd", "n_F_ref", "R", "R_sd", "depth", "on_frac", "n_comp_med"]
    print(long[cols].head(40).to_string(float_format=lambda x: f"{x:.2f}"))

    fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw=dict(height_ratios=[4, 2, 1]))
    lo, hi = np.log10(8), np.log10(100)
    for tid, g in A[A.track.isin(long.index)].groupby("track"):
        g = g.sort_values("t")
        col = plt.cm.turbo((np.log10(g.R.median()) - lo) / (hi - lo))
        ok = g.F_ref.notna()
        axes[0].plot(g.t[ok], g.F_ref[ok], "-", color=col, lw=0.8, alpha=0.8)
        axes[0].scatter(g.t[ok], g.F_ref[ok], s=6 + 50 * g.depth[ok], color=[col], edgecolors="k", lw=0.3)
        if ok.any():
            axes[0].text(g.t[ok].iloc[-1], g.F_ref[ok].iloc[-1], f" {g.R.median():.1f}", fontsize=7, color=col, va="center")
        axes[1].plot(g.t, g.R / g.R.iloc[0], "-", color=col, lw=0.8, alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("carrier of the reference component (Hz)")
    axes[0].set_title(f"{os.path.basename(args.run_dir)}: night-animals in >= {args.min_windows} windows "
                      f"({len(long)} of {len(T)}); colour = rate, size = depth")
    axes[0].grid(alpha=0.3, which="both")
    axes[1].set_ylabel("rate / rate at first window")
    axes[1].grid(alpha=0.3)
    per_w = A.groupby("t").size()
    axes[2].plot(per_w.index, per_w.values, "k.-")
    axes[2].set_ylabel("animals / window")
    axes[2].grid(alpha=0.3)
    fig.savefig(stem + ".png", dpi=110, bbox_inches="tight")
    print(f"wrote {stem}.png / _tracks.csv / _window_animals.csv")


if __name__ == "__main__":
    main()
