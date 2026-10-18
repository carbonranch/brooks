"""Score BROOKS against the hand annotations.

For every window folder with annotations.csv and a detections file, each box
is classed from its label (non-insect / harmonic / singer with a rate /
singer without a rate / other) and crossed with the detections:
  covered   some band of a detected animal -- its fundamental or an attached
            component -- contains the box centre, or overlaps at least half
            of the narrower band
  via       fundamental / component: which kind of band did the covering
  rate ok   a covering animal's rate is within RATE_TOL of the rate written in
            the label (or of 1/2, 1/3 of it)
Writes <root>/scorecard.csv (one row per box) and prints the tallies. Run it
again whenever an annotations file changes.

Usage:
    python analysis/callers/annotation_scorecard.py analysis/callers/runs/annotation [--singers singers_v2.csv]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

RATE_TOL = 0.12
NON_INSECT = ("bird", "owl", "coyote", "cat ", "crow", "rain", "tone generator", "plane", "car", "human", "dog", "train")


def classify(label: str):
    l = label.lower()
    if any(k in l for k in NON_INSECT):
        return "non-insect"
    if "harmonic" in l:
        return "harmonic"
    rate = rate_in(label)
    if any(k in l for k in ("not pick", "don't think we", "not sure we", "missed")):
        return "flagged-miss"
    if any(k in l for k in ("triller", "trill", "chirp", "caller", "singer", "burst", "click", "repeat", "syllable", "cicada", "katydid", "orthopt")):
        return "singer-rated" if rate else "singer"
    return "other"


def rate_in(label: str):
    l = label.lower()
    m = re.search(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*hz", l)           # "60-70Hz": a range
    if m:
        return 0.5 * (float(m.group(1)) + float(m.group(2)))
    m = re.search(r"(\d+(?:\.\d+)?)\s*hz", l)                              # "16 Hz", "~37 Hz"
    return float(m.group(1)) if m else None


def bands_of(S: pd.DataFrame):
    """Per animal: the list of (lo, hi, is_fundamental) bands."""
    out = {}
    for r in S.itertuples():
        bands = [(float(r.F_lo), float(r.F_hi), True)]
        for cs in str(r.components).split(";")[1:]:
            try:
                lo_c, hi_c = [float(v) for v in cs.split("-")]
            except ValueError:
                continue
            bands.append((lo_c, hi_c, False))
        out[int(r.animal)] = bands
    return out


def covers(lo, hi, f_lo, f_hi):
    fc = 0.5 * (f_lo + f_hi)
    if lo <= fc <= hi:
        return True
    olap = max(0.0, min(hi, f_hi) - max(lo, f_lo))
    narrower = min(hi - lo, f_hi - f_lo)
    return narrower > 0 and olap >= 0.5 * narrower


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root")
    ap.add_argument("--singers", default="singers.csv", help="detections file in each window folder (singers_v2.csv for the v2 pipeline)")
    ap.add_argument("--out", default=None, help="scorecard file name (default scorecard.csv, or scorecard_<tag>.csv for a non-default --singers)")
    args = ap.parse_args()
    out = args.out or ("scorecard.csv" if args.singers == "singers.csv" else "scorecard_" + args.singers.replace("singers_", ""))
    rows = []
    n_det = 0
    for f in sorted(glob.glob(os.path.join(args.root, "*", "annotations.csv"))):
        d = os.path.dirname(f); win = os.path.basename(d)
        sp = os.path.join(d, args.singers)
        if not os.path.exists(sp):
            continue
        A = pd.read_csv(f); S = pd.read_csv(sp); S = S[pd.to_numeric(S.animal, errors="coerce").notna()].copy(); S["animal"] = S.animal.astype(int)
        n_det += len(S)
        rate_of = dict(zip(S.animal, S.R))
        chirp_of = dict(zip(S.animal, S.chirp_rate)) if "chirp_rate" in S.columns else {}
        bands = bands_of(S)
        for a in A.itertuples():
            kind = classify(str(a.label)); r_hand = rate_in(str(a.label))
            fund_hits, comp_hits = [], []
            for an, bl in bands.items():
                if any(covers(lo, hi, a.f_lo, a.f_hi) for lo, hi, isf in bl if isf):
                    fund_hits.append(an)
                elif any(covers(lo, hi, a.f_lo, a.f_hi) for lo, hi, isf in bl if not isf):
                    comp_hits.append(an)
            hits = fund_hits + comp_hits
            via = "fundamental" if fund_hits else ("component" if comp_hits else "")
            rate_ok = None
            if r_hand and hits:
                # the hand rate may be the syllable rate (or 1/2, 1/3 of the detected one) or, for a chirper, the chirp rate
                rate_ok = any(abs(rate_of[an] / (m * r_hand) - 1) < RATE_TOL for an in hits for m in (1, 2, 3)) or \
                    any(pd.notna(chirp_of.get(an)) and abs(chirp_of[an] / r_hand - 1) < RATE_TOL for an in hits)
            rows.append(dict(window=win, id=a.id, kind=kind, dur_s=round(a.t1 - a.t0, 1), f_lo=a.f_lo, f_hi=a.f_hi, hand_rate=r_hand,
                             n_fund=len(fund_hits), n_comp=len(comp_hits), covered=len(hits) > 0, via=via, rate_ok=rate_ok,
                             det_rates=";".join(f"{rate_of[an]:.1f}" for an in fund_hits) + (";" if fund_hits and comp_hits else "") + ";".join(f"({rate_of[an]:.1f})" for an in comp_hits),
                             label=str(a.label)[:80]))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(args.root, out), index=False)
    pd.set_option("display.width", 220)
    print(f"{len(T)} boxes in {T.window.nunique()} windows; {n_det} detected animals in {args.singers}\n")
    g = T.groupby("kind").agg(boxes=("id", "size"), covered=("covered", "sum"), by_fund=("via", lambda s: int((s == "fundamental").sum())),
                              by_comp=("via", lambda s: int((s == "component").sum())),
                              rate_ok=("rate_ok", lambda s: int((s == True).sum())), rated=("hand_rate", lambda s: int(s.notna().sum())))  # noqa: E712
    print(g.to_string())
    print("\nnon-insect boxes with a detection covering them (false positives; rates in parentheses = via an attached component):")
    print(T[(T.kind == "non-insect") & T.covered][["window", "id", "dur_s", "f_lo", "f_hi", "label", "det_rates"]].to_string(index=False) or "none")
    M = T[T.kind.isin(["singer", "singer-rated", "flagged-miss"])]
    print("\nsinger boxes accounted for only by a component attached to another singer's fundamental:")
    print(M[M.via == "component"][["window", "id", "f_lo", "f_hi", "det_rates", "label"]].to_string(index=False))
    print("\nsingers with no covering detection or component (misses):")
    print(M[~M.covered][["window", "id", "dur_s", "f_lo", "f_hi", "label"]].to_string(index=False))
    print("\nrated singers whose covering detection(s) have a different rate (not within 12 % of the hand rate or its 2x/3x):")
    print(T[(T.kind == "singer-rated") & T.covered & (T.rate_ok == False)][["window", "id", "hand_rate", "det_rates", "label"]].to_string(index=False))  # noqa: E712
    print("\nharmonic boxes reported as their own singer (fundamental band):")
    print(T[(T.kind == "harmonic") & (T.via == "fundamental")][["window", "id", "f_lo", "f_hi", "det_rates", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
