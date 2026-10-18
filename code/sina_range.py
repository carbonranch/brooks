"""Which SINA reference clips are of species that could plausibly sing at the Arkansas site (MBE, 2026-10-14: "Are we
restricting the SINA matches to species that might plausibly be in Arkansas?" -- the recording-based comparison was not).

A species is in range when SINA's distribution table has records in Arkansas or a neighboring state (LA, TX, OK, MS, MO,
TN; species_geography.csv, n_near > 0), or the clip is one of the hand-supplemented ones (sina_recordings.csv,
supplement). The old position-based candidate list (match_sina.py) also exempted trigs, ground crickets and scaly
crickets because SINA's Arkansas county data for them are thin; that exemption is NOT applied here, because every one
of the 28 clips (27 species) without a record in any of the seven states is a small cricket (Florida and Gulf-coast
Cycloptilum, Pictonemobius and Anaxipha, the European Nemobius sylvestris), so the exemption would make the rule a
no-op (first run, 2026-10-14). No records in seven states is a stronger statement than none in Arkansas.
"""
from __future__ import annotations

import os

import pandas as pd

SMALL_CRICKETS = {"trigonidiinae", "nemobiinae", "mogoplistinae"}
NEAR_STATES = "AR, LA, TX, OK, MS, MO, TN"


def in_range_mask(M: pd.DataFrame, sina_dir: str) -> pd.Series:
    """Boolean per row of a table with columns page, subfamily and wav (sina_reference_measurements layout)."""
    gp = os.path.join(sina_dir, "species_geography.csv"); rp = os.path.join(sina_dir, "sina_recordings.csv")
    near = pd.read_csv(gp).set_index("page").n_near if os.path.exists(gp) else pd.Series(dtype=float)
    supp = pd.Series(dtype=bool)
    if os.path.exists(rp):
        Rr = pd.read_csv(rp).drop_duplicates("wav").set_index("wav")
        supp = Rr["supplement"].astype(str).str.lower().eq("true") if "supplement" in Rr.columns else pd.Series(False, index=Rr.index)
    # hand-kept presence records beyond SINA's distribution table (range_supplement.csv beside this file; iNaturalist etc.)
    sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "range_supplement.csv")
    extra = set(pd.read_csv(sp).page) if os.path.exists(sp) else set()
    n_near = M.page.map(near).fillna(0) if "page" in M.columns else pd.Series(0, index=M.index)
    if "page" in M.columns:
        n_near = n_near + M.page.isin(extra).astype(int)
    is_supp = M.wav.map(supp).fillna(False).astype(bool) if "wav" in M.columns else pd.Series(False, index=M.index)
    small = M.subfamily.astype(str).str.lower().isin(SMALL_CRICKETS) if "subfamily" in M.columns else pd.Series(False, index=M.index)
    return (n_near > 0) | is_supp
