"""One table per call type for the paper, joined from the year analyses:
position (F25, R25), size, sites and nights, SINA candidate by position and reference
recordings on the line, coherence numbers (clear tracks), pulse descriptors, chirp share,
season (first / peak / last week, months), cold limit, day share, within-track coupling.

Writes type_catalog<suffix>.csv and type_catalog<suffix>.md (a markdown table of the
large types for pasting). Reads whatever of the inputs exist.

Usage:
    python analysis/callers/type_catalog.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def rd(root, name, **kw):
    p = os.path.join(root, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--min-n", type=int, default=40)
    args = ap.parse_args()
    root, sfx = args.root, args.suffix
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{sfx}.csv"), index_col=0)
    S = S[S.harmonic_of < 0] if "harmonic_of" in S.columns else S
    C = S[["n", "F25", "R25", "n_sites", "n_nights"]].copy()
    if "n_firm" in S.columns:                      # v8: a type's size is its firm members (mixed-form singer-periods are assigned but do not count)
        C["n_firm"] = S.n_firm; C["n_mixed"] = S.n_mixed; C["form"] = S.form
        # day members of the type's form count too (2026-10-07): 2201 and 2301 are daytime types with 19 and 10 night members
        Dd = rd(root, f"animal_days_streaks{sfx}.csv")
        if Dd is not None and "form" in Dd.columns:
            nd = Dd[(Dd.streak >= 0) & (Dd.form == Dd.streak.map(S.form))].groupby("streak").size()
            C["n_day_firm"] = nd.reindex(C.index).fillna(0).astype(int)
        else:
            C["n_day_firm"] = 0
        C["n"] = S.n_firm + C.n_day_firm
    C.index.name = "type"
    # names
    M = rd(root, "streak_sina_matches.csv")
    if M is not None:
        top = M[M["rank"] == 1].set_index("streak")
        C["candidate_by_position"] = top.species.reindex(C.index); C["candidate_subfamily"] = top.subfamily.reindex(C.index)
    # curated decisions (candidate_decisions.csv beside this script): the name the paper may use, with its status
    # a run may carry its own decisions file (candidate_decisions.csv in the year root, keyed by ITS streak ids,
    # e.g. the v3 file carried over from the v2 curation via type_map_v2_to_v3.csv); else the one beside this script
    dec_p = os.path.join(root, "candidate_decisions.csv")
    if not os.path.exists(dec_p):
        dec_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_decisions.csv")
    if os.path.exists(dec_p):
        Dc = pd.read_csv(dec_p).set_index("streak")
        C["candidate"] = Dc.candidate.reindex(C.index); C["candidate_status"] = Dc.status.reindex(C.index).fillna("undecided"); C["candidate_note"] = Dc.note.reindex(C.index)
    Rf = rd(root, "sina_reference_matches.csv")
    if Rf is not None:
        on = Rf[(Rf["rank"] == 1) & (Rf.z.abs() <= 2.5)]
        C["reference_on_line"] = on.groupby("streak").apply(lambda g: "; ".join(f"{r.name} (z {r.z:+.1f})" for r in g.sort_values("z", key=abs).head(3).itertuples())).reindex(C.index)
    # coherence (clear tracks preferred)
    Co = rd(root, f"streak_coherence_clear{sfx}.csv")
    if Co is None:
        Co = rd(root, f"streak_coherence{sfx}.csv")
    if Co is not None:
        Co = Co.set_index("streak")
        for c_src, c_dst in (("incoh_frac", "cooccur_excess_frac"), ("shared", "shared_response_r"), ("splits", "descriptor_splits"), ("season_max_pct", "season_resid_max_pct"), ("verdict", "song_form"), ("position", "position")):
            if c_src in Co:
                C[c_dst] = Co[c_src].reindex(C.index)
    # pulses
    Ps = rd(root, f"pulse_shapes_streaks{sfx}.csv")
    if Ps is not None:
        Ps = Ps.set_index("streak")
        C["pulse_ms"] = Ps.dur_med.reindex(C.index); C["duty"] = Ps.duty_med.reindex(C.index)
        C["glide_hz"] = Ps.glide_med.reindex(C.index); C["sweep_hz_per_ms"] = (Ps.glide_med / Ps.dur_med).reindex(C.index)
    # chirpers
    Ch = rd(root, f"year_chirpers{sfx}.csv")
    if Ch is not None:
        Ch = Ch.set_index("streak")
        C["chirp_share"] = Ch.chirp_share.reindex(C.index).fillna(0); C["chirp_rate_hz"] = Ch.Rc_med.reindex(C.index)
    # season
    Tm = rd(root, f"type_mix{sfx}_streaks.csv")
    if Tm is not None:
        Tm = Tm.set_index("streak")
        for c in ("hours", "first_week", "peak_week", "last_week", "n_weeks"):
            C[c] = Tm[c].reindex(C.index)
    # thermal limits
    Th = rd(root, f"year_thermal_limits{sfx}.csv")
    if Th is not None:
        Th = Th.set_index("streak")
        C["T_cold_limit_p05"] = Th.T_p05.reindex(C.index); C["T_median"] = Th.T_p50.reindex(C.index); C["T_min_heard"] = Th.T_min.reindex(C.index)
    # day share
    Dn = rd(root, f"year_day_night{sfx}.csv")
    if Dn is not None:
        Dn = Dn.set_index("streak")
        C["day_share"] = Dn.day_share.reindex(C.index); C["day_months"] = Dn.day_months.reindex(C.index)
    # coupling
    Cp = rd(root, f"year_coupling{sfx}.csv", index_col=0)
    if Cp is not None:
        C["coupling_within"] = Cp.within.reindex(C.index); C["coupling_n_tracks"] = Cp.n_tracks.reindex(C.index)
    C = C.sort_values("n", ascending=False)
    C.to_csv(os.path.join(root, f"type_catalog{sfx}.csv"), float_format="%.3f")
    # markdown of the large types
    big = C[C.n >= args.min_n].copy()
    def f(v, d=1):
        return "" if pd.isna(v) else (f"{v:.{d}f}" if isinstance(v, (float, np.floating)) else str(v))
    cols = [("type", lambda r: str(r.Index)), ("kHz @ Hz (25 °C)", lambda r: f"{r.F25 / 1000:.2f} @ {r.R25:.0f}"), ("singer-periods", lambda r: f"{int(r.n)}"),
            ("sites", lambda r: f"{int(r.n_sites)}"), ("candidate (position)", lambda r: f(getattr(r, "candidate_by_position", ""))),
            ("reference on line", lambda r: f(getattr(r, "reference_on_line", ""))),
            ("co-occur excess", lambda r: f(getattr(r, "cooccur_excess_frac", np.nan), 2)), ("shared r", lambda r: f(getattr(r, "shared_response_r", np.nan), 2)),
            ("song form", lambda r: f(getattr(r, "song_form", ""))), ("position", lambda r: f(getattr(r, "position", ""))), ("splits", lambda r: f(getattr(r, "descriptor_splits", ""))), ("pulse ms", lambda r: f(getattr(r, "pulse_ms", np.nan))), ("duty", lambda r: f(getattr(r, "duty", np.nan), 2)),
            ("sweep Hz/ms", lambda r: f(getattr(r, "sweep_hz_per_ms", np.nan))), ("chirp share", lambda r: f(getattr(r, "chirp_share", np.nan), 2)),
            ("peak week", lambda r: f(getattr(r, "peak_week", ""))), ("first–last", lambda r: f"{f(getattr(r, 'first_week', ''))} – {f(getattr(r, 'last_week', ''))}"),
            ("T cold limit", lambda r: f(getattr(r, "T_cold_limit_p05", np.nan))), ("day share", lambda r: f(getattr(r, "day_share", np.nan), 2)),
            ("coupling", lambda r: f(getattr(r, "coupling_within", np.nan), 2))]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in big.itertuples():
        lines.append("| " + " | ".join(fn(r) for _, fn in cols) + " |")
    open(os.path.join(root, f"type_catalog{sfx}.md"), "w").write("\n".join(lines) + "\n")
    print(f"{len(C)} types; {len(big)} with >= {args.min_n} singer-periods in the markdown table")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
