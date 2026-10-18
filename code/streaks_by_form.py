"""Streaks defined by song form as well as carrier and rate (_v8).

track_form.py gives every singer-period a form: trill (a chirp rate in < 10 % of its windows),
chirp (>= 50 %), mixed (between), and a placeable flag (syllable rate steady to 20 % within the
night). The streak chain -- build_streaks (anisotropic clustering in temperature-corrected rate x
carrier) -> split_streaks_by_T -> merge_streaks -- is run SEPARATELY for the firm trillers and
the firm chirpers, in sub-roots holding filtered copies of its inputs, with the same parameters
as the year pipeline. The two results are combined:
  trill streaks   by default (--trill-from-ref) KEEP the reference (_v6) membership and ids for the
                  firm trillers: re-clustering the trillers alone re-cut the big lines (6901 into two
                  sub-lines, 6101 along its cold end) that MBE had decided to keep as single types,
                  so the trill geometry stays as reviewed and only the chirpers are re-defined. With
                  --trill-rebuild the trill chain is run too and its streaks take the reference id
                  sharing the most members, else a fresh id;
  chirp streaks   take that id + 50 (4001's chirpers become 4051), else fresh + 50;
  mixed singer-periods are NOT assigned (streak -1, form = mixed): a streak is either trills or
                  chirps, not a mix (MBE, 2026-10-06); --attach-mixed restores the old nearest-streak assignment, flagged;
  unplaceable singer-periods get streak -1.
MBE (2026-10-06): "to the extent we can reliably do it, we should define streaks by type as well
as carrier and rate."

Writes animal_nights_streaks_v8.csv, streaks_coarse_v8.csv, streak_form_report_v8.csv in <root>,
and the per-class chains in <root>/form_trill and <root>/form_chirp.

Usage:
  python analysis/callers/streaks_by_form.py analysis/callers/runs/v2/year_nov24_nov25 --ref _v6 --out _v8
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BR, BF, K, K_HF, HF = 0.0154, 0.0046, 0.55, 0.25, 10000.0
ALONG, PERP = 0.6, 0.06
MIN_N = 15


def run(args):
    p = subprocess.run([PY] + args, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:]); print(p.stderr[-3000:]); raise SystemExit(f"failed: {' '.join(args)}")
    return p.stdout


def chain(root, sub, A_sub, P_sub):
    os.makedirs(sub, exist_ok=True)
    A_sub.to_csv(os.path.join(sub, "temperature_animal_nights.csv"), index=False, float_format="%.4f")
    P_sub.to_csv(os.path.join(sub, "streak_hourly_points_all.csv"), index=False, float_format="%.4f")
    shutil.copy2(os.path.join(root, "temperature_season_by_type.csv"), os.path.join(sub, "temperature_season_by_type.csv"))
    run([f"{HERE}/build_streaks.py", sub, "--bR", str(BR), "--bF", str(BF), "--coupling", str(K), "--coupling-hf", str(K_HF), "--along-tol", str(ALONG), "--perp-tol", str(PERP), "--suffix", "_k55b"])
    run([f"{HERE}/split_streaks_by_T.py", sub, "--in", "_k55b", "--out", "_v4", "--coupling", str(K), "--slope", "2.4", "--sep-sd", "2.5", "--dbic", "50", "--max-k", "2"])
    run([f"{HERE}/merge_streaks.py", sub, "--in", "_v4", "--out", "_v6", "--coupling", str(K)])
    A6 = pd.read_csv(os.path.join(sub, "animal_nights_streaks_v6.csv")); S6 = pd.read_csv(os.path.join(sub, "streaks_coarse_v6.csv"), index_col=0)
    return A6, S6


def along_across(R25, F25):
    kk = np.where(F25 >= HF, K_HF, K); n = np.hypot(1.0, kk)
    lr, lf = np.log(R25), np.log(F25)
    return (lr + kk * lf) / n, (-kk * lr + lf) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--ref", default="_v6", help="streak tables whose ids are kept where members overlap"); ap.add_argument("--out", default="_v8")
    ap.add_argument("--trill-rebuild", action="store_true", help="re-run the chain for the firm trillers too (default: keep the reference membership and ids)")
    ap.add_argument("--attach-mixed", action="store_true", help="assign mixed singer-periods to the nearest streak (default: leave them unassigned)")
    ap.add_argument("--audio-form", action="store_true", help="take each singer-period's form from track_form_audio.csv (v9: measured from the audio) instead of the chirp channel; "
                                                                  "singer-periods whose rate is twice their bout rhythm (harm2) become unplaceable")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, "temperature_animal_nights.csv"))
    P = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"))
    Fm = pd.read_csv(os.path.join(root, "track_form.csv")).set_index("an")
    A = A.join(Fm[["rc", "Rc", "bw_rel", "R_scatter", "form", "band", "placeable"]].rename(columns={"rc": "rc_win", "Rc": "Rc_win"}), on="an")
    A["form"] = A.form.fillna("trill"); A["placeable"] = A.placeable.fillna(True).astype(bool)
    if args.audio_form:
        Fa = pd.read_csv(os.path.join(root, "track_form_audio.csv")); Fa = Fa[Fa.status == "ok"].set_index("an")
        A["form_channel"] = A.form
        A["form"] = A.an.map(Fa.form_audio).fillna(A.form)                     # audio form where measured, chirp-channel form otherwise
        A["contrast"] = A.an.map(Fa.contrast_max); A["harm2"] = A.an.map(Fa.harm2).fillna(False).astype(bool)
        A.loc[A.harm2, "placeable"] = False
        print(f"audio form for {A.an.isin(Fa.index).sum():,} of {len(A):,} singer-periods; form {A.form.value_counts().to_dict()}; "
              f"changed from the chirp channel: {(A.form != A.form_channel).sum():,}; harm2 -> unplaceable: {int(A.harm2.sum())}")
    firm = A.placeable & A.form.isin(["trill", "chirp"])
    out = {}
    Aref_full = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.ref}.csv"))
    Aref = Aref_full[["an", "streak"]].set_index("an").streak
    Sref = pd.read_csv(os.path.join(root, f"streaks_coarse{args.ref}.csv"), index_col=0)
    for form in ("trill", "chirp"):
        sel = A[firm & (A.form == form)]
        if form == "trill" and not args.trill_rebuild:
            A6 = Aref_full[Aref_full.an.isin(sel.an)].copy()
            S6 = Sref[Sref.index.isin(A6.streak)].copy()
            print(f"trill: {len(sel):,} firm trillers keep their {args.ref} streaks ({len(S6)} streaks)", flush=True)
            out[form] = (A6, S6); continue
        print(f"{form}: {len(sel):,} singer-periods -> chain", flush=True)
        out[form] = chain(root, os.path.join(root, f"form_{form}"), sel.drop(columns=[c for c in ["rc_win", "Rc_win", "bw_rel", "R_scatter", "form", "band", "placeable", "form_channel", "contrast", "harm2"] if c in sel.columns]), P[P.an.isin(sel.an)])
    # ids: trill streaks inherit the reference id with the largest member overlap (identity when kept)
    taken, idmap, report = set(), {}, []
    for form, offset in (("trill", 0), ("chirp", 50)):
        A6, S6 = out[form]
        A6["ref"] = A6.an.map(Aref)
        order = S6.sort_values("n", ascending=False).index
        fresh = 9901
        for sid in order:
            g = A6[A6.streak == sid]
            ov = g.ref[g.ref >= 0].value_counts()
            new = None
            for cand, cnt in ov.items():
                if cand + offset not in taken and cnt >= 0.3 * len(g):
                    new = int(cand) + offset; break
            if new is None:
                while fresh + offset in taken:
                    fresh += 1
                new = fresh + offset; fresh += 1
            taken.add(new); idmap[(form, sid)] = new
            report.append(dict(form=form, chain_id=sid, streak=new, n=int(S6.loc[sid, "n"]), ref_best=int(ov.index[0]) if len(ov) else -1, ref_overlap=int(ov.iloc[0]) if len(ov) else 0,
                               F25=S6.loc[sid, "F25"], R25=S6.loc[sid, "R25"]))
    # assemble members
    parts = []
    for form in ("trill", "chirp"):
        A6, S6 = out[form]
        A6["streak"] = [idmap.get((form, s), -1) if s in S6.index else -1 for s in A6.streak]
        parts.append(A6.drop(columns=["ref"]))
    M = pd.concat(parts, ignore_index=True)
    M = M.merge(A[[c for c in ["an", "rc_win", "Rc_win", "bw_rel", "R_scatter", "form", "band", "placeable", "form_channel", "contrast", "harm2"] if c in A.columns]], on="an", how="left")
    # streak table from members
    def coarse(g):
        return pd.Series(dict(n=len(g), F25=g.F25.median(), F25_iqr=g.F25.quantile(.75) - g.F25.quantile(.25), R25=g.R25.median(), R25_iqr=g.R25.quantile(.75) - g.R25.quantile(.25),
                              F_min=g.F.min(), F_max=g.F.max(), R_min=g.R.min(), R_max=g.R.max(), n_sites=g.site.nunique(), n_nights=g.night.nunique(), depth=g.depth.median(), harmonic_of=-1))
    S = M[M.streak >= 0].groupby("streak").apply(coarse)
    S["form"] = ["chirp" if (s % 100) > 50 else "trill" for s in S.index]
    # mixed and placeable: nearest streak of either form within the builder's tolerance
    rest = A[~firm & A.placeable].copy()
    rest["R25"] = rest.R * np.exp(BR * (25 - rest["T"])); rest["F25"] = rest.F * np.exp(BF * (25 - rest["T"]))
    al_s, ac_s = along_across(S.R25.to_numpy(), S.F25.to_numpy()); al_r, ac_r = along_across(rest.R25.to_numpy(), rest.F25.to_numpy())
    d = np.hypot((al_r[:, None] - al_s[None, :]) / ALONG, (ac_r[:, None] - ac_s[None, :]) / PERP)
    j = d.argmin(axis=1); ok = (d[np.arange(len(rest)), j] <= 1.0) if args.attach_mixed else np.zeros(len(rest), bool)
    rest["streak"] = np.where(ok, S.index.to_numpy()[j], -1)
    rest["type_corrected"] = -1
    cols = ["an", "recorder", "site", "night", "F", "R", "T", "F25", "R25", "depth", "streak", "type_corrected", "rc_win", "Rc_win", "bw_rel", "R_scatter", "form", "band", "placeable"]
    if args.audio_form:
        cols += ["form_channel", "contrast", "harm2"]
    unpl = A[~A.placeable].copy(); unpl["R25"] = unpl.R * np.exp(BR * (25 - unpl["T"])); unpl["F25"] = unpl.F * np.exp(BF * (25 - unpl["T"])); unpl["streak"] = -1; unpl["type_corrected"] = -1
    ALL = pd.concat([M[cols], rest[cols], unpl[cols]], ignore_index=True).rename(columns={"rc_win": "rc", "Rc_win": "Rc"})
    # recompute the streak table with the mixed members included (counts), keep the form label
    S2 = ALL[ALL.streak >= 0].groupby("streak").apply(coarse); S2["form"] = S.form.reindex(S2.index).fillna("trill")
    S2["n_firm"] = ALL[(ALL.streak >= 0) & ALL.form.isin(["trill", "chirp"])].groupby("streak").size().reindex(S2.index).fillna(0).astype(int)
    S2["n_mixed"] = ALL[(ALL.streak >= 0) & (ALL.form == "mixed")].groupby("streak").size().reindex(S2.index).fillna(0).astype(int)
    for c in ("n", "n_sites", "n_nights", "harmonic_of"):
        S2[c] = S2[c].astype(int)
    S2.index.name = "streak"
    S2.to_csv(os.path.join(root, f"streaks_coarse{args.out}.csv"), float_format="%.3f")
    ALL.to_csv(os.path.join(root, f"animal_nights_streaks{args.out}.csv"), index=False, float_format="%.3f")
    pd.DataFrame(report).to_csv(os.path.join(root, f"streak_form_report{args.out}.csv"), index=False, float_format="%.3f")
    big = S2[S2.n_firm >= 40]
    print(f"\n{len(S2)} streaks ({(S2.form == 'chirp').sum()} chirp); {len(big)} with >= 40 firm members ({(big.form == 'chirp').sum()} chirp); "
          f"mixed {'assigned ' + str(int(ok.sum())) + ' of ' if args.attach_mixed else 'left unassigned: '}{len(rest)}; unplaceable {len(unpl)}")
    pd.set_option("display.width", 220)
    print(big.sort_values("F25", ascending=False)[["form", "n", "n_firm", "n_mixed", "F25", "R25", "n_sites"]].round(1).to_string())
    rp = pd.DataFrame(report); print("\nid continuity (trill streaks with a reference id):", int(((rp.form == "trill") & (rp.streak < 9900)).sum()), "of", int((rp.form == "trill").sum()))


if __name__ == "__main__":
    main()
