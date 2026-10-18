"""Match the month's streaks (carrier and rate at 25 degC, coupling) against
SINA species: Walker's per-recording regressions where a spreadsheet exists
(rate at 25 degC, %/degC, carrier), else the "Song at 25 degC" prose rate and
the page's dominant frequency. A streak matches a species when rate and
carrier agree within tolerance; candidates are ranked by combined distance.

Our rates were brought to 25 degC with a slope fitted against ERA5 grid
temperature, which is damped relative to the insect's own temperature, so the
absolute rate scale may be off by up to ~10%; tolerances are set accordingly.

Usage:
    python analysis/callers/match_sina.py analysis/callers/runs/month
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

R_TOL, F_TOL = 0.20, 0.18        # log tolerances
R_MIN, R_MAX = 8.0, 100.0        # what the AM axis can see


def species_table(sdir: str) -> pd.DataFrame:
    P = pd.read_csv(os.path.join(sdir, "sina_songs.csv"))
    G = pd.read_csv(os.path.join(sdir, "sina_regressions.csv"))
    G = G[G.n >= 3].sort_values("n", ascending=False).drop_duplicates("page")
    rows = []
    for p in P.itertuples():
        g = G[G.page == p.page]
        khz_prose = p.kHz_first
        if len(g):
            g = g.iloc[0]
            rows.append(dict(page=p.page, name=p.name, subfamily=p.subfamily, group=p.group, source="walker data",
                             rate25=g.rate25, rate_slope=g.rate_slope_pctC, n_rec=g.n,
                             kHz=g.kHz25 if pd.notna(g.kHz25) else khz_prose, kHz_slope=g.kHz_slope_pctC,
                             pattern=p.pattern))
        elif pd.notna(p.rate_first) and pd.notna(khz_prose):
            rows.append(dict(page=p.page, name=p.name, subfamily=p.subfamily, group=p.group,
                             source="prose at 25C" if p.rate_is_at_25 else "prose", rate25=p.rate_first,
                             rate_slope=np.nan, n_rec=np.nan, kHz=khz_prose, kHz_slope=np.nan, pattern=p.pattern))
    S = pd.DataFrame(rows)
    S["supplement"] = False
    sup = os.path.join(sdir, "supplement_species.csv")
    if os.path.exists(sup):
        X = pd.read_csv(sup)
        X = X[~X.page.isin(S.page)].assign(supplement=True)
        S = pd.concat([S, X[["page", "name", "subfamily", "group", "source", "rate25", "kHz", "pattern", "supplement"]]], ignore_index=True)
    geo = os.path.join(sdir, "species_geography.csv")
    if os.path.exists(geo):
        G = pd.read_csv(geo).set_index("page")
        S["n_AR"] = S.page.map(G.n_AR).fillna(0).astype(int)
        S["n_near"] = S.page.map(G.n_near).fillna(0).astype(int)
        # keep species recorded in AR or a neighbouring state (LA, TX, OK, MS, MO, TN);
        # hand-supplemented rows carry their own range note and are kept; small
        # crickets (trigs, ground and scaly crickets) are kept regardless, since
        # their Arkansas range data are too thin to filter on
        small = S.subfamily.astype(str).str.lower().isin(["trigonidiinae", "nemobiinae", "mogoplistinae"])
        S = S[(S.n_near > 0) | S.supplement | small]
    site_fn = os.path.join(sdir, "site_species.csv")
    site = set(pd.read_csv(site_fn).species.str.strip()) if os.path.exists(site_fn) else set()
    S["site_obs"] = S.name.str.split("|").str[0].str.strip().isin(site)
    S = S[S.rate25.notna() & S.kHz.notna()]
    return S


def main() -> None:
    root = sys.argv[1]
    S = species_table(os.path.join(root, "sina"))
    T = pd.read_csv(os.path.join(root, "streaks_coarse.csv"), index_col=0)
    if "coupling" not in T.columns:          # the year's ridge-frame streak tables carry no between-night slopes
        T["coupling"] = float("nan")
    S["F_Hz"] = S.kHz * 1000
    S["coupling"] = S.kHz_slope / S.rate_slope
    vis = S[(S.rate25 >= R_MIN * 0.8) & (S.rate25 <= R_MAX * 1.25)]
    print(f"{len(S)} SINA species with rate and carrier; {len(vis)} with rates the AM axis can see "
          f"({(vis.source == 'walker data').sum()} from Walker's data, {(vis.source != 'walker data').sum()} from prose)")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 48)
    rows = []
    for sid, t in T.iterrows():
        dR = np.log(vis.rate25 / t.R25)
        dF = np.log(vis.F_Hz / t.F25)
        ok = (np.abs(dR) < R_TOL) & (np.abs(dF) < F_TOL)
        C = vis[ok].assign(dist=np.sqrt((dR[ok] / R_TOL) ** 2 + (dF[ok] / F_TOL) ** 2))
        # a species seen at the site outranks an equally close one that was not
        C = C.assign(rank_key=C.dist * np.where(C.site_obs, 0.5, 1.0)).sort_values("rank_key")
        if C.empty:
            rows.append(dict(streak=sid, n=t.n, F25=t.F25, R25=t.R25, our_coupling=t.coupling, rank=0, species="--"))
        for k, c in enumerate(C.head(3).itertuples()):
            rows.append(dict(streak=sid, n=t.n, F25=t.F25, R25=t.R25, our_coupling=t.coupling, rank=k + 1,
                             species=c.name.split("|")[0].strip(), common=c.name.split("|")[-1].strip(),
                             subfamily=c.subfamily, source=c.source, site_obs=c.site_obs, n_AR=c.n_AR,
                             sina_kHz=c.kHz, sina_rate25=c.rate25,
                             sina_rate_slope=c.rate_slope, sina_coupling=c.coupling, dist=c.dist, pattern=c.pattern))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(root, "streak_sina_matches.csv"), index=False, float_format="%.3f")
    show = M[["streak", "n", "F25", "R25", "our_coupling", "rank", "species", "subfamily", "site_obs", "n_AR", "source",
              "sina_kHz", "sina_rate25", "sina_rate_slope", "sina_coupling", "dist"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}" if abs(x) < 100 else f"{x:.0f}"))
    print(f"\n{M[M['rank'] == 1].streak.nunique()} of {len(T)} streaks have a SINA candidate; "
          f"{M[M['rank'] == 0].streak.nunique()} have none")
    print("\nunmatched streaks (F25 Hz @ R25 Hz, n animal-nights):")
    for sid, t in T.iterrows():
        if (M[(M.streak == sid) & (M['rank'] == 1)]).empty:
            print(f"   {t.F25:6.0f} Hz @ {t.R25:5.1f} Hz  n={t.n:.0f}  coupling={t.coupling}")


if __name__ == "__main__":
    main()
