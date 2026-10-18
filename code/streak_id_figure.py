"""Figure 10: putting names to the lines (house style, paper_page.PaperPage) -- SINA reference recordings measured with BROOKS, on the year's plane.

Every calling-song recording on the SINA species pages (Walker's clips, each captioned with its
recording temperature) was run through the same window analysis as the field audio, so each
species is a point (syllable rate, carrier) measured with the same instrument. A type's streak
is a line in (log rate, log carrier) that the type slides along with temperature, so a reference
clip recorded at ANY temperature should fall on the line if it is the same song, whatever the
clip's temperature. z = the clip's perpendicular distance from the line in units of the line's
own scatter (sd of the singer-periods' residuals, floor 0.03 in log carrier). The recording
temperature is used too (MBE, 2026-10-06): from the type's singer-hours, log rate is fitted
against HRRR air temperature, and z_along = (clip's log rate - the rate the type sings at the
clip's temperature) / the type's scatter in rate at a given temperature, widened by a T_UNC
thermometer uncertainty (the clip's thermometer and HRRR are not the same instrument). A clip of
the right species must be on the line (z_across) AND at the right place along it for its
temperature (z_along).

  (A) the plane: the year's singer-hours gray, each type's fitted line drawn over its rate range
      and numbered, every reference clip measured at its page's dominant frequency as a point
      colored by subfamily: filled if within Z_ON sd of some line, hollow otherwise.
  (B) one row per type (ordered by carrier): the reference clips on the line (|z_across| <= Z_ON)
      whose best line this is (smallest combined z), plotted at z_along, labeled with species and
      recording temperature; filled where |z_across| <= 1, hollow where 1 < |z_across| <= Z_ON.
      Shaded: |z_along| <= 1. At the right, the catalog's
      candidate by published position (SINA's rate at 25 degC and dominant frequency), which is a
      separate line of evidence, and the coherence verdict on the type's song form.

Usage:
  python analysis/callers/streak_id_figure.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
      --out analysis/callers/runs/figures_paper/F09_streak_id
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from year_coupling import GROUP, COL  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402
from sina_range import in_range_mask, NEAR_STATES  # noqa: E402

Z_ON, Z_SHOW = 2.5, 3.0
SD_FLOOR = 0.03
T_UNC = 2.0              # degC: thermometer disagreement between a SINA caption and HRRR air temperature
SD_R_FLOOR = 0.08        # log rate: floor on a type's rate scatter at a given temperature
MIN_N = 40
FLO, FHI = 1500.0, 20000.0
RLO, RHI = 10.0, 100.0
GROUP = dict(GROUP, mogoplistinae="scaly crickets (Mogoplistinae)", gryllidae="field crickets (Gryllinae)", pseudophyllinae="katydids (Tettigoniidae)",
             listroscelidinae="katydids (Tettigoniidae)", gryllotalpidae="mole crickets (Gryllotalpidae)")
COL = dict(COL, **{"scaly crickets (Mogoplistinae)": "tab:olive", "mole crickets (Gryllotalpidae)": "tab:gray"})


def short(name: str) -> str:
    g = name.split("|")[0].strip().split()
    return f"{g[0][0]}. {g[-1]}" if len(g) >= 2 else name


def fit_lines(A: pd.DataFrame, S: pd.DataFrame, H: pd.DataFrame) -> dict:
    lines = {}
    for sid in S.index:
        g = A[A.streak == sid]; x, y = np.log(g.R.to_numpy()), np.log(g.F.to_numpy())
        h = H[(H.streak == sid) & H["T"].notna()]
        bT, aT = np.polyfit(h["T"], np.log(h.R), 1)
        sdT = float(np.hypot(max((np.log(h.R) - (aT + bT * h["T"])).std(), SD_R_FLOOR), bT * T_UNC))
        if len(g) >= 8 and x.std() > 0.02:
            b, a = np.polyfit(x, y, 1)
        else:
            b, a = 0.3, np.log(g.F.median()) - 0.3 * np.log(g.R.median())
        resid = y - (a + b * x)
        lines[sid] = dict(a=a, b=b, sd=max(resid.std(), SD_FLOOR), x_lo=x.min() - 0.15, x_hi=x.max() + 0.15, x_min=np.percentile(x, 1), x_max=np.percentile(x, 99),
                          aT=aT, bT=bT, sdT=sdT, T_lo=h["T"].quantile(0.02), T_hi=h["T"].quantile(0.98))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--min-n", type=int, default=MIN_N); ap.add_argument("--out", required=True)
    ap.add_argument("--number", default="12")
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[(S.harmonic_of < 0) if "harmonic_of" in S.columns else True]; S = S[S.n >= args.min_n]
    if "form" in S.columns:                      # chirp types live on the chirp plane (Figure 8); the reference clips carry no chirp rate yet
        S = S[S.form != "chirp"]
    H = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["R", "F", "T", "an"]).merge(A[["an", "streak"]], on="an")
    P = H.sample(40000, random_state=0)
    M = pd.read_csv(os.path.join(root, "sina", "sina_reference_measurements.csv")).dropna(subset=["R"])
    M = M[M.matches_page_kHz.astype(str).str.lower().eq("true")].copy()
    rm = os.path.join(root, "sina", "sina_reference_remeasured.csv")      # sina_remeasure.py: carrier/band checked against the clip's own spectrum and the page kHz
    if os.path.exists(rm):
        X = pd.read_csv(rm).drop_duplicates("wav").set_index("wav")      # a clip can be listed twice (two caption temperatures)
        M["F_c"] = M.wav.map(X.F_use).fillna(M.F_c); M["f_lo"] = M.wav.map(X.f_lo_use).fillna(M.f_lo); M["f_hi"] = M.wav.map(X.f_hi_use).fillna(M.f_hi)
        n_pk = int(M.wav.map(X.source).eq("peak").sum()); print(f"{n_pk} clips take the re-measured carrier (spectral peak at the page frequency)")
    M["group"] = M.subfamily.astype(str).str.lower().map(GROUP).fillna("other")
    # species that could plausibly be at the site (MBE, 2026-10-14): records in AR or a neighboring state, hand-supplemented, or a small cricket
    M["in_range"] = in_range_mask(M, os.path.join(root, "sina")).to_numpy()
    print(f"{len(M)} clips; {int(M.in_range.sum())} of species with records in the region ({NEAR_STATES}) or exempt small crickets, {int((~M.in_range).sum())} outside the region (shown hollow gray in A, not in B)")
    C = pd.read_csv(os.path.join(root, f"type_catalog{args.suffix}.csv")).set_index("type")
    Co = os.path.join(root, f"streak_coherence_clear{args.suffix}.csv")
    verdict = pd.read_csv(Co).set_index("streak").verdict if os.path.exists(Co) else pd.Series(dtype=str)
    lines = fit_lines(A, S, H)
    ids = paper_ids.load(root, args.suffix)
    # the second source of names: SINA's PUBLISHED rate (at 25 C) and dominant frequency per species (match_sina.py), no clip
    # involved; for each type the species the decision rests on when its status is 'position', else the nearest by position
    Pub = pd.read_csv(os.path.join(root, "streak_sina_matches.csv"))
    Pub["group"] = Pub.subfamily.astype(str).str.lower().map(GROUP).fillna("other")
    pub_of = {}
    for sid in S.index:
        want = C.candidate_by_position.get(sid, np.nan)
        q = Pub[(Pub.streak == sid) & (Pub.species == want)] if isinstance(want, str) else Pub.iloc[0:0]
        if q.empty and isinstance(want, str):
            q = Pub[Pub.species == want]
        if not q.empty and np.isfinite(q.sina_rate25.iloc[0]) and np.isfinite(q.sina_kHz.iloc[0]):
            r = q.iloc[0]; L = lines[sid]; x, y = np.log(r.sina_rate25), np.log(1000 * r.sina_kHz)
            pub_of[sid] = dict(species=r.species, group=r.group, R=float(r.sina_rate25), F=1000 * float(r.sina_kHz),
                               z=(y - (L["a"] + L["b"] * x)) / L["sd"], z_along=(x - (L["aT"] + L["bT"] * 25.0)) / L["sdT"])

    # z of every clip against every line whose rate range covers it
    rows = []
    for m in M.itertuples():
        x, y = np.log(m.R), np.log(m.F_c)
        zs = {sid: (y - (L["a"] + L["b"] * x)) / L["sd"] for sid, L in lines.items() if L["x_lo"] <= x <= L["x_hi"]}
        za = {sid: (x - (lines[sid]["aT"] + lines[sid]["bT"] * m.T)) / lines[sid]["sdT"] for sid in zs}
        zt = {sid: float(np.hypot(zs[sid], za[sid])) for sid in zs}
        best = min(zt, key=zt.get) if zt else None
        for sid, z in zs.items():
            L = lines[sid]
            rows.append(dict(wav=m.wav, name=short(m.name), species=m.name.split("|")[0].strip(), T=m.T, R=m.R, F=m.F_c, group=m.group, streak=sid,
                             z=z, z_along=za[sid], z_tot=zt[sid], R_expected=float(np.exp(L["aT"] + L["bT"] * m.T)), T_in_range=bool(L["T_lo"] <= m.T <= L["T_hi"]), best=(sid == best), in_range=bool(m.in_range)))
        M.loc[m.Index, "zbest"] = abs(zs[best]) if best is not None else np.inf
        M.loc[m.Index, "zbest_tot"] = zt[best] if best is not None else np.inf
    Z = pd.DataFrame(rows)
    Z["paper_id"] = Z.streak.map(paper_ids.load(root, args.suffix))
    Z.to_csv(args.out + "_reference_z_all.csv", index=False, float_format="%.3f")       # every clip, with in_range
    Z = Z[Z.in_range].copy()                                                                 # the rows of (B) and Figure 13 use in-range clips only
    Z.to_csv(args.out + "_reference_z.csv", index=False, float_format="%.3f")
    on = Z[Z.best & (Z.z.abs() <= Z_ON)]
    both = on[on.z_along.abs() <= Z_ON]
    print(f"{len(M)} reference clips at the page frequency, {M.page.nunique()} species; {on.wav.nunique()} clips lie within {Z_ON} sd of their best line across it; "
          f"{both.wav.nunique()} of those are also within {Z_ON} sd of where the type sings at the clip's temperature; {both.streak.nunique()} of {len(S)} types have at least one")

    order = S.sort_values("F25", ascending=False).index
    nrow = len(order)
    page = PaperPage(number=args.number, title="Reference recordings and published positions on the lines")
    Wc, Hc = 7.2, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    A_TOP, A_H = 1.15, 2.20
    row_h = min(0.20, 5.3 / nrow); B_TOP = A_TOP + A_H + 0.60; B_H = row_h * nrow
    XA0, XA1 = 1.45, 6.05                                           # room for the legend to the right of (A)
    XB0, XB1 = 2.55, 7.15                                           # z axis; type labels at the left (no names: MBE 2026-10-13)

    # (A)
    ax = fig.add_axes(rect(XA0, XA1, A_TOP, A_H))
    ax.scatter(P.R, P.F, s=0.4, color="0.87", lw=0, rasterized=True)
    for sid, L in lines.items():
        xx = np.linspace(L["x_min"], L["x_max"], 20)
        ax.plot(np.exp(xx), np.exp(L["a"] + L["b"] * xx), color="0.25", lw=0.7, zorder=2)
        ax.annotate(str(ids.get(sid, sid)), (np.exp(xx[-1]), np.exp(L["a"] + L["b"] * xx[-1])), fontsize=4.5, xytext=(2, 0), textcoords="offset points", va="center", color="0.25")
    out_ = M[~M.in_range]
    ax.scatter(out_.R, out_.F_c, s=9, facecolors="none", edgecolors="0.6", lw=0.5, zorder=3, label=f"species without records in the region ({len(out_)})")
    for grp, g in M[M.in_range].groupby("group"):
        hit = g.zbest <= Z_ON
        ax.scatter(g.R[hit], g.F_c[hit], s=12, color=COL.get(grp, "k"), edgecolors="k", lw=0.4, zorder=4, label=f"{grp} ({len(g)})")
        ax.scatter(g.R[~hit], g.F_c[~hit], s=12, facecolors="none", edgecolors=COL.get(grp, "k"), lw=0.7, zorder=3)
    for k, (sid, pb) in enumerate(pub_of.items()):
        ax.scatter([pb["R"]], [pb["F"]], s=16, marker="D", facecolors="none", edgecolors=COL.get(pb["group"], "k"), lw=0.8, zorder=4,
                   label="published position (SINA text, 25 °C)" if k == 0 else None)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(RLO, RHI); ax.set_ylim(FLO, FHI)
    ax.xaxis.set_major_locator(FixedLocator([10, 15, 20, 30, 50, 70, 100])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator([1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000])); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    ax.set_xlabel("syllable rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="SINA reference recordings (circles),\nby subfamily; diamonds = published", title_fontsize=5.5, frameon=False, handletextpad=0.3, borderpad=0.2)   # beside the panel (MBE, 2026-10-13)
    page.label(ax, "A", "Reference recordings on the plane", y=1 + 0.05 / A_H)

    # (B)
    ax = fig.add_axes(rect(XB0, XB1, B_TOP, B_H))
    ax.axvspan(-1, 1, color="0.92", zorder=0); ax.axvline(0, color="0.5", lw=0.5)
    for i, sid in enumerate(order):
        y = nrow - 1 - i
        g = Z[(Z.streak == sid) & Z.best & (Z.z.abs() <= Z_ON) & (Z.z_along.abs() <= Z_SHOW)].sort_values("z_along")
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="0.97", zorder=0)
        if g.empty:
            if sid not in pub_of or abs(pub_of[sid]["z_along"]) > Z_SHOW:
                ax.text(-Z_SHOW + 0.1, y, "no reference recording on this line at its temperature", fontsize=4.5, color="0.5", va="center", style="italic")
        labels = []
        for r in g.itertuples():
            c = COL.get(r.group, "k"); tight = abs(r.z) <= 1
            ax.scatter(r.z_along, y, s=10, color=c if tight else "none", edgecolors="k" if tight else c, lw=0.3 if tight else 0.8, zorder=3)
            if labels and labels[-1]["name"] == r.name and abs(r.z_along - labels[-1]["z"][-1]) < 0.35:
                labels[-1]["z"].append(r.z_along); labels[-1]["T"].append(r.T)
            else:
                labels.append(dict(name=r.name, group=r.group, z=[r.z_along], T=[r.T]))
        # labels (MBE, 2026-10-13: "a lot of overlaps"): taller rows; a label goes in line to the right of its marker, else above,
        # else below, else in line to the left, never into a neighboring row; it must clear every marker and every placed label
        # in this row; a label that fits nowhere is dropped and counted at the row's right edge
        pts_per_z = (XB1 - XB0) * 72 / (2 * Z_SHOW)
        pb = pub_of.get(sid)
        if pb is not None and abs(pb["z_along"]) <= Z_SHOW:
            on_line = abs(pb["z"]) <= Z_ON; cpb = COL.get(pb["group"], "k")
            ax.scatter(pb["z_along"], y, s=14, marker="D", color=cpb if on_line else "none", edgecolors="k" if on_line else cpb, lw=0.4 if on_line else 0.8, zorder=3)
            labels.append(dict(name=f"{short(pb['species'])} (published)", group=pb["group"], z=[pb["z_along"]], T=[]))
        marks = sorted([float(r.z_along) for r in g.itertuples()] + ([pb["z_along"]] if pb is not None and abs(pb["z_along"]) <= Z_SHOW else [])); m_half = 3.2 / pts_per_z
        slots = [(0, "center", "left"), (0.30, "center", "center"), (-0.30, "center", "center"), (0, "center", "right")]
        placed = {k: [(m - m_half, m + m_half) for m in marks] if k in (0, 3) else [] for k in range(len(slots))}
        dropped = 0
        for L in labels:
            txt = L["name"] + (" " + ", ".join(f"{t:.0f}°" for t in L["T"]) if L["T"] else ""); zc = float(np.mean(L["z"]))
            w_z = 0.55 * 4.3 * len(txt) / pts_per_z
            ok = False
            for k, (dyr, va, ha) in enumerate(slots):
                if ha == "left":
                    lo_, hi_ = zc + m_half + 0.02, zc + m_half + 0.02 + w_z
                elif ha == "right":
                    lo_, hi_ = zc - m_half - 0.02 - w_z, zc - m_half - 0.02
                else:
                    lo_, hi_ = zc - w_z / 2, zc + w_z / 2
                if lo_ >= -Z_SHOW - 0.02 and hi_ <= Z_SHOW + 0.02 and all(hi_ < a0 - 0.03 or lo_ > a1 + 0.03 for a0, a1 in placed[k]):
                    ok = True; break
            if not ok:
                dropped += 1; continue
            placed[k].append((lo_, hi_))
            ax.text(zc + (m_half + 0.02 if ha == "left" else (-(m_half + 0.02) if ha == "right" else 0)), y + dyr, txt, fontsize=4.3, ha=ha, va=va, color=COL.get(L["group"], "k"))
        if dropped:
            ax.text(Z_SHOW - 0.03, y - 0.33, f"+{dropped}", fontsize=4, ha="right", va="center", color="0.5")
    ax.set_ylim(-0.6, nrow - 0.4); ax.set_xlim(-Z_SHOW, Z_SHOW)
    ax.set_yticks(range(nrow)); ax.set_yticklabels([f"{ids.get(s_, s_)}  {S.loc[s_, 'F25'] / 1000:.1f} kHz @ {S.loc[s_, 'R25']:.0f} Hz" for s_ in order[::-1]], fontsize=5)
    ax.set_xlabel("z along the line: the clip's rate against the type's rate at the clip's temperature (published positions at 25 °C); negative = slower")
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    page.label(ax, "B", "Recordings on each type's line", x=-(XB0 - XA0) / (XB1 - XB0), y=1 + 0.08 / B_H)
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png and {args.out}_reference_z.csv")


if __name__ == "__main__":
    main()
