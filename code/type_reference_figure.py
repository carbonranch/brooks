"""A formal classification of every call type and an idealized reference fold for each (MBE, 2026-10-13: "formally
classify each type and have some kind of idealized reference").

For each type with >= MIN_N singer-nights (the 25 trill types of the main plane, the 15 chirp types of the chirp plane)
the N_PER clearest singer-nights whose night temperature lies in T_LO..T_HI (widened when the type is rare there; clear
pulse measurement first, then depth, one per recorder-night), the deepest window of each without another singer in
its band (streak_gallery.window_of; for chirpers a window that carries a chirp rate), the loudest 30 s for the band and
a FINE fold (5.8 ms frames, 0.73 ms hop) of the band +/- 700 Hz:
  trill types   folded at the window's syllable rate into NPH_TRILL phase bins
  chirp types   folded at the hour's corrected chirp rate (the chirp plane's hourly table) into NPH_CHIRP bins, so
                one period is one chirp and the syllables inside it are resolved
Each fold is put in dB above its rows' quiet level, aligned so the pulse (or chirp) begins at phase 0, resampled onto a
grid of relative frequency (+/- REL from the track's carrier) and normalized; the idealized reference is the
pixel-wise MEDIAN over the sample's folds with contrast >= MIN_CONTRAST dB. Descriptors per window from
fold_subclass.describe (duty, glide, curvature, bandwidth, sub-peaks) give the per-type medians.

Classification rules (all from the type's own numbers, stated here and in the manifest; nothing is hand-assigned):
  form     trill or chirp: the plane the type was clustered on (song form read from each singer's audio,
           track_form_audio.py)
  tick     a trill type with R25 <= TICK_HZ syllables per second (each syllable audible on its own)
  glide    the median change of the syllable's frequency centroid from its onset to its end, as a share of the
           carrier: flat if |glide| < GLIDE_FLAT, else falling or rising
  hooked   a falling glide (|glide| >= GLIDE_FLAT) whose centroid path bends below its chord by more than HOOK_FRAC of
           the carrier mid-pulse (the quadratic term times the squared half-width of the pulse): the carrier falls
           steeply at first and then levels, the shape heard as a hooked tick
  chirp    syllables per chirp = chirp duty / chirp rate x syllable rate (Rs, where the syllables were read)

Outputs (<out>.pdf/.png the figure; <out>_table.csv and .md the classification; per-type fold stacks cached in
<root>/type_reference/ so the figure can be redrawn with --skip-build):
  (A) the 25 trill types, one cell each: the median fold, x = one syllable period (its length at 25 C in the cell
      head), y = percent from the carrier, color = dB above quiet normalized; the median envelope as a thin line
  (B) the 15 chirp types, one cell each: the median fold over one chirp period

Usage:
  python analysis/callers/type_reference_figure.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --unit-roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --jobs 8 \
      --out analysis/callers/runs/figures_paper/F15_type_reference
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs as W  # noqa: E402
import fold_subclass as FS  # noqa: E402
import streak_gallery as G  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402
import paper_ids  # noqa: E402

TZ = "America/Chicago"
MIN_N = 40
N_PER = 40
T_LO, T_HI = 21.0, 29.0
MIN_SAMPLE = 15              # widen the temperature window until this many singer-nights are available
NPH_TRILL, NPH_CHIRP = 64, 160
REL = 0.08                   # relative-frequency half-range of the reference grid
NREL = 41
MIN_CONTRAST = 4.0           # dB above the rows' quiet level in the fine fold (5.8 ms frames spread a pulse's power over more bins than the 11.6 ms analysis frames)
TICK_HZ = 25.0
GLIDE_FLAT = 0.010           # |glide| / carrier below this: flat
HOOK_FRAC = 0.004            # mid-pulse bend below the chord, as a share of the carrier, for a falling glide to count as hooked
CH_OFF = 100000


# ----------------------------------------------------------------------------------------------------------------------
# windows and folds
def best_window(an: str, unit_roots, form: str, hourly: pd.DataFrame | None):
    """The track's deepest window without another kept singer in its band (streak_gallery.window_of; a chirp type's window
    must carry a chirp rate). The reference is meant to be the type at its clearest, so the deepest window, not a random
    one as in fold_subclass. For a chirper the fold rate is the hour's corrected chirp rate from the chirp plane's table
    (the detector's own chirp rate can sit at 2x or 3x), the syllable rate read there is carried along."""
    w = G.window_of(an, unit_roots, form)
    if w is None:
        return None
    out = dict(recorder=w["recorder"], site=w["site"], t=w["t"], F=w["F_fix"], F_lo=w["F_lo"], F_hi=w["F_hi"], R=w["R"], Rs=np.nan, R_syl=np.nan)
    if form == "chirp":
        if not np.isfinite(w.get("Rc", np.nan)) or w["Rc"] <= 0:
            return None
        out["R_syl"] = w["R"] if np.isfinite(w["R"]) else np.nan
        out["R"] = float(w["Rc"])
        if hourly is not None:
            h = hourly[(hourly.an == an)]
            if len(h):
                hh = h[h.hour == w["t"].floor("h")]
                if len(hh):
                    out["R"] = float(hh.R.iloc[0]); out["Rs"] = float(hh.Rs.iloc[0]) if pd.notna(hh.Rs.iloc[0]) else np.nan
                else:
                    out["R"] = float(h.R.median()); out["Rs"] = float(h.Rs.median()) if h.Rs.notna().any() else np.nan
    elif not np.isfinite(out["R"]) or out["R"] <= 0:
        return None
    return out


def fine_fold(w: dict, nph: int):
    sig, sr, _, _ = W.read_window(w["recorder"], w["site"], w["t"], 300.0)
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2; dt = 128 / sr
    r_lo = max(0, int(np.searchsorted(f, w["F_lo"])) - 1); r_hi = min(len(f) - 1, int(np.searchsorted(f, w["F_hi"])) + 1)
    try:
        cols = W.best_segment(P, r_lo, r_hi, dt, w["R"], FS.FOLD_S)
    except ValueError:
        cols = slice(0, P.shape[1])
    try:
        R = W.refine_rate(P, r_lo, r_hi, cols, dt, w["R"])
    except ValueError:
        R = w["R"]
    if not (0.97 * w["R"] <= R <= 1.03 * w["R"]):
        R = w["R"]
    a0 = cols.start * 128; a1 = min(len(sig), cols.stop * 128 + 512)
    seg = sig[a0:a1]
    ff, tf, Zf = scipy.signal.stft(seg, fs=sr, nperseg=FS.NPERSEG, noverlap=FS.NPERSEG - FS.HOP, nfft=FS.NFFT, window="hann", boundary=None, padded=False)
    Pf = np.abs(Zf) ** 2
    rows = (ff >= w["F"] - FS.PAD_HZ) & (ff <= w["F"] + FS.PAD_HZ)
    Pf = Pf[rows]; fk = ff[rows]
    b = np.floor(((tf * R) % 1.0) * nph).astype(int)
    F = np.zeros((Pf.shape[0], nph)); cnt = np.bincount(b, minlength=nph)
    for k in range(nph):
        F[:, k] = Pf[:, b == k].sum(axis=1)
    F /= np.maximum(cnt, 1)
    Fdb = 10 * np.log10(F + 1e-20); Fdb -= np.percentile(Fdb, 10, axis=1, keepdims=True)
    return Fdb, fk, R


def _one(args):
    key, an, unit_roots, form, hourly = args
    nph = NPH_CHIRP if form == "chirp" else NPH_TRILL
    try:
        w = best_window(an, unit_roots, form, hourly)
        if w is None:
            return dict(key=key, an=an, status="no window")
        Fdb, fk, R = fine_fold(w, nph)
        FS.NPH = nph                                   # describe() reads the module constant
        d = FS.describe(Fdb, fk, w["F"], R)
        rel = np.linspace(-REL, REL, NREL); fr = d["fk"] / w["F"] - 1
        img = np.stack([np.interp(rel, fr, np.clip(d["fold"][:, j], 0, None), left=0, right=0) for j in range(nph)], axis=1)
        d.update(key=key, an=an, status="ok", t=str(w["t"]), R=R, F=w["F"], site=w["site"], Rs=w.get("Rs", np.nan), R_syl=w.get("R_syl", np.nan),
                 img=img / (img.max() + 1e-9))
        d.pop("fold"); d.pop("fk")
        return d
    except SystemExit as e:
        return dict(key=key, an=an, status=f"no audio: {str(e)[:40]}")
    except Exception as e:  # noqa: BLE001
        return dict(key=key, an=an, status=f"{type(e).__name__}: {str(e)[:60]}")


def sample_members(g: pd.DataFrame, clear: set) -> tuple[pd.DataFrame, float, float]:
    """N_PER singer-nights with night temperature in T_LO..T_HI (widened symmetrically until MIN_SAMPLE are available), the
    clearest first: those with a clear pulse measurement (pulse_shapes snr >= 6 dB), then by depth; one per recorder-night."""
    lo, hi = T_LO, T_HI
    while True:
        m = g[g["T"].between(lo, hi)]
        if len(m) >= MIN_SAMPLE or (lo <= g["T"].min() and hi >= g["T"].max()):
            break
        lo -= 1.0; hi += 1.0
    m = m.assign(clear=m.an.isin(clear)).sort_values(["clear", "depth"], ascending=[False, False]).drop_duplicates(["recorder", "night"])
    return m.head(N_PER), lo, hi


# ----------------------------------------------------------------------------------------------------------------------
def build(root, suffix, unit_roots, jobs, seed, cache):
    os.makedirs(cache, exist_ok=True)
    rng = np.random.default_rng(seed)
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{suffix}.csv")); S = pd.read_csv(os.path.join(root, f"streaks_coarse{suffix}.csv"), index_col=0)
    S = S[(S.n >= MIN_N) & (S.form == "trill")]
    ids = paper_ids.load(root, suffix)
    cdir = os.path.join(root, "chirp")
    Ac = pd.read_csv(os.path.join(cdir, "animal_nights_streaks_c2.csv")); Sc = pd.read_csv(os.path.join(cdir, "streaks_coarse_c2.csv"), index_col=0); Sc = Sc[Sc.n >= MIN_N]
    cid = pd.read_csv(os.path.join(cdir, "paper_ids_chirp.csv")).set_index("streak").paper_id
    hourly = pd.read_csv(os.path.join(cdir, "streak_hourly_points_all.csv"), usecols=["hour", "R", "Rs", "an"])
    hourly["hour"] = pd.to_datetime(hourly.hour, utc=True, format="ISO8601").dt.tz_convert(TZ)
    pp = os.path.join(root, f"pulse_shapes{suffix}.csv")
    clear = set(pd.read_csv(pp).query("status == 'ok' and snr_db >= 6").an) if os.path.exists(pp) else set()
    jobs_list, meta = [], []
    for sid in S.index:
        g = A[(A.streak == sid) & (A.form == "trill")]
        m, lo, hi = sample_members(g, clear)
        meta.append(dict(key=int(sid), paper_id=str(ids[sid]), form="trill", n_sampled=len(m), T_lo=lo, T_hi=hi, T_sample=float(m["T"].median())))
        jobs_list += [(int(sid), an, unit_roots, "trill", None) for an in m.an]
    for sid in Sc.index:
        g = Ac[Ac.streak == sid]
        m, lo, hi = sample_members(g, clear)
        hs = hourly[hourly.an.isin(m.an)]
        meta.append(dict(key=int(sid) + CH_OFF, paper_id=str(cid[sid]), form="chirp", n_sampled=len(m), T_lo=lo, T_hi=hi, T_sample=float(m["T"].median())))
        jobs_list += [(int(sid) + CH_OFF, an, unit_roots, "chirp", hs) for an in m.an]
    print(f"{len(meta)} types, {len(jobs_list)} windows to fold", flush=True)
    with Pool(jobs) as pool:
        res = pool.map(_one, jobs_list, chunksize=2)
    # group results by type (a track can belong to a trill type on the main plane and a chirp type on the chirp plane, so
    # the key travels with the job, not with the track)
    rows = []
    by_key = {}
    for r in res:
        by_key.setdefault(r["key"], []).append(r)
    for mt in meta:
        rs = by_key.get(mt["key"], []); ok = [r for r in rs if r["status"] == "ok" and r["contrast"] >= MIN_CONTRAST]
        fails = pd.Series([r["status"] for r in rs if r["status"] != "ok"]).value_counts().to_dict()
        if ok:
            IM = np.stack([r["img"] for r in ok]); np.save(os.path.join(cache, f"folds_{mt['key']}.npy"), IM)
            ENV = np.stack([r["env"] for r in ok]); np.save(os.path.join(cache, f"env_{mt['key']}.npy"), ENV)
            D = pd.DataFrame([{k: v for k, v in r.items() if k not in ("img", "env")} for r in ok]); D.to_csv(os.path.join(cache, f"windows_{mt['key']}.csv"), index=False)
        mt.update(n_ok=len(ok), fails=str(fails) if fails else "")
        rows.append(mt)
        print(f"  {mt['paper_id']:>3s}: {len(ok)}/{len(rs)} folds (T {mt['T_lo']:.0f}-{mt['T_hi']:.0f}) {fails if fails else ''}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(cache, "sample.csv"), index=False)


def md_table(D: pd.DataFrame) -> str:
    cols = list(D.columns)
    fmt = lambda v: "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:g}" if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in D.itertuples(index=False)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
def classify(root, suffix, cache, out):
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{suffix}.csv"), index_col=0)
    Sc = pd.read_csv(os.path.join(root, "chirp", "streaks_coarse_c2.csv"), index_col=0); Sc.index = Sc.index + CH_OFF
    SS = pd.concat([S, Sc])
    C = pd.read_csv(os.path.join(root, f"type_catalog{suffix}.csv")).set_index("type")
    Ac = pd.read_csv(os.path.join(root, "chirp", "animal_nights_streaks_c2.csv")); Ac["night"] = pd.to_datetime(Ac.night)
    Pc = pd.read_csv(os.path.join(root, "chirp", "streak_hourly_points_all.csv"), usecols=["an", "Rs", "T"]).merge(Ac[["an", "streak"]], on="an")
    meta = pd.read_csv(os.path.join(cache, "sample.csv"))
    meta = meta[meta.key.isin(SS.index)]                   # a type dropped from the catalog since the build (QC exclusion) is not drawn
    rows = []
    for mt in meta.itertuples():
        k = int(mt.key); p = os.path.join(cache, f"windows_{k}.csv")
        D = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        F25, R25 = SS.loc[k, "F25"], SS.loc[k, "R25"]
        r = dict(type=mt.paper_id, form=mt.form, F25_Hz=round(F25), R25_Hz=round(R25, 1), n_singer_nights=int(SS.loc[k, "n"]), n_sites=int(SS.loc[k, "n_sites"]), n_nights=int(SS.loc[k, "n_nights"]),
                 n_folds=int(mt.n_ok), T_sample=round(mt.T_sample, 1))
        if len(D):
            dur_ms = 1000 * D.duty / D.R
            glide_rel = D.glide_hz / D.F
            bend = D.curve_hz * (D.duty / 2) ** 2          # mid-pulse departure of the centroid from the chord
            bend_rel = bend / D.F
            r.update(duty=round(float(D.duty.median()), 2), dur_ms=round(float(dur_ms.median()), 1), glide_pct=round(100 * float(glide_rel.median()), 1),
                     bend_pct=round(100 * float(bend_rel.median()), 2), bw_Hz=round(float(D.bw_hz.median())), n_sub=round(float(D.n_sub.median()), 1),
                     glide_iqr_pct=round(100 * float(glide_rel.quantile(0.75) - glide_rel.quantile(0.25)), 1))
            g, b = float(glide_rel.median()), float(bend_rel.median())
            if mt.form == "trill":
                r["glide_class"] = "flat" if abs(g) < GLIDE_FLAT else ("falling" if g < 0 else "rising")
                # hooked: a falling glide whose centroid sits below the chord mid-pulse (fell fast, then leveled); a flat song's
                # small arch is not a hook, so the glide must be a real one
                r["hooked"] = bool(g < -GLIDE_FLAT and b < -HOOK_FRAC)
                r["tick"] = bool(R25 <= TICK_HZ)
        if mt.form == "chirp":
            q = Pc[(Pc.streak == k - CH_OFF) & Pc.Rs.notna()]
            Rs = float(q.Rs.median()) if len(q) else np.nan
            r.update(Rs_Hz=round(Rs, 1) if np.isfinite(Rs) else np.nan, Rs_read_share=round(float(Pc[Pc.streak == k - CH_OFF].Rs.notna().mean()), 2))
            if len(D):
                r["syllables_per_chirp"] = round(float((D.duty / D.R * Rs).median()), 1) if np.isfinite(Rs) else np.nan
                r["chirp_ms"] = r["dur_ms"]
            g2 = Ac[Ac.streak == k - CH_OFF]; wk = g2.night.dt.to_period("W-SAT")
            r.update(first_week=str(wk.min().start_time.date()), peak_week=str(wk.value_counts().idxmax().start_time.date()), last_week=str(wk.max().start_time.date()))
        elif k in C.index:
            r.update(first_week=C.loc[k, "first_week"], peak_week=C.loc[k, "peak_week"], last_week=C.loc[k, "last_week"], day_share=round(float(C.loc[k, "day_share"]), 2),
                     coupling_within=round(float(C.loc[k, "coupling_within"]), 2))
        rows.append(r)
    T = pd.DataFrame(rows)
    T["order"] = [(0 if f == "trill" else 1, int(str(t).lstrip("C"))) for f, t in zip(T.form, T.type)]
    T = T.sort_values("order").drop(columns="order")
    T.to_csv(out + "_table.csv", index=False)
    cols_t = ["type", "F25_Hz", "R25_Hz", "tick", "dur_ms", "duty", "glide_pct", "glide_class", "bend_pct", "hooked", "bw_Hz", "n_singer_nights", "n_sites", "n_folds", "T_sample", "peak_week", "day_share", "coupling_within"]
    cols_c = ["type", "F25_Hz", "R25_Hz", "Rs_Hz", "Rs_read_share", "chirp_ms", "duty", "syllables_per_chirp", "glide_pct", "bw_Hz", "n_singer_nights", "n_sites", "n_folds", "T_sample", "peak_week"]
    with open(out + "_table.md", "w") as fh:
        fh.write("# Call types: formal classification (v3)\n\n")
        fh.write(f"Rules: form = plane clustered on (song form from each singer's audio); tick = trill with R25 <= {TICK_HZ:.0f} Hz; glide = median centroid change onset to end "
                 f"as % of carrier, flat if |glide| < {100 * GLIDE_FLAT:.1f} %; hooked = a falling glide whose mid-pulse centroid sits more than {100 * HOOK_FRAC:.1f} % of the carrier below the chord (bend_pct < -{100 * HOOK_FRAC:.1f}); "
                 f"syllables per chirp = chirp duty / chirp rate x syllable rate. Descriptors are medians over the type's fine folds (n_folds random windows of singer-nights near 25 C; T_sample = their median night temperature).\n\n")
        fh.write("## Trill types\n\n" + md_table(T[T.form == "trill"][cols_t]) + "\n\n")
        fh.write("## Chirp types\n\n" + md_table(T[T.form == "chirp"][cols_c]) + "\n")
    pd.set_option("display.width", 250)
    print(T[T.form == "trill"][cols_t].to_string(index=False)); print(T[T.form == "chirp"][cols_c].to_string(index=False))
    return T


# ----------------------------------------------------------------------------------------------------------------------
def draw(root, suffix, cache, out, T, number):
    meta = pd.read_csv(os.path.join(cache, "sample.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{suffix}.csv"), index_col=0)
    Sc = pd.read_csv(os.path.join(root, "chirp", "streaks_coarse_c2.csv"), index_col=0); Sc.index = Sc.index + CH_OFF
    SS = pd.concat([S, Sc])
    meta = meta[meta.key.isin(SS.index)]
    tr = meta[meta.form == "trill"].copy(); tr["num"] = tr.paper_id.astype(int); tr = tr.sort_values("num")
    ch = meta[meta.form == "chirp"].copy(); ch["num"] = ch.paper_id.str.lstrip("C").astype(int); ch = ch.sort_values("num")
    page = PaperPage(number=number, title="The call types: an idealized reference for each")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    NCOL = 5; X0, X1 = 1.25, 7.15; CW = (X1 - X0) / NCOL; CELL_W = CW - 0.14
    HEAD, CELL_H, PITCH = 0.13, 0.60, 0.86
    Trow = T.set_index("type")

    def cell(mt, x0, top, first_col, last_row, form):
        k = int(mt.key); p = os.path.join(cache, f"folds_{k}.npy")
        ax = fig.add_axes(rect(x0, x0 + CELL_W, top + HEAD, CELL_H))
        F25, R25 = SS.loc[k, "F25"], SS.loc[k, "R25"]
        head = f"{mt.paper_id}  {F25 / 1000:.1f} kHz @ {R25:.0f} Hz  {1000 / R25:.0f} ms" if form == "trill" else f"{mt.paper_id} {F25 / 1000:.1f} kHz @ {R25:.1f} Hz {1000 / R25:.0f} ms"
        ax.set_title(head, fontsize=BODY_PT - 0.5 if form == "trill" else BODY_PT - 1.0, loc="left", pad=1.5)
        if os.path.exists(p):
            IM = np.load(p); ENV = np.load(os.path.join(cache, f"env_{k}.npy"))
            med = np.median(IM, axis=0); med = med / (med.max() + 1e-9)
            ax.imshow(med, origin="lower", aspect="auto", extent=[0, 1, -100 * REL, 100 * REL], cmap="magma", vmin=0, vmax=1, interpolation="bilinear", rasterized=True)
            env = np.median(ENV, axis=0); env = env / (env.max() + 1e-9)
            ax.plot(np.linspace(0, 1, len(env), endpoint=False), -100 * REL + env * 2 * 100 * REL * 0.9, color="w", lw=0.5, alpha=0.85)
        else:
            ax.text(0.5, 0.5, "no fold", ha="center", va="center", fontsize=BODY_PT, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(-100 * REL, 100 * REL)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-5, 0, 5])
        ax.tick_params(labelsize=BODY_PT - 1, length=2, pad=1.5)
        if not first_col:
            ax.set_yticklabels([])
        if not last_row:
            ax.set_xticklabels([])
        return ax

    # (A) trill types
    top0 = 1.15
    n_rows_a = int(np.ceil(len(tr) / NCOL)); axes_a = []
    for i, mt in enumerate(tr.itertuples()):
        rr, cc = divmod(i, NCOL)
        axes_a.append(cell(mt, X0 + cc * CW, top0 + rr * PITCH, cc == 0, rr == n_rows_a - 1 or i + NCOL >= len(tr), "trill"))
    page.label(axes_a[0], "A", "Trill types: the median fold over one syllable period", y=1 + (HEAD + 0.06) / CELL_H)
    # (B) chirp types
    top1 = top0 + n_rows_a * PITCH + 0.32
    n_rows_b = int(np.ceil(len(ch) / NCOL)); axes_b = []
    for i, mt in enumerate(ch.itertuples()):
        rr, cc = divmod(i, NCOL)
        axes_b.append(cell(mt, X0 + cc * CW, top1 + rr * PITCH, cc == 0, rr == n_rows_b - 1 or i + NCOL >= len(ch), "chirp"))
    page.label(axes_b[0], "B", "Chirp types: the median fold over one chirp period", y=1 + (HEAD + 0.06) / CELL_H)
    fig.text((X0 - 0.42 - fx0) / Wc, 1 - (top0 + n_rows_a * PITCH / 2 - 1.0) / Hc, "percent from the carrier", rotation=90, va="center", ha="center", fontsize=BODY_PT)
    fig.text((X0 - 0.42 - fx0) / Wc, 1 - (top1 + n_rows_b * PITCH / 2 - 1.0) / Hc, "percent from the carrier", rotation=90, va="center", ha="center", fontsize=BODY_PT)
    fig.text(((X0 + X1) / 2 - fx0) / Wc, 1 - (top1 + n_rows_b * PITCH - 0.02 - 1.0) / Hc, "phase within one period", va="top", ha="center", fontsize=BODY_PT)
    page.finish(fig, out)
    print(f"wrote {out}.pdf / .png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--number", default="15")
    ap.add_argument("--skip-build", action="store_true"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache = os.path.join(args.root, "type_reference")
    if not args.skip_build:
        build(args.root, args.suffix, args.unit_roots, args.jobs, args.seed, cache)
    T = classify(args.root, args.suffix, cache, args.out)
    draw(args.root, args.suffix, cache, args.out, T, args.number)


if __name__ == "__main__":
    main()
