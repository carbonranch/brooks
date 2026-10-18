"""F10: a SINA reference recording beside field singers of the matching type at about the same temperature.

One row per type. First cell: the reference clip (species, recording temperature, measured rate
and carrier). Then N_FIELD singers of the type from distinct recorder-nights whose HRRR
temperature is within T_TOL degC of the clip's, the deepest clear tracks first. Every cell is the
loudest DUR s of the excerpt, a fine display spectrogram in dB above the row's quiet level, on one
log-frequency axis per row zoomed to the band. The reference clip is chosen from
F09_streak_id_reference_z.csv: the clip of the catalog's candidate species with the smallest
combined z on that type's line (or the smallest-z clip of any species when --any).

Usage:
  python analysis/callers/reference_vs_streak_grid.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
      --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 \
      --z analysis/callers/runs/figures_paper/F09_streak_id_reference_z.csv \
      --streaks 5301 6101 6701 5101 6401 --out analysis/callers/runs/figures_paper/F10_reference_vs_streak
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specdisplay as SD  # noqa: E402
import streak_gallery as G  # noqa: E402

N_FIELD = 4
POOL = 3                 # candidates analysed per field cell shown; the best-looking excerpts are kept (MBE: not just crappy examples)
T_TOL = 1.5
R_TOL = 0.15             # log units: a field singer at the clip's temperature should also sing near the clip's rate (HRRR is a loose proxy for the insect's temperature)
DUR = 0.5
VMAX = 30.0


def short(name: str) -> str:
    g = name.split("|")[0].strip().split(); return f"{g[0][0]}. {g[-1]}"


def loudest_stretch(ex_db: np.ndarray, f: np.ndarray, lo: float, hi: float, dt: float, dur: float) -> tuple[int, int]:
    rows = (f >= lo) & (f <= hi); bp = ex_db[rows].sum(axis=0) if rows.any() else ex_db.sum(axis=0)
    n = int(round(dur / dt))
    if ex_db.shape[1] <= n:
        return 0, ex_db.shape[1]
    return int(np.argmax(np.convolve(bp, np.ones(n), "valid"))), n


# a reference clip is a chirper if its band envelope has a 1.5-12 Hz rhythm whose spectral peak stands CHIRP_CONTRAST times
# above its octave neighborhood (O. fultoni 393x, B. malivolans 477x, G. fultoni 218x; trills 2-8x, wavering tree-cricket trills up to 33x)
# (duty is recorded but no longer used: it misfired on bout-structured trills)
CHIRP_CONTRAST, CHIRP_DUTY = 50.0, 0.55
CHIRP_LO, CHIRP_HI = 1.5, 12.0


def overlapping_singers(an: str, w: dict, unit_roots) -> int:
    """Other kept animals in the same window whose band overlaps the cell's display range."""
    unit, track = an.split(":")
    for root in unit_roots:
        p = os.path.join(root, unit, "night_night_global_window_animals.csv")
        if os.path.exists(p):
            Wd = pd.read_csv(p, usecols=["t", "track", "F_lo", "F_hi", "keep"])
            Wd["t"] = pd.to_datetime(Wd.t, utc=True)
            g = Wd[(Wd.t == pd.Timestamp(w["t"]).tz_convert("UTC")) & Wd.keep & (Wd.track != int(track))]
            lo, hi = 0.8 * w["F_lo"], 1.25 * w["F_hi"]
            return int(((g.F_lo <= hi) & (g.F_hi >= lo)).sum())
    return 0


def reference_cell(fn: str, lo: float, hi: float, R: float):
    sig, sr = sf.read(fn, dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != 44100:
        sig = scipy.signal.resample_poly(sig, 44100, sr).astype("float32"); sr = 44100
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2; dt = 128 / sr
    L = 10 * np.log10(P + 1e-20); quiet = np.percentile(L, 10, axis=1, keepdims=True); D = L - quiet
    k, n = loudest_stretch(D, f, lo, hi, dt, DUR)
    seg = sig[k * 128: k * 128 + n * 128]
    ff, tf, Pf = SD.fine_spec(seg, sr)
    # song form of the clip, as the detector's bout test: the band envelope smoothed over two syllable periods,
    # slow trend removed, has a rhythm at 1.5-12 Hz far above the spectrum's median only for a chirper
    # (a share-of-time-loud test called A. fultoni's and N. palustris' trills chirpers: pauses between bouts)
    rows = (f >= lo) & (f <= hi)
    env = 10 * np.log10(P[rows].sum(axis=0) + 1e-20)
    w = max(3, int(round(2.0 / R / dt))); env = np.convolve(env, np.ones(w) / w, "same")
    slow = scipy.signal.medfilt(env, 2 * int(5.0 / dt) + 1) if len(env) > 2 * int(5.0 / dt) + 1 else env.mean()
    x = env - slow; x = x - x.mean()
    F = np.fft.rfftfreq(len(x), dt); S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    band = (F >= CHIRP_LO) & (F <= CHIRP_HI) & (F < R / 3)
    if band.any():
        f0 = F[band][np.argmax(S[band])]; nb = (F >= f0 / 2) & (F <= 2 * f0) & (np.abs(F / f0 - 1) > 0.15)
        contrast = float(S[band].max() / np.median(S[nb]))
    else:
        contrast = 0.0
    lo_db, hi_db = np.percentile(env, 10), np.percentile(env, 95)
    duty = float((env - lo_db > 0.5 * (hi_db - lo_db)).mean())
    # contrast alone, as in track_form_audio.py: the duty test called A. vernalis' trill (duty 0.54) a chirper and left
    # 9101 with no field cells (MBE: "why are there no examples for 9101?")
    is_chirp = contrast > CHIRP_CONTRAST
    return ff, tf, SD.db_above_quiet(Pf, SD.quiet_on_grid(ff, f, quiet.ravel())), dict(is_chirp=is_chirp, contrast=contrast, duty=duty)


def field_cell(w: dict):
    x = G.analyse(w)
    k, n = loudest_stretch(x["ex_db"], x["f"], w["F_lo"], w["F_hi"], x["dt"], DUR)
    seg = x["sig_ex"][k * 128: k * 128 + n * 128]
    ff, tf, Pf = SD.fine_spec(seg, x["sr"])
    Df = SD.db_above_quiet(Pf, SD.quiet_on_grid(ff, x["f"], x["quiet"]))
    # how good the excerpt looks: in-band level above quiet, less the clutter in the rows around the band
    band = (ff >= w["F_lo"]) & (ff <= w["F_hi"]); near = (ff >= 0.8 * w["F_lo"]) & (ff <= 1.25 * w["F_hi"]) & ~band
    score = float(Df[band].mean() - 0.5 * Df[near].mean()) if band.any() and near.any() else float(Df.mean())
    return ff, tf, Df, x["R"], score


def main() -> None:
    global DUR
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--z", required=True, help="F09's per-clip z table"); ap.add_argument("--streaks", type=int, nargs="*", default=None)
    ap.add_argument("--all-candidates", action="store_true", help="every type with >= 40 singer-periods and a candidate by position whose species has a measured clip")
    ap.add_argument("--per-page", type=int, default=6, help="rows per page (pages are written as _pN.png and one PDF)")
    ap.add_argument("--candidate", nargs="*", default=[], metavar="STREAK=Species name", help="override the reference species for a type, e.g. '8104=Bucrates malivolans'")
    ap.add_argument("--any", action="store_true", help="pick the smallest-z clip of any species, not only the catalog candidate")
    ap.add_argument("--n-field", type=int, default=N_FIELD); ap.add_argument("--t-tol", type=float, default=T_TOL); ap.add_argument("--r-tol", type=float, default=R_TOL)
    ap.add_argument("--ignore-form", action="store_true", help="do not require field singers to share the clip's song form (a continuous-trill clip can read as chirps when it has pauses)")
    ap.add_argument("--paper-number", default=None, help="house-style letter pages (paper_page.PaperPage) numbered N.1, N.2, ...: one row per type, 6 rows per page")
    ap.add_argument("--page-start", type=int, default=1, help="number of the first page (the chirp pages continue the trill pages' numbering)")
    ap.add_argument("--n-pages", type=int, default=None, help="total page count printed in the legend ('N of 6') when the set is drawn in two runs")
    ap.add_argument("--chirp-plane", action="store_true", help="the chirp types (chirp/ tables, ids C1..): the clip's and the singers' rate is the chirp rate, --z is chirp_reference_z.py's table")
    ap.add_argument("--dur", type=float, default=DUR, help="excerpt length in s (1.0 for chirpers, so that a cell holds a few chirps)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    DUR = args.dur
    SD.paper_fonts()
    root = args.root
    if args.chirp_plane:
        cdir = os.path.join(root, "chirp")
        A = pd.read_csv(os.path.join(cdir, "animal_nights_streaks_c2.csv"))
        S = pd.read_csv(os.path.join(cdir, "streaks_coarse_c2.csv"), index_col=0)
        C = pd.DataFrame()
        args.pid_path = os.path.join(cdir, "paper_ids_chirp.csv")
    else:
        A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
        S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
        C = pd.read_csv(os.path.join(root, f"type_catalog{args.suffix}.csv")).set_index("type")
        args.pid_path = os.path.join(root, f"paper_ids{args.suffix}.csv")
    Z = pd.read_csv(args.z)
    M = pd.read_csv(os.path.join(root, "sina", "sina_reference_measurements.csv")).set_index("wav")
    rm = os.path.join(root, "sina", "sina_reference_remeasured.csv")
    if os.path.exists(rm):
        X = pd.read_csv(rm).drop_duplicates("wav").set_index("wav")
        for a, b in (("F_c", "F_use"), ("f_lo", "f_lo_use"), ("f_hi", "f_hi_use")):
            M[a] = X[b].reindex(M.index).fillna(M[a])
    pp = os.path.join(root, f"pulse_shapes{args.suffix}.csv")
    clear = set(pd.read_csv(pp).query("status == 'ok' and snr_db >= 6").an) if os.path.exists(pp) else set()
    adir = os.path.join(root, "sina", "audio")
    M["species"] = M.name.str.split("|").str[0].str.strip()
    Mok = M[M.matches_page_kHz.astype(str).str.lower().eq("true") & M.R.notna()]
    streaks = args.streaks
    if streaks is None and args.chirp_plane:
        streaks = list(S[S.n >= 40].sort_values("F25").index)                                # C1.. by carrier
    cand_col = "candidate" if "candidate" in C.columns else "candidate_by_position"       # the curated decision when the catalog carries one
    if args.all_candidates:
        ok_status = C.candidate_status.isin(["recording", "position"]) if "candidate_status" in C.columns else pd.Series(True, index=C.index)
        streaks = [sid for sid in S[S.n >= 40].sort_values("F25", ascending=False).index
                   if sid in C.index and ok_status.get(sid, False) and isinstance(C[cand_col].get(sid), str) and (Mok.species == C[cand_col][sid]).any()]
        skipped = [f"{sid} ({C.candidate_status.get(sid, '?')})" for sid in S[S.n >= 40].index if sid not in streaks]
        print(f"{len(streaks)} types with an accepted candidate that has a measured clip; skipped: {', '.join(skipped)}")

    rows = []
    overrides = {int(k): v for k, v in (x.split("=", 1) for x in args.candidate)}
    for sid in streaks:
        cand = overrides.get(sid, C[cand_col].get(sid) if cand_col in C.columns else None)
        z = Z[(Z.streak == sid)]                                   # every clip this line's rate range covers (best line or not)
        if z[z.best].empty:
            print(f"{sid}: no clip has this line as its best; skipped"); continue
        zc = z[z.species == cand] if (isinstance(cand, str) and not args.any) else z[z.best]
        if not zc.empty:
            r = zc.assign(zt=np.hypot(zc.z, zc.z_along)).sort_values("zt").iloc[0]
            znote = f"z across {r.z:+.1f}, along {r.z_along:+.1f}"
        elif isinstance(cand, str) and (Mok.species == cand).any():   # the candidate's clip exists but this line's rate range does not reach it
            mm = Mok[Mok.species == cand].assign(dT=lambda d: (d["T"] - 25).abs()).sort_values("dT").iloc[0]
            r = pd.Series(dict(wav=mm.name, species=cand, T=mm["T"], R=mm.R, F=mm.F_c, z=np.nan, z_along=np.nan))      # M is indexed by wav
            znote = "outside this line's rate range"
        else:
            r = z[z.best].assign(zt=lambda d: np.hypot(d.z, d.z_along)).sort_values("zt").iloc[0]; znote = f"z across {r.z:+.1f}, along {r.z_along:+.1f}"
        m = M.loc[[r.wav]].iloc[0]                                   # a wav can be listed twice
        ref = reference_cell(os.path.join(adir, r.wav), m.f_lo, m.f_hi, float(m.R) if np.isfinite(m.R) and m.R > 0 else 40.0)   # the bout test smooths over syllables: the clip's syllable rate, not its chirp rate
        ref_chirp = ref[3]["is_chirp"]
        g = A[(A.streak == sid)].copy()
        if "form" in g.columns and not args.ignore_form:   # v8 tables carry each singer-period's form: only firm members of the clip's form (no mixed ones)
            g = g[g.form == ("chirp" if ref_chirp else "trill")]
        g["dT"] = (g["T"] - r["T"]).abs(); g["dR"] = np.abs(np.log(g.R / r.R))
        for tol, rtol in ((args.t_tol, args.r_tol), (2 * args.t_tol, args.r_tol), (2 * args.t_tol, 2 * args.r_tol), (4 * args.t_tol, 2 * args.r_tol)):
            cand_rows = g[(g.dT <= tol) & (g.dR <= rtol)]
            if len(cand_rows) >= 3 * args.n_field:
                break
        cand_rows = cand_rows.assign(clear=cand_rows.an.isin(clear)).sort_values(["clear", "depth"], ascending=[False, False])
        pool, used = [], set()
        for _, p in cand_rows.iterrows():
            if (p.recorder, p.night) in used:
                continue
            w = G.window_of(p.an, args.unit_roots, "chirp" if args.chirp_plane else None)
            if w is None:
                continue
            if args.chirp_plane:                 # the window must carry a chirp rate; the singer-night's corrected chirp rate was already matched to the clip's
                if not np.isfinite(w.get("Rc", np.nan)):
                    continue
            elif not np.isfinite(w["R"]) or abs(np.log(w["R"] / r.R)) > rtol:   # the window shown must itself be near the clip's rate (a deepest window can sit at a sub-harmonic)
                continue
            field_chirp = np.isfinite(w.get("Rc", np.nan))                  # the window carries a chirp (bout) rate
            if field_chirp != ref_chirp and not args.ignore_form:                                     # the field singer must have the clip's song form (MBE: no chirpers beside a trilling reference)
                continue
            try:
                cell = field_cell(w)
            except Exception as e:  # noqa: BLE001
                print(f"  {p.an}: {type(e).__name__}"); continue
            used.add((p.recorder, p.night)); pool.append((p, w, cell, overlapping_singers(p.an, w, args.unit_roots)))
            if len(pool) >= POOL * args.n_field:
                break
        # no other singer in the display range first, then the best-looking excerpt (MBE: avoid overlapping songs)
        picks = sorted(pool, key=lambda t: (t[3] > 0, -t[2][4]))[:args.n_field]
        picks = [t[:3] for t in picks]; picks.sort(key=lambda t: t[0]["T"])
        rows.append(dict(sid=sid, r=r, m=m, ref=ref, picks=picks, tol=tol, rtol=rtol, znote=znote))
        print(f"{sid}: {r.species} {r.wav} {r['T']:.1f} °C, {r.R:.1f} Hz, {m.F_c:.0f} Hz ({znote}; {'chirper' if ref_chirp else 'trill'}, rhythm contrast {ref[3]['contrast']:.0f}x, duty {ref[3]['duty']:.2f}); "
              f"{len(picks)} field singers within {tol:g} °C and {100 * rtol:.0f} % in rate: " + ", ".join(f"{p.recorder[-4:]} {p.night} {p['T']:.0f}° {p.R:.1f} Hz" for p, w, _ in picks))

    ncol = 1 + args.n_field
    pages = [rows[i:i + args.per_page] for i in range(0, len(rows), args.per_page)]
    pdf = PdfPages(args.out + ".pdf")
    for ip, prow in enumerate(pages):
        if args.paper_number:
            draw_paper_page(prow, ncol, args, S, C, args.out + f"_{ip + args.page_start}", ip + args.page_start, args.n_pages or (len(pages) + args.page_start - 1))
        else:
            draw_page(prow, ncol, args, S, pdf, args.out + (f"_p{ip + 1}" if len(pages) > 1 else ""))
    pdf.close()
    print(f"wrote {args.out}.pdf" + (f" and {len(pages)} page PNGs" if len(pages) > 1 else " / .png"))


def draw_paper_page(rows, ncol, args, S, C, out, ipage, npages):
    """House style (MBE, 2026-10-13): the evidence for a name, one row per type, for the reader to judge. Row label: the type, the
    clip's species and temperature, its z across and along the line, and the nearest published species where it differs."""
    from paper_page import PaperPage, BODY_PT
    pp = getattr(args, "pid_path", os.path.join(args.root, f"paper_ids{args.suffix}.csv")); pid = pd.read_csv(pp).set_index("streak").paper_id.to_dict() if os.path.exists(pp) else {}
    page = PaperPage(number=f"{args.paper_number}.{ipage}", title=f"Reference recordings beside field singers of the type they land on ({ipage} of {npages})")
    Wc, Hc = 7.2, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))
    X0, X1, GAP, REF_GAP, ROW0, ROW_PITCH, CELL_H = 1.0, 7.5, 0.07, 0.30, 1.45, 1.40, 0.85     # a wider gap after the reference cell (MBE)
    cw = (X1 - X0 - (ncol - 2) * GAP - REF_GAP) / ncol
    xs = [X0] + [X0 + cw + REF_GAP + (j - 1) * (cw + GAP) for j in range(1, ncol)]

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    for i, row in enumerate(rows):
        r, m, sid = row["r"], row["m"], row["sid"]
        los = [m.f_lo] + [w["F_lo"] for _, w, _ in row["picks"]]; his = [m.f_hi] + [w["F_hi"] for _, w, _ in row["picks"]]
        ylo, yhi = 0.8 * min(los), 1.25 * max(his)
        rate = (lambda v: f"{v:.1f} Hz") if getattr(args, "chirp_plane", False) else (lambda v: f"{v:.0f} Hz")     # chirp rates to a tenth
        cells = [(row["ref"][0], row["ref"][1], row["ref"][2], m.f_lo, m.f_hi, f"{r['T']:.0f} °C, {rate(r.R)}", True)]
        for p_, w, (ff, tf, Df, R, score) in row["picks"]:
            cells.append((ff, tf, Df, w["F_lo"], w["F_hi"], f"{p_['T']:.0f} °C, {rate(p_.R if getattr(args, 'chirp_plane', False) else R)}", False))
        top = ROW0 + i * ROW_PITCH
        first = None
        for j in range(ncol):
            ax = fig.add_axes(rect(xs[j], xs[j] + cw, top, CELL_H))
            if j == 0:
                first = ax
            if j >= len(cells):
                ax.set_axis_off()
                if j == 1 and not row["picks"]:
                    ax.text(0.02, 0.5, "no field singers within the temperature and rate window of the clip", transform=ax.transAxes, fontsize=BODY_PT, va="center", color="0.3", style="italic")
                continue
            ff, tf, Df, lo, hi, ttl, is_ref = cells[j]
            sel = (ff >= ylo) & (ff <= yhi)
            vmax = max(VMAX, float(np.percentile(Df[sel], 99.5))) if is_ref else VMAX
            inb = (ff[sel] >= lo) & (ff[sel] <= hi)
            ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(inb[:, None], Df[sel].shape)), cmap="gray", vmin=0, vmax=vmax, shading="auto", rasterized=True)
            ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(~inb[:, None], Df[sel].shape)), cmap="magma", vmin=0, vmax=vmax, shading="auto", rasterized=True)
            ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(0, DUR)
            ticks = [t for t in (1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000) if ylo <= t <= yhi]
            ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
            ax.set_title(ttl, fontsize=BODY_PT, loc="left", pad=2, color="tab:red" if is_ref else "k")
            ax.tick_params(labelsize=5, length=1.5, pad=1.5)
            for sp in ax.spines.values():
                sp.set_linewidth(1.2 if is_ref else 0.5); sp.set_edgecolor("tab:red" if is_ref else "k")
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("kHz")
            ax.set_xticks([0, DUR / 2, DUR])
            if i == len(rows) - 1:
                ax.set_xticklabels(["0", f"{DUR / 2:g}", f"{DUR:g}"]); ax.set_xlabel("s")
            else:
                ax.set_xticklabels([])
        pub = C.candidate_by_position.get(sid, np.nan) if "candidate_by_position" in C.columns else np.nan
        pub_txt = f"; nearest published: {short(pub)}" if isinstance(pub, str) and short(pub) != short(m["name"]) else ("; also nearest published" if isinstance(pub, str) else "")
        zn = row["znote"].replace("z across", "z across").replace(", along", ", along")
        # row label: "(A) Type 1, " in bold, then the clip's full species name in bold italic (MBE, 2026-10-13: full names, italicized)
        head = page.label(first, chr(65 + i), f"Type {pid.get(sid, sid)}, ", y=1 + 0.20 / CELL_H)
        fig.canvas.draw()
        bb = head.get_window_extent(fig.canvas.get_renderer())
        x_ax = first.transAxes.inverted().transform((bb.x1 + 0.3 * head.get_fontsize() / 72 * fig.dpi, bb.y0))[0]   # the trailing space is not in the extent
        first.text(x_ax, 1 + 0.20 / CELL_H, m["name"].split("|")[0].strip(), transform=first.transAxes, fontsize=head.get_fontsize(), fontweight="bold", fontstyle="italic", va="bottom", ha="left")
    page.finish(fig, out)


def draw_page(rows, ncol, args, S, pdf, out_png):
    pp = os.path.join(args.root, f"paper_ids{args.suffix}.csv"); pid = pd.read_csv(pp).set_index("streak").paper_id.to_dict() if os.path.exists(pp) else {}
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.1 * ncol + 1.6, 2.15 * len(rows) + 1.2), squeeze=False)
    for i, row in enumerate(rows):
        r, m = row["r"], row["m"]
        los = [m.f_lo] + [w["F_lo"] for _, w, _ in row["picks"]]; his = [m.f_hi] + [w["F_hi"] for _, w, _ in row["picks"]]
        ylo, yhi = 0.8 * min(los), 1.25 * max(his)
        cells = [(row["ref"][0], row["ref"][1], row["ref"][2], m.f_lo, m.f_hi, f"SINA reference: {short(m['name'])}, {r['T']:.0f} °C, {'chirps' if row['ref'][3]['is_chirp'] else 'trill'}\n{r.R:.1f} Hz, {m.F_c / 1000:.2f} kHz; {row['znote']}", True)]
        for p, w, (ff, tf, Df, R, score) in row["picks"]:
            cells.append((ff, tf, Df, w["F_lo"], w["F_hi"], f"{p.night[2:]} {p.recorder[-4:]} {p.site[:7]} · {p['T']:.0f} °C\n{R:.1f} Hz, {w['F_fix'] / 1000:.2f} kHz", False))
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(cells):
                ax.set_axis_off()
                if j == 1 and not row["picks"]:                   # say why the row is empty rather than leave a blank
                    why = (f"no field singers shown: the clip {'chirps' if row['ref'][3]['is_chirp'] else 'trills'} and this type's singers "
                           f"{'trill' if row['ref'][3]['is_chirp'] else 'chirp'} (same position on the plane, different song form)"
                           if not args.ignore_form else "no field singers within the temperature and rate window of the clip")
                    ax.text(0.02, 0.5, "\n".join(__import__('textwrap').wrap(why, 60)), transform=ax.transAxes, fontsize=8, va="center", color="0.3", style="italic")
                continue
            ff, tf, Df, lo, hi, ttl, is_ref = cells[j]
            sel = (ff >= ylo) & (ff <= yhi)
            # the reference clips were recorded close to the insect and saturate a 0-30 dB scale; they get their own
            # ceiling, stated in the cell title; the field cells share one scale
            vmax = max(VMAX, float(np.percentile(Df[sel], 99.5))) if is_ref else VMAX
            inb = (ff[sel] >= lo) & (ff[sel] <= hi)                      # the band in color, outside it grayscale (MBE, 2026-10-13; before: a white veil)
            ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(inb[:, None], Df[sel].shape)), cmap="gray", vmin=0, vmax=vmax, shading="auto", rasterized=True)
            ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(~inb[:, None], Df[sel].shape)), cmap="magma", vmin=0, vmax=vmax, shading="auto", rasterized=True)
            if is_ref and vmax > VMAX:
                ttl += f"; color to {vmax:.0f} dB"
            ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(0, DUR)
            ticks = [t for t in (1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000) if ylo <= t <= yhi]
            ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
            ax.set_title(ttl, fontsize=7.0, loc="left", pad=2, color="tab:red" if is_ref else "k", fontweight="bold" if is_ref else "normal")
            ax.tick_params(labelsize=6.5, length=2)
            if is_ref:
                for sp in ax.spines.values():
                    sp.set_edgecolor("tab:red"); sp.set_linewidth(2.0)
            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            if i < len(rows) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
        sid = row["sid"]
        axes[i, 0].set_ylabel(f"type {pid.get(sid, sid)} (streak {sid})\n{S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz at 25 °C\nkHz", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("s", fontsize=7.5)
    fig.suptitle(textwrap.fill(f"A SINA reference recording (red frame: species, recording temperature, measured rate and carrier) beside {args.n_field} singers of the type it lands on, "
                               f"from distinct recorder-nights with the clip's song form (chirps or trill) at about the same air temperature and rate (within {args.t_tol:g} °C and {100 * args.r_tol:.0f} % where enough singers): of {POOL * args.n_field} clear, deep tracks analyzed, those with no other singer in the display range first, then the excerpts with the most in-band level above quiet and the least clutter. "
                               f"Each cell: the loudest {DUR:g} s, one log-frequency axis per row zoomed to the band; grayscale = outside the detected band; color = dB above the row's quiet level (0-{VMAX:.0f}; a reference clip recorded close to the insect gets its own ceiling, stated in its title).", 170),
                 fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    pdf.savefig(fig, bbox_inches="tight"); fig.savefig(out_png + ".png", dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
