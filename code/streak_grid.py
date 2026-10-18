"""Per streak, a grid of raw excerpts: rows = positions along the line (cold/slow end at
the top to warm/fast end at the bottom), columns = individual singers at that position
from distinct recorder-nights. Every cell is a short excerpt of the loudest 30 s of the
singer's deepest window, zoomed to the band under analysis, on ONE shared log-frequency
axis for the whole grid, so a glide that is a fixed fraction of the carrier has the
same slope at every row and the carrier's drift along the line is visible as the
pattern moving up the page (MBE, 2026-10-05).

Color = dB above the row's quiet level over the whole 300 s window. Cells are ordered
left to right by temperature within a row. The y range is the streak's analyzed
region: from 0.8x the lowest band edge to 1.25x the highest among the drawn singers.

Writes <root>/grid/streak_<id>_grid.png and a combined grid<suffix>.pdf.

Usage:
    python analysis/callers/streak_grid.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 [--rows 12 --cols 5 --dur 0.5] [--streaks ...]
"""
from __future__ import annotations

import argparse
import textwrap
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import streak_gallery as G  # noqa: E402
import specdisplay as SD  # noqa: E402
import paper_ids  # noqa: E402
SD.paper_fonts()

HF_SPLIT, C_LO, C_HI = 10000.0, 0.55, 0.25
CHIRP_C = 0.35           # the chirp plane's coupling (chirp_year.py): slope of log carrier on log chirp rate used for the along-line coordinate
MASK_ALPHA = 0.18        # veil over the rows outside the detected band
FORM_CHIRP, FORM_TRILL = 0.5, 0.1


def sync_average(ex_db: np.ndarray, dt: float, R: float, k: int = 2) -> np.ndarray:
    """Period-synchronous average: each pixel becomes the mean of itself and the same phase
    point in the k periods before and after (2k+1 periods). Pulses at R pass unchanged; a
    non-periodic event is divided by 2k+1; a neighbour at another rate smears as its phase
    drifts. A sliding fold with a short window -- the detector's own logic as a display
    filter (MBE, 2026-10-05: 'filter the data to only show the thing we've identified')."""
    per = int(round(1.0 / R / dt))
    if per < 1:
        return ex_db
    out = np.zeros_like(ex_db); cnt = np.zeros(ex_db.shape[1])
    for m in range(-k, k + 1):
        s = m * per
        src = slice(max(0, -s), ex_db.shape[1] - max(0, s)); dst = slice(max(0, s), ex_db.shape[1] - max(0, -s))
        out[:, dst] += ex_db[:, src]; cnt[dst] += 1
    return out / np.maximum(cnt, 1)


def choose_grid(A: pd.DataFrame, c: float, rows: int, cols: int, clear: set | None = None) -> list[pd.DataFrame]:
    """rows groups of cols singer-periods at even quantiles of along-line position; deep tracks,
    distinct recorder-nights across the whole grid; within a row ordered by temperature.
    clear: singer-period ids with a measured pulse SNR >= 6 dB (pulse_shapes); preferred when enough."""
    A = A[A.R.notna() & (A.R > 0)].copy()
    lr, lf = np.log(A.R), np.log(A.F)
    A["along"] = ((lr - lr.median()) + c * (lf - lf.median())) / np.hypot(1, c)
    good = A[A.depth >= A.depth.quantile(0.4)] if len(A) >= 40 else A
    if clear is not None:
        gc = good[good.an.isin(clear)]
        if len(gc) >= rows * cols:
            good = gc
    targets = np.quantile(A.along, np.linspace(0.04, 0.96, rows))
    used, out = set(), []
    for tq in targets:
        cand = good.assign(d=(good.along - tq).abs()).sort_values("d")
        picks = []
        for _, r in cand.iterrows():
            if (r.recorder, r.night) in used:
                continue
            used.add((r.recorder, r.night)); picks.append(r)
            if len(picks) >= cols:
                break
        if picks:
            out.append(pd.DataFrame(picks).sort_values("T"))
    return out


def draw_grid(args):
    sid, groups, S_row, cand, unit_roots, out_dir, dur, sync, inset, vector_pdf, tag, form = args
    SD.paper_fonts()
    cells = []       # (row index, col index, record, window dict or None, err)
    for i, grp in enumerate(groups):
        for j, (_, r) in enumerate(grp.iterrows()):
            w = G.window_of(r.an, unit_roots, form)
            # a chirper's window often carries only the chirp rate (the syllable rate inside the chirps is read in 1-20 % of
            # some chirp types' hours); G.analyse then folds at the chirp rate, so such a window is a cell, not "no window"
            has_rate = w is not None and (np.isfinite(w["R"]) or (form == "chirp" and np.isfinite(w.get("Rc", np.nan))))
            if not has_rate:
                cells.append((i, j, r, None, "no window")); continue
            try:
                x = G.analyse(w)
                cells.append((i, j, r, dict(w, **x), ""))
            except SystemExit as e:
                cells.append((i, j, r, None, f"no audio"))
            except Exception as e:  # noqa: BLE001
                cells.append((i, j, r, None, f"{type(e).__name__}"))
    ok = [c for c in cells if c[3] is not None]
    if not ok:
        return sid, None, 0
    ylo = 0.8 * min(c[3]["F_lo"] for c in ok); yhi = 1.25 * max(c[3]["F_hi"] for c in ok)
    if S_row.get("letter"):
        sid_, p_ = render_letter(sid, cells, groups, S_row, ylo, yhi, out_dir, dur, sync, inset, tag)
        return sid_, p_, len(ok)
    nrow, ncol = len(groups), max(len(g) for g in groups)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol + 1.2, 1.75 * nrow + 1.4), squeeze=False, sharex=True, sharey=True)
    n_half = None
    for (i, j, r, w, err) in cells:
        ax = axes[i, j]
        if w is None:
            ax.text(0.5, 0.5, err, ha="center", va="center", fontsize=7, transform=ax.transAxes); continue
        ff, tf, Df, _, _ = cell_image(w, dur, sync)
        sel = (ff >= ylo) & (ff <= yhi)
        # the detected band in color, everything outside it in grayscale on the same scale (MBE, 2026-10-13; before: a white veil)
        inb = (ff[sel] >= w["F_lo"]) & (ff[sel] <= w["F_hi"])
        ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(inb[:, None], Df[sel].shape)), cmap="gray", vmin=0, vmax=30, shading="auto", rasterized=True)
        ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(~inb[:, None], Df[sel].shape)), cmap="magma", vmin=0, vmax=30, shading="auto", rasterized=True)
        rc = f" c{w['Rc']:.1f}" if np.isfinite(w.get("Rc", np.nan)) else ""
        rate_txt = "no syllable rate" if w.get("fold_at_chirp") else f"{w['R']:.1f} Hz"
        ax.set_title(f"{w['t']:%y-%m-%d} {w['recorder'][-4:]} {w['site'][:6]} · {r['T']:.0f} °C · {rate_txt}{rc}", fontsize=6.3, loc="left", pad=2)
        ax.tick_params(labelsize=6, length=2)
    for i, grp in enumerate(groups):
        for j in range(len(grp), ncol):
            axes[i, j].set_axis_off()
        axes[i, 0].set_ylabel(f"pos {i + 1}/{nrow}\n" + (f"chirps {grp.R.median():.1f} Hz" if S_row.get("chirp_plane") else f"R {grp.R.median():.0f} Hz") + f" · F {grp.F.median() / 1000:.2f} k", fontsize=6.5)
    ax0 = axes[0, 0]
    ax0.set_yscale("log"); ax0.set_ylim(ylo, yhi)
    ticks = [t for t in [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000] if ylo <= t <= yhi]
    if len(ticks) < 3:
        ticks = list(np.round(np.geomspace(ylo * 1.02, yhi * 0.98, 4), -2))
    ax0.yaxis.set_major_locator(FixedLocator(ticks)); ax0.yaxis.set_minor_locator(FixedLocator([]))
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    for ax in axes[-1, :len(groups[-1])]:
        ax.set_xlabel("s", fontsize=7)
    rate25 = f"chirps {S_row['R25']:.1f} Hz" if S_row.get("chirp_plane") else f"{S_row['R25']:.0f} Hz"
    head = f"{S_row.get('label', f'streak {sid}')}: {S_row['F25'] / 1000:.2f} kHz @ {rate25} at 25 °C, {int(S_row['n'])} singer-periods" + (f"; SINA candidate by position: {cand}" if cand else "")
    if S_row.get("season"):
        head += S_row["season"] + f" ({int(S_row['n_season'])} singer-periods)"
    head = textwrap.fill(head, 115 if inset is not None else 150)   # narrower when the two insets sit at the top right; the day pages' heads ran off the left edge
    filt = (f"RATE-FILTERED VIEW: period-synchronous average over {2 * sync + 1} periods at the track's rate; " if sync else "raw spectrogram; ")
    body = ("rows = positions along the line, cold/slow (top) to warm/fast (bottom); columns = individual singers at that position, distinct recorder-nights, by temperature\n"
            + filt + f"{dur:.1f} s of the loudest 30 s of each singer's deepest window; one shared log-frequency axis (kHz); grayscale = outside the detected band; color = dB above the row's quiet level (0–30)")
    if inset is not None:
        body = textwrap.fill(body.replace("\n", " "), 115)
    fig.suptitle(head + "\n" + body, fontsize=8.5, x=0.37 if inset is not None else 0.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.935 if inset is not None else 0.95))
    if inset is not None:
        # where the rows sit on the plane: a locator on the whole plane (this streak orange, a box around the zoom) and, beside it,
        # the zoom on the streak with the row centers as open rings (MBE, 2026-10-13: on the whole plane the twelve rings overlapped)
        chirp = bool(S_row.get("chirp_plane"))
        Rc_, Fc_ = np.array([g.R.median() for g in groups]), np.array([g.F.median() for g in groups])
        xlo, xhi = Rc_.min() / 1.25, Rc_.max() * 1.25; ylo_, yhi_ = Fc_.min() / 1.2, Fc_.max() * 1.2
        axl = fig.add_axes([0.745, 0.945, 0.075, 0.047])
        axl.scatter(inset["R_all"], inset["F_all"], s=0.3, color="0.85", lw=0, rasterized=True)
        axl.scatter(inset["R_s"], inset["F_s"], s=0.5, color="tab:orange", lw=0, alpha=0.6, rasterized=True)
        axl.add_patch(plt.Rectangle((xlo, ylo_), xhi - xlo, yhi_ - ylo_, fc="none", ec="k", lw=0.6, zorder=5))
        axl.set_xscale("log"); axl.set_yscale("log"); axl.set_ylim(1500, 20000)
        if chirp:
            axl.set_xlim(1.2, 12); axl.xaxis.set_major_locator(FixedLocator([2, 5, 10]))
        else:
            axl.set_xlim(10, 100); axl.xaxis.set_major_locator(FixedLocator([10, 30, 100]))
        axl.xaxis.set_minor_locator(FixedLocator([])); axl.yaxis.set_major_locator(FixedLocator([2000, 5000, 20000])); axl.yaxis.set_minor_locator(FixedLocator([]))
        axl.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axl.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axl.tick_params(labelsize=4.5, length=1.5, pad=1); axl.set_title("the chirp plane" if chirp else "the year plane", fontsize=5.5, pad=1.5)
        axi = fig.add_axes([0.845, 0.945, 0.135, 0.047])
        axi.scatter(inset["R_all"], inset["F_all"], s=0.6, color="0.85", lw=0, rasterized=True)
        axi.scatter(inset["R_s"], inset["F_s"], s=1.0, color="tab:orange", lw=0, alpha=0.6, rasterized=True)
        # open rings only, no numbers: the rows are in order along the line (MBE, 2026-10-13)
        axi.scatter(Rc_, Fc_, s=16, facecolors="none", edgecolors="k", lw=0.6, zorder=5)
        axi.set_xscale("log"); axi.set_yscale("log"); axi.set_xlim(xlo, xhi); axi.set_ylim(ylo_, yhi_)
        xt = [t for t in ([1.5, 2, 2.5, 3, 4, 5, 6, 8, 10] if chirp else [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]) if xlo <= t <= xhi]
        yt = [t for t in [1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000] if ylo_ <= t <= yhi_]
        axi.xaxis.set_major_locator(FixedLocator(xt)); axi.yaxis.set_major_locator(FixedLocator(yt)); axi.xaxis.set_minor_locator(FixedLocator([])); axi.yaxis.set_minor_locator(FixedLocator([]))
        axi.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axi.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axi.tick_params(labelsize=4.5, length=1.5, pad=1); axi.set_xlabel("chirps/s" if chirp else "Hz", fontsize=5, labelpad=0); axi.set_ylabel("kHz", fontsize=5, labelpad=0)
        axi.set_title("the rows on the streak, in order", fontsize=5.5, pad=1.5)
    p = os.path.join(out_dir, f"{S_row.get('fname', f'streak_{sid}')}{('_' + tag) if tag else ''}_grid{'_sync' if sync else ''}.png"); fig.savefig(p, dpi=120)
    if vector_pdf:
        fig.savefig(p[:-4] + ".pdf")
    plt.close(fig)
    return sid, p, len(ok)


def cell_image(w: dict, dur: float, sync: int):
    """The dur-second stretch of the singer's 1 s excerpt with the most band power, as a fine display spectrogram in dB above quiet."""
    ex, f, dt = w["ex_db"], w["f"], w["dt"]
    n_half = int(round(dur / dt))
    rows_b = (f >= w["F_lo"]) & (f <= w["F_hi"])
    bp = ex[rows_b].sum(axis=0) if rows_b.any() else ex.sum(axis=0)
    if ex.shape[1] > n_half:
        k = int(np.argmax(np.convolve(bp, np.ones(n_half), "valid")))
    else:
        k, n_half = 0, ex.shape[1]
    seg = w["sig_ex"][k * 128: k * 128 + n_half * 128]
    ff, tf, Pf = SD.fine_spec(seg, w["sr"])
    Df = SD.db_above_quiet(Pf, SD.quiet_on_grid(ff, f, w["quiet"]))
    if sync:
        Df = sync_average(Df, tf[1] - tf[0], w["R"], sync)
    return ff, tf, Df, k, n_half


CLIP_S = 5.0             # listening clip around each drawn cell (MBE, 2026-10-13: "save the audio snippets somewhere - i'd like to listen to them")


def save_clip(w: dict, t_mid, path: str) -> float:
    """CLIP_S seconds of the recording centered on the drawn stretch, peak-normalized to -3 dBFS, 16-bit WAV. Returns the peak gain applied (dB)."""
    import soundfile as sf
    import window_blobs as W
    sig, sr, t0, _ = W.read_window(w["recorder"], w["site"], t_mid, CLIP_S)
    peak = float(np.abs(sig).max()) if len(sig) else 0.0
    gain = (10 ** (-3 / 20) / peak) if peak > 0 else 1.0
    sf.write(path, np.clip(sig * gain, -1, 1), sr, subtype="PCM_16")
    return float(20 * np.log10(gain)) if peak > 0 else 0.0


def render_letter(sid, cells, groups, S_row, ylo, yhi, out_dir, dur, sync, inset, tag):
    """The same grid on an 8.5 x 11 page in the house style (MBE, 2026-10-13: "those need to be on 8.5 x 11 pages"):
    head line and the two insets at the top, the rows x cols cells below, one legend line. Writes <fname>_grid.pdf/.png."""
    from paper_page import PaperPage, BODY_PT, LABEL_PT
    chirp = bool(S_row.get("chirp_plane"))
    nrow, ncol = len(groups), max(len(g) for g in groups)
    rate25 = f"chirps {S_row['R25']:.1f} Hz" if chirp else f"{S_row['R25']:.0f} Hz"
    legend = ("Rows are positions along the line from the cold, slow end at the top to the warm, fast end at the bottom; columns are singers at that position from "
              f"distinct recorder-nights, ordered by air temperature. Each cell is {dur:g} s of the loudest 30 s of the singer's deepest window on one shared log-frequency axis, "
              "the detected band in color (dB above the row's quiet level, 0 to 30) and the rest in grayscale"
              + ("; a cell whose singer had no syllable rate in that window is folded at its chirp rate" if chirp else "")
              + ". The cell head gives the night, recorder, site, air temperature and rate (c: chirp rate). Top right: the type on the plane, and the row positions on the streak, in order.")
    label = S_row.get("label", f"streak {sid}"); label = label[0].upper() + label[1:]
    if sync:                                  # the rate-filtered companion page (MBE 2026-10-17: number those S7 too)
        legend = (f"RATE-FILTERED VIEW of the grid: each cell is the period-synchronous average of the excerpt over {2 * sync + 1} periods at the singer's own rate, so the singer's "
                  "syllables stand out and neighbors at other rates smear. ") + legend
    page = PaperPage(number=f"S7-{S_row.get('pid', sid)}" + (" (rate-filtered)" if sync else ""), title=f"{label}: {S_row['F25'] / 1000:.2f} kHz @ {rate25} at 25 °C, {int(S_row['n'])} singer-nights.", legend=legend)   # the grids are Figure 7's supplement (MBE 2026-10-16: "make those Figure 7S")
    Wc, Hc = 7.2, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    X0, X1 = fx0 + 0.52, fx0 + Wc - 0.03
    GRID_TOP, HEAD_H, GAP = 1.95, 0.10, 0.04
    pitch_h = (1.0 + Hc - GRID_TOP - 0.22) / nrow; cell_h = pitch_h - HEAD_H - GAP
    pitch_w = (X1 - X0) / ncol; cell_w = pitch_w - 0.07
    # head
    head = f"{label}, {S_row['F25'] / 1000:.2f} kHz @ {rate25} at 25 °C"
    head = textwrap.fill(head, 62) if len(head) > 62 else head        # the head must end before the insets at 4.55 in (MBE 2026-10-16: long heads overflowed into them)
    fig.text((fx0 + 0.02 - fx0) / Wc, 1 - 0.02 / Hc, head, fontsize=LABEL_PT, fontweight="bold", va="top", ha="left", linespacing=1.1)
    y_sub = 0.20 + (0.13 if "\n" in head else 0.0)
    fig.text((fx0 + 0.02 - fx0) / Wc, 1 - y_sub / Hc, "rows: positions along the line, cold and slow at the top, warm and fast at the bottom\ncolumns: singers at that position, distinct recorder-nights, by air temperature",
             fontsize=BODY_PT, va="top", ha="left", linespacing=1.3)
    ticks = [t for t in [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000] if ylo <= t <= yhi]
    if len(ticks) < 3:
        ticks = list(np.round(np.geomspace(ylo * 1.02, yhi * 0.98, 4), -2))
    first_axes = {}; audio_rows = []
    for (i, j, r, w, err) in cells:
        top = GRID_TOP + i * pitch_h + HEAD_H
        ax = fig.add_axes(rect(X0 + j * pitch_w, X0 + j * pitch_w + cell_w, top, cell_h))
        if j == 0:
            first_axes[i] = ax
        ax.set_yscale("log"); ax.set_ylim(ylo, yhi); ax.set_xlim(0, dur)
        ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(FixedLocator([])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        ax.set_xticks([0, dur / 2, dur]); ax.tick_params(labelsize=BODY_PT - 1, length=1.5, pad=1.5, width=0.5)
        if j > 0:
            ax.set_yticklabels([])
        if i < nrow - 1:
            ax.set_xticklabels([])
        else:                                   # the end label only on the last column, so neighbors' labels do not touch
            ax.set_xticklabels(["0", f"{dur / 2:g}", f"{dur:g}" if j == ncol - 1 else ""])
            if j == 0:
                ax.set_xlabel("s", fontsize=BODY_PT - 1, labelpad=1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        if w is None:
            ax.text(0.5, 0.5, err, ha="center", va="center", fontsize=BODY_PT - 1, transform=ax.transAxes, color="0.4"); continue
        ff, tf, Df, k_ex, n_ex = cell_image(w, dur, sync)
        if S_row.get("audio"):
            adir = os.path.join(out_dir, "audio", S_row.get("fname", f"streak_{sid}")); os.makedirs(adir, exist_ok=True)
            t_mid = w["t_ex"] + pd.Timedelta(seconds=(k_ex + n_ex / 2) * w["dt"])
            fn = f"pos{i + 1:02d}_{j + 1}_{w['t']:%Y%m%d}_{w['recorder'][-4:]}_{w['site']}.wav"
            try:
                gain = save_clip(w, t_mid, os.path.join(adir, fn))
                audio_rows.append(dict(row=i + 1, col=j + 1, an=r.an, file=fn, clip_start_local=str((t_mid - pd.Timedelta(seconds=CLIP_S / 2)).tz_convert(G.TZ)), drawn_from_s=round(CLIP_S / 2 - dur / 2, 2), drawn_to_s=round(CLIP_S / 2 + dur / 2, 2),
                                       T=round(float(r["T"]), 1), R_Hz=round(float(w["R"]), 2), chirp_Hz=(round(float(w["Rc"]), 2) if np.isfinite(w.get("Rc", np.nan)) else ""), F_lo=round(w["F_lo"]), F_hi=round(w["F_hi"]), gain_dB=round(gain, 1)))
            except Exception as e:  # noqa: BLE001
                audio_rows.append(dict(row=i + 1, col=j + 1, an=r.an, file="", error=f"{type(e).__name__}: {str(e)[:60]}"))
        sel = (ff >= ylo) & (ff <= yhi); inb = (ff[sel] >= w["F_lo"]) & (ff[sel] <= w["F_hi"])
        ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(inb[:, None], Df[sel].shape)), cmap="gray", vmin=0, vmax=30, shading="auto", rasterized=True)
        ax.pcolormesh(tf, ff[sel], np.ma.masked_array(Df[sel], mask=np.broadcast_to(~inb[:, None], Df[sel].shape)), cmap="magma", vmin=0, vmax=30, shading="auto", rasterized=True)
        rc = f"c{w['Rc']:.1f}" if np.isfinite(w.get("Rc", np.nan)) else ""
        rate = (rc if w.get("fold_at_chirp") else f"{w['R']:.1f}" + (f" {rc}" if rc else "")) + " Hz"
        ax.set_title(f"{w['t']:%y-%m-%d} {w['recorder'][-4:]} {w['site'][:6]}  {r['T']:.0f} °C  {rate}", fontsize=4.5, loc="left", pad=1.2)
    for i, grp in enumerate(groups):
        if i in first_axes:
            rl = f"chirps {grp.R.median():.1f} Hz" if chirp else f"{grp.R.median():.0f} Hz"
            first_axes[i].set_ylabel(f"{i + 1}\n{rl}\n{grp.F.median() / 1000:.2f} kHz", fontsize=BODY_PT - 1, labelpad=2)
    if inset is not None:
        Rc_, Fc_ = np.array([g.R.median() for g in groups]), np.array([g.F.median() for g in groups])
        xlo, xhi = Rc_.min() / 1.25, Rc_.max() * 1.25; ylo_, yhi_ = Fc_.min() / 1.2, Fc_.max() * 1.2
        axl = fig.add_axes(rect(fx0 + 4.55, fx0 + 5.35, 1.0 + 0.12, 0.70))
        axl.scatter(inset["R_all"], inset["F_all"], s=0.3, color="0.85", lw=0, rasterized=True)
        axl.scatter(inset["R_s"], inset["F_s"], s=0.5, color="tab:orange", lw=0, alpha=0.6, rasterized=True)
        axl.add_patch(plt.Rectangle((xlo, ylo_), xhi - xlo, yhi_ - ylo_, fc="none", ec="k", lw=0.6, zorder=5))
        axl.set_xscale("log"); axl.set_yscale("log"); axl.set_ylim(1500, 20000)
        if chirp:
            axl.set_xlim(1.2, 12); axl.xaxis.set_major_locator(FixedLocator([2, 5, 10]))
        else:
            axl.set_xlim(10, 100); axl.xaxis.set_major_locator(FixedLocator([10, 30, 100]))
        axl.xaxis.set_minor_locator(FixedLocator([])); axl.yaxis.set_major_locator(FixedLocator([2000, 5000, 20000])); axl.yaxis.set_minor_locator(FixedLocator([]))
        axl.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axl.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axl.tick_params(labelsize=BODY_PT - 1.5, length=1.5, pad=1); axl.set_title("the chirp plane" if chirp else "the year plane", fontsize=BODY_PT, pad=2)
        axl.set_xlabel("chirps/s" if chirp else "Hz", fontsize=BODY_PT - 1, labelpad=0); axl.set_ylabel("kHz", fontsize=BODY_PT - 1, labelpad=0)
        axi = fig.add_axes(rect(fx0 + 5.75, fx0 + Wc - 0.03, 1.0 + 0.12, 0.70))
        axi.scatter(inset["R_all"], inset["F_all"], s=0.6, color="0.85", lw=0, rasterized=True)
        axi.scatter(inset["R_s"], inset["F_s"], s=1.0, color="tab:orange", lw=0, alpha=0.6, rasterized=True)
        axi.scatter(Rc_, Fc_, s=16, facecolors="none", edgecolors="k", lw=0.6, zorder=5)
        axi.set_xscale("log"); axi.set_yscale("log"); axi.set_xlim(xlo, xhi); axi.set_ylim(ylo_, yhi_)
        xt = [t for t in ([1.5, 2, 2.5, 3, 4, 5, 6, 8, 10] if chirp else [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]) if xlo <= t <= xhi]
        yt = [t for t in [1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000] if ylo_ <= t <= yhi_]
        axi.xaxis.set_major_locator(FixedLocator(xt)); axi.yaxis.set_major_locator(FixedLocator(yt)); axi.xaxis.set_minor_locator(FixedLocator([])); axi.yaxis.set_minor_locator(FixedLocator([]))
        axi.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); axi.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axi.tick_params(labelsize=BODY_PT - 1.5, length=1.5, pad=1); axi.set_title("the rows on the streak, in order", fontsize=BODY_PT, pad=2)
    base = os.path.join(out_dir, f"{S_row.get('fname', f'streak_{sid}')}{('_' + tag) if tag else ''}_grid{'_sync' if sync else ''}")
    page.finish(fig, base)
    plt.close(fig)
    if audio_rows:
        pd.DataFrame(audio_rows).to_csv(os.path.join(out_dir, "audio", S_row.get("fname", f"streak_{sid}"), "index.csv"), index=False)
    for ext in ("_content.pdf", "_content.png"):               # the bare canvases would double the files in the supplement folder
        if os.path.exists(base + ext):
            os.remove(base + ext)
    return sid, base + ".png"


def _safe(args):
    try:
        return draw_grid(args)
    except Exception as e:  # noqa: BLE001
        plt.close("all")
        return args[0], None, f"FAILED {type(e).__name__}: {str(e)[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--streaks", type=int, nargs="*", default=None); ap.add_argument("--rows", type=int, default=12); ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--dur", type=float, default=0.5); ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--sync", type=int, default=0, help="k > 0: period-synchronous average over 2k+1 periods at the track's rate (a labeled filtered view); 0 = raw")
    ap.add_argument("--out-dir", default=None, help="default <root>/grid")
    ap.add_argument("--inset", action="store_true", help="mark the rows' positions on the year plane in a corner inset")
    ap.add_argument("--vector-pdf", action="store_true", help="also write a PDF per streak with editable text (pictures rasterized)")
    ap.add_argument("--period", default="night", choices=["night", "day"], help="day: the DAY singer-periods of each type (animal_days_streaks<suffix>.csv), pages tagged _day")
    ap.add_argument("--nights", nargs=2, default=None, metavar=("FIRST", "LAST"), help="use only singer-periods from nights in this date range (a season of a biphasic streak)")
    ap.add_argument("--tag", default="", help="suffix for the output file names, e.g. spring")
    ap.add_argument("--form", default=None, choices=["chirp", "trill"],
                    help="only singer-periods of this song form: chirp = a chirp rate in >= FORM_CHIRP of its hours, trill = in < FORM_TRILL (the streak's coordinates are rate and carrier; the chirp rhythm is a third axis the streak ignores, MBE 2026-10-06)")
    ap.add_argument("--letter", action="store_true", help="8.5 x 11 pages in the house style (paper_page.PaperPage) instead of the large working pages; the combined PDF is then the pages merged")
    ap.add_argument("--audio", action="store_true", help=f"with --letter: save a {CLIP_S:g} s listening clip (16-bit WAV, peak-normalized) around every drawn cell in <out-dir>/audio/<page>/, with an index.csv")
    ap.add_argument("--chirp-plane", action="store_true",
                    help="the chirp types of the chirp plane (<root>/chirp, ids C1..): positions along the line in log carrier against log CHIRP rate, cells from windows with a chirp rate, pages named type_C<n>; use --dur 1.0 (MBE 2026-10-13: QC grids for the chirp types too)")
    args = ap.parse_args()
    root = args.root
    if args.chirp_plane:
        cdir = os.path.join(root, "chirp")
        A = pd.read_csv(os.path.join(cdir, "animal_nights_streaks_c2.csv"))
        cid = pd.read_csv(os.path.join(cdir, "paper_ids_chirp.csv")).set_index("streak").paper_id.to_dict()
    else:
        A = pd.read_csv(os.path.join(root, f"animal_{'days' if args.period == 'day' else 'nights'}_streaks{args.suffix}.csv"))
        cid = {}
    ids = {} if args.chirp_plane else paper_ids.load(root, args.suffix)
    if "depth" not in A.columns:                     # the day table carries hours, not depth: rank by hours
        A["depth"] = A.n_hours.astype(float)
    season = "; DAY singer-periods" if args.period == "day" else ""
    if args.form:
        Hh = pd.read_csv(os.path.join(root, "streak_hourly_points_all.csv"), usecols=["an", "rc_frac"])
        rc = Hh.groupby("an").rc_frac.mean()
        A["rc"] = A.an.map(rc).fillna(0.0)
        A = A[A.rc >= FORM_CHIRP] if args.form == "chirp" else A[A.rc < FORM_TRILL]
        season += f"; {args.form}ers only (chirp rate in {'>= ' + format(FORM_CHIRP, '.0%') if args.form == 'chirp' else '< ' + format(FORM_TRILL, '.0%')} of the singer's hours)"
    if args.nights:
        nd = pd.to_datetime(A.night); A = A[(nd >= pd.Timestamp(args.nights[0])) & (nd <= pd.Timestamp(args.nights[1]))]
        season = f"; nights {args.nights[0]} to {args.nights[1]} only"
    S = pd.read_csv(os.path.join(cdir if args.chirp_plane else root, "streaks_coarse_c2.csv" if args.chirp_plane else f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[S.harmonic_of < 0] if "harmonic_of" in S.columns else S
    mp = os.path.join(root, "streak_sina_matches.csv")
    cands = pd.read_csv(mp).query("rank == 1").set_index("streak").species.to_dict() if (os.path.exists(mp) and not args.chirp_plane) else {}
    out_dir = args.out_dir or os.path.join(root, "grid"); os.makedirs(out_dir, exist_ok=True)
    P_all = None
    if args.inset:
        P_all = pd.read_csv(os.path.join(cdir if args.chirp_plane else root, "streak_hourly_points_all.csv"), usecols=["R", "F", "an"]).drop_duplicates().merge(A[["an", "streak"]], on="an", how="left")
        sub = P_all.sample(min(len(P_all), 30000), random_state=0)
    pp = os.path.join(root, f"pulse_shapes{args.suffix}.csv")
    clear = None
    if os.path.exists(pp):
        Pq = pd.read_csv(pp); clear = set(Pq[(Pq.status == "ok") & (Pq.snr_db >= 6)].an)
    sids = args.streaks or [s for s in S.sort_values("n", ascending=False).index if S.loc[s, "n"] >= args.min_n]
    jobs = []
    for sid in sids:
        g = A[A.streak == sid]
        # v8 tables: a type has a form (streaks_coarse 'form') and each member a form; show FIRM members of the type's form only --
        # the mixed singer-periods assigned to a trill type can chirp (MBE, 2026-10-06: "4001 in v8 still had trills and chirps together")
        if "form" in A.columns and "form" in S.columns and args.form is None:
            g = g[g.form == S.loc[sid, "form"]]
        c = CHIRP_C if args.chirp_plane else (C_LO if S.loc[sid, "F25"] < HF_SPLIT else C_HI)
        groups = choose_grid(g, c, args.rows, args.cols, clear)
        if groups:
            inset = None
            if P_all is not None:
                gs_ = P_all[P_all.streak == sid]
                inset = dict(R_all=sub.R.to_numpy(), F_all=sub.F.to_numpy(), R_s=gs_.R.to_numpy(), F_s=gs_.F.to_numpy())
            # the type's song form (v7 tables carry a per-member form): a chirp type's cells come from windows with a chirp rate, a trill type's from windows without
            tform = "chirp" if args.chirp_plane else (args.form or (S.loc[sid, "form"] if "form" in S.columns else None))
            season_s = season + (f"; firm {tform}ers only, cells from windows {'with' if tform == 'chirp' else 'without'} a chirp rate" if tform else "")
            if args.chirp_plane:
                season_s = season + "; chirp plane: positions along the line in carrier against CHIRP rate; cells from windows with a chirp rate (c = the detector's chirp rate)"
            # pages named by the paper's type number (MBE, 2026-10-13: streak ids are hard to parse): type_05_streak_3601, type_C3; a
            # main-plane streak without a paper number (the chirp-form ones, numbered on the chirp plane) keeps streak_<id>
            if args.chirp_plane:
                pid = cid.get(sid, sid); label, fname = f"type {pid} (chirp plane streak {sid})", f"type_{pid}"
            elif sid in ids:
                pid = ids[sid]; label, fname = f"type {pid} (streak {sid})", f"type_{pid:02d}_streak_{sid}" if isinstance(pid, int) else f"type_{pid}_streak_{sid}"
            else:
                pid = f"streak-{sid}"; label, fname = f"streak {sid} (chirp-form; its type is on the chirp plane)", f"streak_{sid}"
            srow = dict(S.loc[sid].to_dict(), season=season_s, n_season=len(g), chirp_plane=args.chirp_plane, label=label, fname=fname, letter=args.letter, pid=pid, audio=args.audio)
            jobs.append((cid.get(sid, sid) if args.chirp_plane else sid, groups, srow, cands.get(sid, ""), args.unit_roots, out_dir, args.dur, args.sync, inset, args.vector_pdf, args.tag or (args.form or "") or ("day" if args.period == "day" else ""), tform))
    print(f"{len(jobs)} streaks, up to {args.rows} x {args.cols} cells each", flush=True)
    pages = []
    with Pool(args.jobs) as pool:
        for sid, p, n in pool.imap(_safe, jobs):
            print(f"  streak {sid}: {n}" + ("" if p else " (no page)"), flush=True)
            if p:
                pages.append(p)
    if not args.streaks:
        pdfp = os.path.join(out_dir if args.out_dir else root, f"grid{'_sync' if args.sync else ''}{args.suffix}.pdf")
        def page_key(q):                       # type_05_streak_3601 -> (0, 5); type_C3 -> (1, 3); streak_3151 -> (2, 3151)
            parts = os.path.basename(q).split("_")
            if parts[0] == "type":
                return (1, int(parts[1][1:])) if parts[1].startswith("C") else (0, int(parts[1]))
            return (2, int(parts[1]))
        pages.sort(key=page_key)
        if args.period == "day":
            pdfp = pdfp.replace(".pdf", "_day.pdf")      # pages in id order (the pool returns them in finishing order; MBE: "random order")
        if args.letter:                              # the letter pages are PDFs already: merge them
            from pypdf import PdfReader, PdfWriter
            wr = PdfWriter()
            for p in pages:
                for pg in PdfReader(p[:-4] + ".pdf").pages:
                    wr.add_page(pg)
            with open(pdfp, "wb") as fh:
                wr.write(fh)
        else:
            with PdfPages(pdfp) as pdf:
                for p in pages:
                    img = plt.imread(p)
                    fig = plt.figure(figsize=(img.shape[1] / 120, img.shape[0] / 120)); fig.figimage(img, resize=True); pdf.savefig(fig); plt.close(fig)
        print("wrote", out_dir, "and", pdfp)


if __name__ == "__main__":
    main()
