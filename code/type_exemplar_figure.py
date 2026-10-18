"""Figure 16 (provisional; the exemplar form of the type reference, numbered 15 until MBE reversed the two on 2026-10-14) (MBE, 2026-10-14: the median folds of type_reference_figure are "too washed out ... maybe just
exemplars of folds from one window instead"): one exemplar window per type, drawn as a short raw excerpt beside its fine
fold, each at its own full dynamic range.

The exemplar is chosen from the type-reference sample (type_reference_figure.py: up to 40 of the type's clearest
singer-nights at 21-29 C, one deepest clean window each, per-window descriptors cached): among windows whose fold
contrast is at least EXEMPLAR_MIN_DB and in the upper half of the type's, the one whose duty, glide and bandwidth are
closest (standardized) to the type's medians -- a typical clear singer, not an extreme one. The window is re-read and
drawn as in Figures 5 and 10: left, N_PERIODS periods of the loudest stretch as a fine spectrogram, the detected band in
color and the rest in grayscale; right, the fold over two periods (syllable period for trills, chirp period for chirp
types), dB above each row's quiet level over the window.

  (A) the 25 trill types   (B) the 6 chirp types

Usage:
  python analysis/callers/type_exemplar_figure.py analysis/callers/runs/v3/year_nov24_nov25 --suffix _v9 \
      --unit-roots analysis/callers/runs/v3/y2024 analysis/callers/runs/v3/y2025 --out analysis/callers/runs/figures_paper/F16_type_exemplars
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_blobs as W  # noqa: E402
import fold_subclass as FS  # noqa: E402
import specdisplay as SD  # noqa: E402
from type_reference_figure import best_window, CH_OFF, NPH_TRILL, NPH_CHIRP  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402

TZ = "America/Chicago"
EXEMPLAR_MIN_DB = 8.0
N_PERIODS_TRILL, N_PERIODS_CHIRP = 6, 2.5
PAD_REL = 0.12            # display band: carrier +/- this fraction (log) around the detected band


def pick_exemplar(D: pd.DataFrame) -> pd.Series:
    Dc = D[D.contrast >= max(EXEMPLAR_MIN_DB, D.contrast.quantile(0.5))]
    if Dc.empty:                                       # a type with few or faint windows: its clearest window
        return D.sort_values("contrast", ascending=False).iloc[0]
    D = Dc
    feats = ["duty", "glide_hz", "bw_hz"]
    X = D[feats].to_numpy(float); med = np.nanmedian(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-9
    d = np.nansum(((X - med) / sd) ** 2, axis=1)
    return D.iloc[int(np.argmin(d))]


def read_exemplar(w: dict, nph: int, n_periods: float):
    """The window's audio once: the loudest 30 s for the band, the refined rate, the fine fold and a fine spectrogram of the first n_periods."""
    sig, sr, t0, _ = W.read_window(w["recorder"], w["site"], w["t"], 300.0)
    f, _, Z = scipy.signal.stft(sig, fs=sr, nperseg=512, noverlap=384, window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2; dt = 128 / sr
    r_lo = max(0, int(np.searchsorted(f, w["F_lo"])) - 1); r_hi = min(len(f) - 1, int(np.searchsorted(f, w["F_hi"])) + 1)
    try:
        cols = W.best_segment(P, r_lo, r_hi, dt, w["R"], 30.0)
    except ValueError:
        cols = slice(0, P.shape[1])
    try:
        R = W.refine_rate(P, r_lo, r_hi, cols, dt, w["R"])
    except ValueError:
        R = w["R"]
    if not (0.97 * w["R"] <= R <= 1.03 * w["R"]):
        R = w["R"]
    quiet = np.percentile(10 * np.log10(P + 1e-20), 10, axis=1)
    a0 = cols.start * 128; a1 = min(len(sig), cols.stop * 128 + 512); seg = sig[a0:a1]
    # fold (fine frames) over the whole loudest stretch
    ff, tf, Zf = scipy.signal.stft(seg, fs=sr, nperseg=FS.NPERSEG, noverlap=FS.NPERSEG - FS.HOP, nfft=FS.NFFT, window="hann", boundary=None, padded=False)
    Pf = np.abs(Zf) ** 2
    lo, hi = w["F_lo"] * (1 - PAD_REL), w["F_hi"] * (1 + PAD_REL); rows = (ff >= lo) & (ff <= hi)
    Pf = Pf[rows]; fk = ff[rows]
    b = np.floor(((tf * R) % 1.0) * nph).astype(int)
    F = np.zeros((Pf.shape[0], nph)); cnt = np.bincount(b, minlength=nph)
    for k in range(nph):
        F[:, k] = Pf[:, b == k].sum(axis=1)
    F /= np.maximum(cnt, 1)
    Fdb = 10 * np.log10(F + 1e-20); Fdb -= np.percentile(Fdb, 10, axis=1, keepdims=True)
    # align the fold so the pulse (envelope over the detected band) starts at phase 0
    band = (fk >= w["F_lo"]) & (fk <= w["F_hi"])
    env = np.clip(Fdb[band], 0, None).sum(axis=0); env = env / (env.max() + 1e-9)
    k_pk = int(np.argmax(env)); k = k_pk
    while env[(k - 1) % nph] >= 0.5 and k > k_pk - nph:
        k -= 1
    Fdb = np.roll(Fdb, -(k % nph), axis=1)
    # the excerpt: the n_periods-long stretch inside the loudest 30 s with the most band power
    n_ex = int(round(n_periods / R * sr)); band_rows = (f >= w["F_lo"]) & (f <= w["F_hi"])
    bp = P[band_rows][:, cols].sum(axis=0); n_cols = max(1, int(round(n_periods / R / dt)))
    kk = int(np.argmax(np.convolve(bp, np.ones(n_cols), "valid"))) if len(bp) > n_cols else 0
    e0 = a0 + kk * 128; ex = sig[e0:e0 + n_ex]
    fe, te, Pe = SD.fine_spec(ex, sr); De = SD.db_above_quiet(Pe, SD.quiet_on_grid(fe, f, quiet))
    T_local = w["t"].tz_convert(TZ) if w["t"].tzinfo else w["t"]
    return dict(Fdb=Fdb, fk=fk, R=R, fe=fe, te=te, De=De, lo=lo, hi=hi, t=T_local, period_ms=1000 / R)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v9"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--number", default="16"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root; cache = os.path.join(root, "type_reference")
    meta = pd.read_csv(os.path.join(cache, "sample.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    Sc = pd.read_csv(os.path.join(root, "chirp", "streaks_coarse_c2.csv"), index_col=0); Sc.index = Sc.index + CH_OFF
    SS = pd.concat([S, Sc]); meta = meta[meta.key.isin(SS.index)]
    hourly = pd.read_csv(os.path.join(root, "chirp", "streak_hourly_points_all.csv"), usecols=["hour", "R", "Rs", "an"])
    hourly["hour"] = pd.to_datetime(hourly.hour, utc=True, format="ISO8601").dt.tz_convert(TZ)
    wx = pd.concat([pd.read_csv(os.path.join(r, "weather_hourly_hrrr.csv")) for r in args.unit_roots if os.path.exists(os.path.join(r, "weather_hourly_hrrr.csv"))]) if True else None
    if wx is not None and len(wx):
        wx["t"] = pd.to_datetime(wx.t, utc=True); wx = wx.dropna(subset=["T_C"]).drop_duplicates("t").sort_values("t"); wt, wT = wx.t.astype("int64").to_numpy(), wx.T_C.to_numpy()
    tr = meta[meta.form == "trill"].copy(); tr["num"] = tr.paper_id.astype(int); tr = tr.sort_values("num")
    ch = meta[meta.form == "chirp"].copy(); ch["num"] = ch.paper_id.str.lstrip("C").astype(int); ch = ch.sort_values("num")
    tp = os.path.join(os.path.dirname(args.out), "F15_type_reference_table.csv")
    shapes = pd.read_csv(tp).assign(type=lambda d: d.type.astype(str)).set_index("type")["shape"].to_dict() if os.path.exists(tp) else {}
    cells = {}
    for mt in list(tr.itertuples()) + list(ch.itertuples()):
        D = pd.read_csv(os.path.join(cache, f"windows_{int(mt.key)}.csv"))
        if D.empty:
            continue
        ex = pick_exemplar(D); form = mt.form
        w = best_window(ex.an, args.unit_roots, form, hourly if form == "chirp" else None)
        if w is None:
            print(f"  {mt.paper_id}: no window for {ex.an}"); continue
        try:
            c = read_exemplar(w, NPH_CHIRP if form == "chirp" else NPH_TRILL, N_PERIODS_CHIRP if form == "chirp" else N_PERIODS_TRILL)
        except Exception as e:  # noqa: BLE001
            print(f"  {mt.paper_id}: {type(e).__name__} {e}"); continue
        c["T"] = float(np.interp(pd.Timestamp(w["t"]).tz_convert("UTC").value, wt, wT)) if wx is not None and len(wx) else np.nan
        c["an"] = ex.an; c["contrast"] = float(ex.contrast); cells[int(mt.key)] = c
        print(f"  {mt.paper_id:>3s}: {ex.an}  {c['t']:%Y-%m-%d %H:%M}  {c['T']:.0f} C  R {c['R']:.1f}  contrast {ex.contrast:.0f} dB")

    page = PaperPage(number=args.number, title="The call types: one exemplar singer of each")
    Wc, Hc = 7.2, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]
    NCOL = 5; X0, X1 = 1.05, 7.35; CW = (X1 - X0) / NCOL
    EXC_W, FOLD_W, GAPW = 0.78, 0.40, 0.04; HEAD = 0.21                               # HEAD holds the two-line head
    # row pitch from the page: 37 trill types (8 rows of 5) + 2 chirp rows overran the fixed 0.92 in pitch and the chirp row fell off the
    # page (MBE 2026-10-16: "the chirp types in Figure 17 were cut off so I just deleted them")
    n_rows_a_ = int(np.ceil(len(tr) / NCOL)); n_rows_b_ = int(np.ceil(len(ch) / NCOL))
    PITCH = (Hc - 0.15 - 0.30 - 0.12) / (n_rows_a_ + n_rows_b_); CELL_H = PITCH - HEAD - 0.11

    def draw(mt, x0, top, form, first_col):
        k = int(mt.key); c = cells.get(k); F25, R25 = SS.loc[k, "F25"], SS.loc[k, "R25"]
        axe = fig.add_axes(rect(x0, x0 + EXC_W, top + HEAD, CELL_H)); axf = fig.add_axes(rect(x0 + EXC_W + GAPW, x0 + EXC_W + GAPW + FOLD_W, top + HEAD, CELL_H))
        head = f"{mt.paper_id}  {F25 / 1000:.1f} kHz @ {R25:.0f} Hz" if form == "trill" else f"{mt.paper_id}  {F25 / 1000:.1f} kHz @ {R25:.1f} Hz"
        if c is not None:
            head += f"   {c['T']:.0f} °C"
        if isinstance(shapes.get(str(mt.paper_id)), str):          # the five types split off on 2026-10-16 have no shape word yet                      # the syllable shape word of the type table (flat / downslur / hockey stick / chevron) on its own line: beside the numbers it ran into the next cell
            head += f"\n{shapes[str(mt.paper_id)]}"
        axe.set_title(head, fontsize=BODY_PT - 0.5, loc="left", pad=1.5, linespacing=1.1)
        if c is None:
            axe.text(0.5, 0.5, "no exemplar", ha="center", va="center", fontsize=BODY_PT, transform=axe.transAxes); axf.set_axis_off(); return axe
        lo, hi = c["lo"], c["hi"]
        sel = (c["fe"] >= lo) & (c["fe"] <= hi); inb = (c["fe"][sel] >= lo / (1 - PAD_REL)) & (c["fe"][sel] <= hi / (1 + PAD_REL))
        De = c["De"][sel]; vmax = max(12.0, float(np.percentile(De[inb] if inb.any() else De, 99.5)))
        axe.pcolormesh(c["te"], c["fe"][sel], np.ma.masked_array(De, mask=np.broadcast_to(inb[:, None], De.shape)), cmap="gray", vmin=0, vmax=vmax, shading="auto", rasterized=True)
        axe.pcolormesh(c["te"], c["fe"][sel], np.ma.masked_array(De, mask=np.broadcast_to(~inb[:, None], De.shape)), cmap="magma", vmin=0, vmax=vmax, shading="auto", rasterized=True)
        axe.set_yscale("log"); axe.set_ylim(lo, hi); axe.set_xlim(0, c["te"][-1])
        ticks = [t for t in (1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000) if lo <= t <= hi]
        axe.yaxis.set_major_locator(FixedLocator(ticks[:3] if len(ticks) > 3 else ticks)); axe.yaxis.set_minor_locator(FixedLocator([])); axe.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        axe.tick_params(labelsize=BODY_PT - 1, length=1.5, pad=1.5); axe.set_xticks([0, round(c["te"][-1], 2)])
        axe.set_xticklabels(["0", f"{c['te'][-1]:.2f} s"])
        if first_col:
            axe.set_ylabel("kHz", fontsize=BODY_PT - 1, labelpad=1)
        else:
            axe.set_yticklabels([])                    # a label here would print over the previous cell's fold; the head gives the carrier
        # fold: two periods, dB above quiet, own ceiling
        Fdb = c["Fdb"]; per = c["period_ms"]
        axf.imshow(np.concatenate([Fdb, Fdb], axis=1), origin="lower", aspect="auto", extent=[0, 2 * per, c["fk"][0], c["fk"][-1]], cmap="magma", vmin=0, vmax=max(6.0, float(np.percentile(Fdb, 99.5))), interpolation="nearest", rasterized=True)
        axf.axvline(per, color="w", lw=0.4, alpha=0.6); axf.set_yscale("log"); axf.set_ylim(lo, hi); axf.yaxis.set_major_locator(FixedLocator([])); axf.yaxis.set_minor_locator(FixedLocator([]))
        axf.set_xticks([round(per)]); axf.set_xticklabels([f"{per:.0f} ms"]); axf.tick_params(labelsize=BODY_PT - 1.5, length=1.5, pad=1.5)   # one period marked; a "0" collided with the excerpt's end label
        return axe

    top0 = 1.15; n_rows_a = int(np.ceil(len(tr) / NCOL)); first_ax = None
    for i, mt in enumerate(tr.itertuples()):
        rr, cc = divmod(i, NCOL); ax = draw(mt, X0 + cc * CW, top0 + rr * PITCH, "trill", cc == 0)
        first_ax = first_ax or ax
    page.label(first_ax, "A", "Trill types: an exemplar's excerpt and its fold over two syllable periods", y=1 + (HEAD + 0.06) / CELL_H)
    top1 = top0 + n_rows_a * PITCH + 0.3; first_b = None
    for i, mt in enumerate(ch.itertuples()):
        rr, cc = divmod(i, NCOL); ax = draw(mt, X0 + cc * CW, top1 + rr * PITCH, "chirp", cc == 0)
        first_b = first_b or ax
    page.label(first_b, "B", "Chirp types: an exemplar's excerpt and its fold over two chirp periods", y=1 + (HEAD + 0.06) / CELL_H)
    page.finish(fig, args.out)
    pd.DataFrame([dict(key=k, an=c["an"], t=str(c["t"]), T=c["T"], R=c["R"], contrast=c["contrast"]) for k, c in cells.items()]).to_csv(args.out + "_exemplars.csv", index=False)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
