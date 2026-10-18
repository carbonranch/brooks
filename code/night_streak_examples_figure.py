"""Streak examples from one night before any linking: for each cluster marked on the night's carrier-rate plane,
the strongest window-animal early, in the middle and late in the night, shown as an excerpt of its band (dB above the
window's quiet level) beside its fold (the detector's fold over the animal's on-period runs at its refined rate; for
the chirper, at its chirp rate). Used by night_streaks_figure.py (Figure 5) and runnable alone.

Streaks are bands around a line F = F0 (R / R0)^b on the night's plane, +/- F_tol in carrier, between R_lo and R_hi;
the chirper cluster by chirp rate and carrier (STREAKS, CHIRPER: MBE's marks of 2026-10-10). In each third of the
cluster's own time span the strongest kept window-animal (map peak height) is shown, if it sings at least MIN_ON of
its window, is not a 2x/3x rate image of another singer in its rows, is not the spill of a stronger same-rate
neighbor in adjacent rows, and its fold over the band +/- PAD_HZ peaks inside its own band.

Usage (alone, letter page):
    python analysis/callers/night_streak_examples_figure.py --unit-dir analysis/callers/runs/v3/y2025/BARLT_00008662_House_20250920_night \
        --weather analysis/callers/runs/v2/y2025/weather_hourly_hrrr.csv --out analysis/callers/runs/figures_paper/F04b_night_streak_examples
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_blobs2 as W2  # noqa: E402
import specdisplay as SD  # noqa: E402
from window_figure import logband, DF, PAD_HZ, excerpt_start  # noqa: E402
from paper_page import PaperPage, PAGE_DPI, BODY_PT  # noqa: E402

# the night's clusters as MBE marked them on the plane (2026-10-10): a band around the line F = F0 (R / R0)^b,
# +/- F_tol in carrier, between R_lo and R_hi; the chirper cluster by chirp rate and carrier
STREAKS = [
    dict(F0=5540.0, R0=17.5, b=0.79, R_lo=12.0, R_hi=27.0, F_tol=0.045, name="Triller streak from 5.2 kHz at 16 Hz to 6.6 kHz at 22 Hz"),
    # the line from 4.7 kHz at 26 Hz to 6.3 kHz at 44 Hz (F0 5280 at R0 32, b 0.56) was dropped 2026-10-10: loud only at dusk on this night
    dict(F0=3250.0, R0=32.0, b=0.84, R_lo=26.0, R_hi=48.0, F_tol=0.07, name="Triller streak from 2.9 kHz at 28 Hz to 4.4 kHz at 46 Hz"),
    dict(F0=3500.0, R0=46.0, b=0.72, R_lo=34.0, R_hi=64.0, F_tol=0.06, name="Triller streak from 2.9 kHz at 36 Hz to 4.3 kHz at 62 Hz"),
    dict(F0=2520.0, R0=40.0, b=0.61, R_lo=30.0, R_hi=58.0, F_tol=0.07, name="Triller streak from 2.2 kHz at 32 Hz to 3.1 kHz at 56 Hz"),
]
CHIRPER = dict(F_lo=7200.0, F_hi=8600.0, Rc_lo=2.5, Rc_hi=4.3, name="3 Hz chirper at 7.3 to 8.5 kHz")
MIN_ON = 0.3                    # an example must sing at least this fraction of its window
EXCERPT_MAX_S = 2.0             # longest excerpt
PX_PER_CYCLE = 8                # an excerpt is cut so that one pulse period spans at least this many page pixels (at PAGE_DPI)


def in_streak(W: pd.DataFrame, s: dict) -> pd.Series:
    F_line = s["F0"] * (W.R / s["R0"]) ** s["b"]
    return W.R.between(s["R_lo"], s["R_hi"]) & ((W.F_c / F_line - 1).abs() <= s["F_tol"])


def in_chirper(W: pd.DataFrame, c: dict = CHIRPER) -> pd.Series:
    return W.chirp_rate.between(c["Rc_lo"], c["Rc_hi"]) & W.F_c.between(c["F_lo"], c["F_hi"]) & W.bout_rate.isna()


def rate_multiple(W: pd.DataFrame) -> pd.Series:
    """True where a window-animal's rate is 2x or 3x (within 3 %) that of another kept animal in the same window
    whose band overlaps its own: the unabsorbed rate harmonics of the chorus sit on the night's plane at twice the
    chorus's rate and would be picked as 'strong' members of the line above it."""
    out = pd.Series(False, index=W.index)
    for t, g in W.groupby("t"):
        R = g.R.values; lo = g.F_lo.values; hi = g.F_hi.values
        for i in range(len(g)):
            if not np.isfinite(R[i]):
                continue
            for m in (2.0, 3.0):
                cand = np.isfinite(R) & (np.abs(R[i] / (m * R) - 1) < 0.03) & (lo <= hi[i]) & (hi >= lo[i])
                cand[i] = False
                if cand.any():
                    out.loc[g.index[i]] = True
                    break
    return out


def animal_strength(unit_dir: str) -> pd.DataFrame:
    """Per (window time, animal): the map peak height (dB) of its strongest fundamental part's peak, from the
    per-window peak and parts files -- how strong the signal is, which modulation depth does not measure."""
    rows = []
    for pp in sorted(glob.glob(os.path.join(unit_dir, "BARLT_*_night_parts.csv"))):
        m = re.search(r"_(\d{8})_(\d{4})_night_parts", pp)
        t = pd.Timestamp(f"{m.group(1)} {m.group(2)[:2]}:{m.group(2)[2:]}", tz=W2.TZ)
        P = pd.read_csv(pp); B = pd.read_csv(pp.replace("_parts.csv", ".csv"))
        if P.empty or B.empty:
            continue
        P = P[P.am_harm_of < 0]
        P["peak_db"] = P.blob.map(B.set_index("blob").peak_db)
        g = P.groupby("animal").peak_db.max()
        rows.append(pd.DataFrame({"t": t, "animal": g.index.astype(int), "peak_db": g.values}))
    return pd.concat(rows, ignore_index=True)


def rate_shadow(W: pd.DataFrame) -> pd.Series:
    """True where a stronger kept animal in the same window sings at the same rate (within 3 %) in rows within
    PAD_HZ of this one's band: this one's fold would be dominated by the neighbor outside its own band."""
    out = pd.Series(False, index=W.index)
    for t, g in W.groupby("t"):
        R = g.R.values; lo = g.F_lo.values; hi = g.F_hi.values; pk = g.peak_db.values
        for i in range(len(g)):
            if not np.isfinite(R[i]):
                continue
            cand = np.isfinite(R) & (np.abs(R[i] / R - 1) < 0.03) & (lo <= hi[i] + PAD_HZ) & (hi >= lo[i] - PAD_HZ) & (pk > pk[i])
            cand[i] = False
            if cand.any():
                out.loc[g.index[i]] = True
    return out


def picks_for(W: pd.DataFrame, mask: pd.Series, n: int = 3, ok=None):
    """The strongest member in each of n equal thirds of the cluster's own time span through the night, so a cluster
    that only sings late, or stops at midnight, still shows its warm, middle and cool examples; picks at least an
    hour apart where the cluster allows; candidates tried strongest-first until one passes ok()."""
    M = W[mask].sort_values("t")
    if M.empty:
        return []
    out, used = [], []
    bounds = np.linspace(M.t.astype("int64").min(), M.t.astype("int64").max(), n + 1)
    for k in range(n):
        c = M[(M.t.astype("int64") >= bounds[k]) & (M.t.astype("int64") <= bounds[k + 1])]
        for tu in used:
            c = c[(c.t - tu).abs() > pd.Timedelta(hours=1)]
        if c.empty:
            continue
        for _, a in c.sort_values("peak_db", ascending=False).iterrows():
            if ok is None or ok(a):
                out.append(a); used.append(a.t); break
    return sorted(out, key=lambda a: a.t)


class Examples:
    """The night's candidate window-animals and the window analyses behind the chosen examples."""

    def __init__(self, unit_dir: str, weather_csv: str, streaks=STREAKS, chirper=CHIRPER):
        base = os.path.basename(os.path.normpath(unit_dir)); parts_ = base.split("_")
        self.rec, self.site = parts_[0] + "_" + parts_[1], parts_[2]
        W = pd.read_csv(os.path.join(unit_dir, "night_night_global_window_animals.csv")); W["t"] = pd.to_datetime(W.t, utc=True).dt.tz_convert(W2.TZ)
        W = W[W.keep.astype(bool)]
        W = W.merge(animal_strength(unit_dir), on=["t", "animal"], how="left")
        W = W[W.peak_db.notna() & (W.on_frac >= MIN_ON)]
        W = W[~rate_multiple(W)]
        W = W[~rate_shadow(W)]
        self.W = W
        Wx = pd.read_csv(weather_csv); Wx["t"] = pd.to_datetime(Wx.t, utc=True).dt.tz_convert(W2.TZ); self.Wx = Wx.dropna(subset=["T_C"]).sort_values("t")
        self.cache, self.resolved = {}, {}
        self.rows = []
        for s in streaks:
            self.rows.append((s, picks_for(W, in_streak(W, s) & W.R.notna(), ok=lambda a, s=s: self.resolve(a, s) is not None)))
        self.rows.append((chirper, picks_for(W, in_chirper(W, chirper), ok=lambda a: self.resolve(a, chirper) is not None)))
        self.chirper = chirper

    def temp_at(self, t):
        return float(np.interp(t.value, self.Wx.t.astype("int64"), self.Wx.T_C))

    def window(self, t):
        if t not in self.cache:
            sig, sr, t0, _ = W2.read_window(self.rec, self.site, t, 300.0); X = W2.analyse_signal(sig, sr, 300.0); X["sig"], X["sr"] = sig, sr
            X["quiet"] = np.percentile(10 * np.log10(X["P"] + 1e-20), 10, axis=1); self.cache[t] = X
        return self.cache[t]

    def resolve(self, a, s):
        """(a, X, part) for pick a, or None when the animal is not re-found in the window or its fold over the band
        +/- PAD_HZ peaks outside its own band (a member whose fold is owned by a louder same-rate neighbor)."""
        key = (a.t, int(a.animal), id(s))
        if key in self.resolved:
            return self.resolved[key]
        X = self.window(a.t); B, parts = X["B"], X["parts"].copy(); parts["R"] = parts.blob.map(B.R_fine)
        fund = parts[parts.am_harm_of < 0]
        is_chirp = s is self.chirper if hasattr(self, "chirper") else "Rc_lo" in s
        R_use = float(a.chirp_rate) if is_chirp else float(a.R)
        cand = fund[(fund.f_c / a.F_c - 1).abs() < 0.06].copy()
        cand["dr"] = (cand.R / R_use - 1).abs()
        cand = cand[cand.dr < 0.05].sort_values("dr")
        out = None
        if not cand.empty:
            p = cand.iloc[0]; f, P, dt = X["f"], X["P"], X["dt"]; d = X["per_blob"][int(p.blob)]
            p_lo = max(0, int(np.searchsorted(f, p.f_c - PAD_HZ)) - 1); p_hi = min(len(f) - 1, int(np.searchsorted(f, p.f_c + PAD_HZ)) + 1)
            Fm = W2.fold_runs(P, p_lo, p_hi, d["cols"], dt, float(p.R))
            Fdb = 10 * np.log10(Fm + 1e-20); Fdb -= np.percentile(Fdb, 10, axis=1, keepdims=True)
            fk = f[p_lo:p_hi + 1]; top = fk[int(np.argmax(Fdb.max(axis=1)))]
            if p.f_lo - DF <= top <= p.f_hi + DF:
                out = (a, X, p)
            else:
                print(f"  skip {a.t:%H:%M} {a.F_c:.0f} Hz: fold peaks at {top:.0f} Hz, outside {p.f_lo:.0f}-{p.f_hi:.0f}")
        self.resolved[key] = out
        return out


def draw_row(fig, ex: Examples, s: dict, picks, cells, last_row: bool, label_first: bool = True):
    """One cluster's examples. cells: list of (spectrogram rect, fold rect) figure rects, one per pick. Each cell is
    headed by its time, temperature, carrier and rate; the fold has no title."""
    W2.PHASE_BINS = 48
    is_chirp = "Rc_lo" in s
    resolved = [ex.resolve(a, s) for a in picks]
    found = [c for c in resolved if c is not None]
    lo = min(c[2].f_c for c in found) - PAD_HZ if found else 0; hi = max(c[2].f_c for c in found) + PAD_HZ if found else 1   # one frequency axis per row
    R_max = max([float(a.R) for a in picks if np.isfinite(a.R)] or [20.0])
    for j, (c, (r_s, r_f)) in enumerate(zip(resolved, cells)):
        ax_s = fig.add_axes(r_s); ax_f = fig.add_axes(r_f)
        if c is None:
            ax_s.set_axis_off(); ax_f.set_axis_off(); ax_s.text(0.5, 0.5, "animal not re-found", ha="center", va="center", transform=ax_s.transAxes); continue
        a, X, p = c; t = a.t
        f, P, dt, per_blob = (X[k] for k in ("f", "P", "dt", "per_blob"))
        d = per_blob[int(p.blob)]; R = float(p.R)
        n_px = int(round(r_s[2] * fig.get_figwidth() * PAGE_DPI))
        R_pulse = float(a.R) if np.isfinite(a.R) else R
        T = EXCERPT_MAX_S if is_chirp else min(EXCERPT_MAX_S, n_px / (PX_PER_CYCLE * R_max))
        s0 = excerpt_start(d, f, p, dt, T, 300.0)
        sig, sr = X["sig"], X["sr"]
        fd, td, Pd = SD.fine_spec(sig[int(s0 * sr):int((s0 + T) * sr)], sr, hop=64)
        Dd = SD.db_above_quiet(Pd, SD.quiet_on_grid(fd, f, X["quiet"])); band = (fd >= lo) & (fd <= hi)
        img = Dd[band]; dcol = td[1] - td[0]
        k = max(1, img.shape[1] // n_px); n = (img.shape[1] // k) * k
        blk = img[:, :n].reshape(img.shape[0], -1, k)
        img = blk.max(axis=2) if (1.0 / R_pulse) >= 3 * k * dcol else blk.mean(axis=2)
        te = np.arange(img.shape[1] + 1) * k * dcol
        own = (fd[band] >= p.f_lo) & (fd[band] <= p.f_hi)
        vmax = max(12.0, float(np.percentile(img[own] if own.any() else img, 99.5)))
        # the detected band in color, the rows outside it in grayscale on the same scale (as Figures 10 and 13; MBE 2026-10-13: "did we decide not
        # to do the grayscale thing for F05?" -- it had never been applied here)
        edges = np.concatenate([[fd[band][0]], fd[band][1:] - np.diff(fd[band]) / 2, [fd[band][-1]]])
        ax_s.pcolormesh(te, edges, np.ma.masked_array(img, mask=np.broadcast_to(own[:, None], img.shape)), cmap="gray", vmin=0, vmax=vmax, shading="flat", rasterized=True)
        ax_s.pcolormesh(te, edges, np.ma.masked_array(img, mask=np.broadcast_to(~own[:, None], img.shape)), cmap="magma", vmin=0, vmax=vmax, shading="flat", rasterized=True)
        logband(ax_s, lo, hi)
        ax_s.set_xlim(0, T); ax_s.set_xlabel("s" if last_row else "")
        if j == 0 and label_first:
            ax_s.set_ylabel("kHz")
        rate_txt = f"Rc {R:.2f} Hz" if is_chirp else f"{R:.1f} Hz"
        ax_s.set_title(f"{t:%H:%M}, {ex.temp_at(t):.1f} °C: {p.f_c:.0f} Hz, {rate_txt}", loc="left", fontsize=BODY_PT)
        p_lo = max(0, int(np.searchsorted(f, lo)) - 1); p_hi = min(len(f) - 1, int(np.searchsorted(f, hi)) + 1)
        Fm = W2.fold_runs(P, p_lo, p_hi, d["cols"], dt, R)
        Fdb = 10 * np.log10(Fm + 1e-20); Fdb -= np.percentile(Fdb, 10, axis=1, keepdims=True)
        fk = f[p_lo:p_hi + 1]; per = 1000 / R
        ax_f.imshow(np.concatenate([Fdb, Fdb], axis=1), origin="lower", aspect="auto", extent=[0, 2 * per, fk[0] - DF / 2, fk[-1] + DF / 2], cmap="magma", vmin=0, vmax=max(6.0, float(np.percentile(Fdb, 99.5))), interpolation="nearest")
        ax_f.axvline(per, color="w", lw=0.4, alpha=0.6); ax_f.set_yscale("log"); ax_f.set_ylim(lo, hi); ax_f.set_yticks([]); ax_f.yaxis.set_minor_locator(FixedLocator([]))
        ax_f.set_xlabel("ms" if last_row else "")
        for axx in (ax_s, ax_f):
            for sp in axx.spines.values():
                sp.set_linewidth(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-dir", required=True); ap.add_argument("--weather", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ex = Examples(args.unit_dir, args.weather)
    page = PaperPage(number="S1", title="Streak examples from one night before linking")
    Wc, Hc = 7.0, page.size[1]; fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))
    pitch, cell_w, spec_w, fold_w, h = 1.6, 2.05, 0.85, 0.42, 0.95
    for i, (s, picks) in enumerate(ex.rows):
        top = 1.15 + i * pitch
        fig.text((1.05 - fx0) / Wc, 1 - (top - 1.0) / Hc, f"({chr(65 + i)}) {s['name']}", fontsize=8, fontweight="bold", va="bottom", ha="left")
        cells = [([(1.05 + j * cell_w - fx0) / Wc, 1 - (top + 0.3 - 1.0 + h) / Hc, spec_w / Wc, h / Hc],
                  [(1.05 + j * cell_w + spec_w + 0.2 - fx0) / Wc, 1 - (top + 0.3 - 1.0 + h) / Hc, fold_w / Wc, h / Hc]) for j in range(len(picks))]
        draw_row(fig, ex, s, picks, cells, last_row=(i == len(ex.rows) - 1))
    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
