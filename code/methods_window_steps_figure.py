"""Figure 4: overview of the singer identification process, each step on data from the focal night (House 8662,
20 Sep 2025; the 23:05 window).

Layout and wording follow MBE's Figure 4 of 2026-10-10 (MBE_Figures/MBE_F04_ProcessOverview.pdf):
  (A) Modulation map: the window's map with every thresholded peak outlined (peak 9 orange, the trill of (C)
      green, the chirper of (C) and (F) pink)
  (B) Peaks partitioned into parts by rows: peak 9 split into two singers by the on/off pattern of its rows
      (the per-row fold and the per-row on/off series the detector compares)
  (C) Rate harmonics: the trill's spectrum with its peaks at mR, the chirper's comb at every multiple of its chirp rate
  (D) Same-rate peaks at different carriers joined when their rate-locked on/off patterns match: the 2.5 kHz tree
      cricket's 2F and 4F components (joined) and a same-rate piece in the next band (not joined)
  (E) Harmonic post filter: the window's singers on the plane, those flagged as carrier harmonics hollow with an
      arrow to the fundamental and the level difference
  (F) A chirper: the modulation map identifies the chirp rate (purple), not the syllable repeat rate (orange);
      1.5 s of chirps and 0.4 s inside one chirp
  (G) Direct detection of the chirp syllable repeat rate: the band's envelope with chirps shaded and pulses marked,
      one chirp, the inter-pulse intervals

Usage:
    python analysis/callers/methods_window_steps_figure.py --out analysis/callers/runs/figures_paper/F04_process_overview
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_blobs2 as W2  # noqa: E402
import specdisplay as SD  # noqa: E402
import postfilter  # noqa: E402
from window_figure import khz_log, logband, rate_log, DF  # noqa: E402
from paper_page import PaperPage, BODY_PT  # noqa: E402

REC, SITE = "BARLT_00008662", "House"
T_MAIN = "2025-09-20 23:05"
FMIN, FMAX = 1500.0, 12000.0
TRILL_ANIMAL, CHIRP_ANIMAL, SPLIT_PEAK, ROOT_ANIMAL, SEP_ANIMAL = 3, 14, 9, 0, 2
C_TRILL, C_CHIRP, C_SPLIT, C_PURPLE = "tab:green", "tab:pink", "tab:orange", "tab:purple"
# page geometry (inches from the page's top-left), after MBE's layout
X0, X1 = 1.25, 7.55                      # full-width panels
XL = (1.25, 4.20); XR = (4.70, 7.55)     # the two half-width columns
ROWS = {"A": (1.25, 1.30), "BC": (3.30, 0.95), "DE": (5.05, 0.95), "F": (6.55, 0.88), "G": (8.20, 0.73)}
TITLE_DROP = {"BC": 0.08, "DE": 0.08}    # the half-width titles are three lines; they grow upward from this gap above the axes


def analyse(tstr):
    t = pd.Timestamp(tstr, tz=W2.TZ)
    sig, sr, t0, _ = W2.read_window(REC, SITE, t, 300.0)
    X = W2.analyse_signal(sig, sr, 300.0)
    X["sig"], X["sr"], X["t_local"], X["t_center"] = sig, sr, t0.tz_convert(W2.TZ), t
    X["parts"] = X["parts"].copy(); X["parts"]["R"] = X["parts"].blob.map(X["B"].R_fine)
    X["quiet"] = np.percentile(10 * np.log10(X["P"] + 1e-20), 10, axis=1)
    return X


def row_spectrum(X, f_lo, f_hi):
    rows = np.flatnonzero((X["f"] >= f_lo) & (X["f"] <= f_hi))
    return X["excess"][rows].mean(axis=0)


def series(X, f_lo, f_hi, R):
    rows = np.flatnonzero((X["f"] >= f_lo) & (X["f"] <= f_hi))
    s = W2.locked_series(X["P"][rows], X["dt"], R).mean(axis=0)
    return s, W2.highpass(s)


def comb_axis(ax, lo=2.5, hi=90.0):
    ax.axhline(W2.THRESH_DB, color="k", ls=":", lw=0.8)
    rate_log(ax, lo, hi); ax.set_xlabel("modulation rate (Hz)"); ax.set_ylabel("excess (dB)"); ax.grid(alpha=0.25)


def chirper_excerpts(fig, rects, X, an, color):
    """Two excerpts for a chirper: 1.5 s of chirps and 0.4 s inside one chirp, centered where its chirp-rate-locked
    amplitude over its band peaks. rects: two figure rects."""
    f, parts, per_blob, sig, sr = X["f"], X["parts"], X["per_blob"], X["sig"], X["sr"]
    p = parts[(parts.animal == an.name) & (parts.am_harm_of < 0)].sort_values("f_lo").iloc[0]
    d = per_blob[int(p.blob)]; fk = f[d["p_lo"]:d["p_hi"] + 1]; rows = np.flatnonzero((fk >= an.F_lo) & (fk <= an.F_hi))
    s = d["raw"][rows].mean(axis=0); c0 = (int(np.argmax(s)) + 0.5) * W2.SLOW_S
    lo, hi = an.F_lo - 400, an.F_hi + 400
    for rect, (ta, tb), lab, nper, hop in ((rects[0], (c0 - 0.75, c0 + 0.75), "1.5 s: the chirps", 512, 32), (rects[1], (c0 - 0.2, c0 + 0.2), "0.4 s: the syllables inside one chirp", 256, 8)):
        ax = fig.add_axes(rect)
        ta, tb = max(0.0, ta), min(300.0, tb)
        fd, td, Pd = SD.fine_spec(sig[int(ta * sr):int(tb * sr)], sr, nperseg=nper, hop=hop)
        Dd = SD.db_above_quiet(Pd, SD.quiet_on_grid(fd, f, X["quiet"])); band = (fd >= lo) & (fd <= hi)
        ax.pcolormesh(td + ta, fd[band], Dd[band], cmap="magma", vmin=0, vmax=25, shading="auto", rasterized=True)
        logband(ax, lo, hi); ax.set_xlabel("seconds into the window"); ax.set_ylabel("kHz"); ax.set_title(lab, fontsize=BODY_PT, loc="left")
        ax.axhline(an.F_lo - DF / 2, color="w", lw=0.5, ls=":"); ax.axhline(an.F_hi + DF / 2, color="w", lw=0.5, ls=":")
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(1.5)


def chirp_trace(sig, sr, F_lo, F_hi, t_a, t_b, fs_env=4000.0):
    """The time-domain reader's intermediates for drawing: envelope, 3 ms and 40 ms smoothings, threshold, chirp
    segments (index arrays), pulse peak times and heights, inter-pulse intervals (s)."""
    import chirp_channel as CC
    a, b = int(t_a * sr), int(t_b * sr); x = sig[a:b]
    sos = scipy.signal.butter(6, [max(100.0, F_lo), min(sr / 2 - 100.0, F_hi)], btype="band", fs=sr, output="sos")
    xb = scipy.signal.sosfiltfilt(sos, x)
    q = max(1, int(round(sr / fs_env))); env = scipy.signal.decimate(np.abs(scipy.signal.hilbert(xb)), q); fs = sr / q
    fast = scipy.signal.savgol_filter(env, int(CC.PULSE_SMOOTH_S * fs) | 1, 2); slow = scipy.signal.savgol_filter(env, int(CC.CHIRP_SMOOTH_S * fs) | 1, 2)
    lo_, hi_ = np.percentile(slow, 20), np.percentile(slow, 97); thr = lo_ + 0.4 * (hi_ - lo_)
    idx = np.flatnonzero(slow > thr); cuts = np.flatnonzero(np.diff(idx) > int(CC.CHIRP_GAP_S * fs))
    segs = [s_ for s_ in np.split(idx, cuts + 1) if len(s_) and (s_[-1] - s_[0]) / fs >= CC.CHIRP_MIN_S]
    t = t_a + np.arange(len(env)) / fs
    peak_t, peak_v, intervals = [], [], []
    pad = int(0.01 * fs)
    for s_ in segs:
        i0, i1 = max(0, s_[0] - pad), min(len(fast), s_[-1] + pad); y = fast[i0:i1]
        pk, _ = scipy.signal.find_peaks(y, prominence=CC.PULSE_PROM * (y.max() - y.min()), distance=max(1, int(fs / CC.PULSE_MAX_HZ)))
        peak_t += list(t[i0 + pk]); peak_v += list(y[pk])
        if len(pk) >= 2:
            intervals += list(np.diff(pk) / fs)
    return dict(t=t, env=env, fast=fast, slow=slow, thr=thr, segs=segs, peak_t=peak_t, peak_v=peak_v, intervals=intervals)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    X = analyse(T_MAIN)
    f, fr, ex, B, parts, A, per_blob, dt = (X[k] for k in ("f", "fr", "excess", "B", "parts", "A", "per_blob", "dt"))
    Aidx = A.set_index("animal")

    def fund_part(animal):
        return parts[(parts.animal == animal) & (parts.am_harm_of < 0)].sort_values("f_lo").iloc[0]

    page = PaperPage(number=4, title="Overview of singer identification process")
    Wc, Hc = 7.0, page.size[1]
    fx0 = 1.0 + (6.5 - Wc) / 2
    fig = plt.figure(figsize=(Wc, Hc))

    def rect(x0, x1, top, h):
        return [(x0 - fx0) / Wc, 1 - (top - 1.0 + h) / Hc, (x1 - x0) / Wc, h / Hc]

    def title(ax, letter, text, h, drop=0.05, chars=None):
        page.label(ax, letter, text, y=1 + drop / h, chars=chars)

    # ---- (A) the modulation map with every peak outlined ----
    top, h = ROWS["A"]
    ax = fig.add_axes(rect(X0, X1, top, h))
    sel = (f >= FMIN) & (f <= FMAX)
    fe = np.concatenate([[f[sel][0] - DF / 2], f[sel] + DF / 2]); re_ = np.concatenate([[fr[0] - (fr[1] - fr[0]) / 2], fr + (fr[1] - fr[0]) / 2])
    ax.pcolormesh(re_, fe, ex[sel], cmap="gray", vmin=0, vmax=max(15, W2.THRESH_DB * 2.5), shading="flat", rasterized=True)
    special = {int(fund_part(TRILL_ANIMAL).blob): C_TRILL, int(fund_part(CHIRP_ANIMAL).blob): C_CHIRP, SPLIT_PEAK: C_SPLIT}
    for b in B.itertuples():
        c = special.get(int(b.Index), "w")
        ax.add_patch(plt.Rectangle((b.am_lo * 0.97, f[int(b.r_lo)] - DF / 2), b.am_hi * 1.03 - b.am_lo * 0.97, f[int(b.r_hi)] - f[int(b.r_lo)] + DF,
                                   fill=False, lw=1.0 if c != "w" else 0.6, color=c, zorder=3 if c != "w" else 2))
    ax.set_xscale("log"); ax.set_xlim(fr[0], fr[-1]); khz_log(ax, FMIN, FMAX)
    ax.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 30, 50])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel("modulation rate (Hz)"); ax.set_ylabel("carrier (kHz)")
    title(ax, "A", "Modulation map", h)

    # ---- (B) a peak split by rows ----
    top, h = ROWS["BC"]
    k = SPLIT_PEAK if SPLIT_PEAK in per_blob and parts[parts.blob == SPLIT_PEAK].animal.nunique() > 1 else int(parts.groupby("blob").animal.nunique().idxmax())
    d = per_blob[k]; b = B.loc[k]; g = parts[parts.blob == k].sort_values("f_lo")
    fk = f[d["p_lo"]:d["p_hi"] + 1]
    Fdb = 10 * np.log10(d["F"] + 1e-20); Fdb -= np.percentile(Fdb, 10, axis=1, keepdims=True)
    raw = d["raw"] / (np.percentile(d["raw"], 99, axis=1, keepdims=True) + 1e-12)
    wB = XL[1] - XL[0]; w1 = 0.36 * wB
    axb1 = fig.add_axes(rect(XL[0], XL[0] + w1, top, h)); axb2 = fig.add_axes(rect(XL[0] + w1 + 0.12, XL[1], top, h))
    per = 1000 / b.R_fine
    axb1.imshow(np.concatenate([Fdb, Fdb], axis=1), origin="lower", aspect="auto", extent=[0, 2 * per, fk[0] - DF / 2, fk[-1] + DF / 2], cmap="magma", vmin=0, vmax=max(6, np.percentile(Fdb, 99.5)), interpolation="nearest")
    axb1.set_xlabel("ms (two periods)"); axb1.set_ylabel("kHz"); axb1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:.1f}"))
    axb2.imshow(raw, origin="lower", aspect="auto", extent=[0, raw.shape[1] * W2.SLOW_S, fk[0] - DF / 2, fk[-1] + DF / 2], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axb2.set_xlabel("seconds into the window"); axb2.set_yticklabels([])
    cols_b = plt.get_cmap("tab10")
    for i, p in enumerate(g.itertuples()):
        for axx in (axb1, axb2):
            axx.axhline(p.f_lo - DF / 2, color=cols_b(i), lw=1.0); axx.axhline(p.f_hi + DF / 2, color=cols_b(i), lw=1.0)
        axb2.text(3, p.f_c, f"animal {int(p.animal)}", color="w", fontsize=BODY_PT, va="center", fontweight="bold")
    for axx in (axb1, axb2):
        axx.axhline(f[int(b.r_lo)] - DF / 2, color="w", lw=0.6, ls=":"); axx.axhline(f[int(b.r_hi)] + DF / 2, color="w", lw=0.6, ls=":")
    title(axb1, "B", f"Peaks partitioned into parts by rows. Peak {k} (orange above) split into two peaks by comparing on/off pattern of rows.", h, TITLE_DROP["BC"], chars=48)

    # ---- (C) rate harmonics absorbed ----
    ax = fig.add_axes(rect(XR[0], XR[1], top, h))
    pt, pc = fund_part(TRILL_ANIMAL), fund_part(CHIRP_ANIMAL)
    ex3 = row_spectrum(X, pt.f_lo, pt.f_hi); R3 = float(pt.R)
    ex14 = row_spectrum(X, pc.f_lo, pc.f_hi); R14 = float(pc.R)
    ax.plot(fr, ex3, color=C_TRILL, lw=0.9); ax.plot(fr, ex14, color=C_CHIRP, lw=0.9)
    for R, c in ((R3, C_TRILL), (R14, C_CHIRP)):
        ax.axvline(R, color=c, lw=1.0, alpha=0.8)
        for m in range(2, 7):
            if R * m <= fr[-1]:
                ax.axvline(R * m, color=c, lw=0.5, ls=":", alpha=0.8)
    comb_axis(ax)
    title(ax, "C", "Rate harmonics. Peaks for trill (green) and chirp (pink) at mR ± 1.5% in the same/adjacent row are absorbed to R.", h, TITLE_DROP["BC"], chars=48)

    # ---- (D) co-variation join ----
    top, h = ROWS["DE"]
    comps = parts[(parts.animal == ROOT_ANIMAL) & (parts.am_harm_of < 0)].drop_duplicates("f_lo").sort_values("f_lo")
    link_of = {l[0]: (l[2], l[3]) for l in X["links"]}
    s_root, hp_root = series(X, comps.f_lo.iloc[0], comps.f_hi.iloc[0], float(comps.R.iloc[0]))
    sep = fund_part(SEP_ANIMAL)
    s_sep, hp_sep = series(X, sep.f_lo, sep.f_hi, float(sep.R)); r_sep, z_sep = W2.shift_z(hp_sep, hp_root)
    ax = fig.add_axes(rect(XL[0], XL[1], top, h))
    tt = (np.arange(len(s_root)) + 0.5) * W2.SLOW_S
    off = 0
    for i, c in enumerate(comps.itertuples()):
        s, hp = series(X, c.f_lo, c.f_hi, float(c.R))
        y = s / (np.percentile(s, 99) + 1e-12) + off
        ax.plot(tt[:len(y)], y, color=plt.get_cmap("tab10")(0), lw=0.7)
        lab = f"{c.f_lo:.0f}-{c.f_hi:.0f} Hz" + ("  (the fundamental, root of the test)" if i == 0 else (f"  joined: r {link_of[c.Index][0]:.2f}, z {link_of[c.Index][1]:.1f}" if c.Index in link_of else "  joined"))
        ax.text(2, off + 1.08, lab, fontsize=5, va="bottom", color=plt.get_cmap("tab10")(0))
        off += 1.45
    ax.plot(tt[:len(s_sep)], s_sep / (np.percentile(s_sep, 99) + 1e-12) + off, color="0.45", lw=0.7)
    ax.text(2, off + 1.08, f"{sep.f_lo:.0f}-{sep.f_hi:.0f} Hz @ {sep.R:.2f} Hz  not joined: r {r_sep:.2f}, z {z_sep:.1f}", fontsize=5, va="bottom", color="0.3")
    ax.set_xlim(0, 300); ax.set_ylim(-0.1, off + 1.5); ax.set_yticks([]); ax.set_xlabel("seconds into the window"); ax.set_ylabel("rate-locked amplitude")
    title(ax, "D", "Peaks with same rate at different carrier frequencies are joined if their rate locked on-off pattern over the 5 minute window match.", h, TITLE_DROP["DE"], chars=48)

    # ---- (E) postfilter flags on the plane ----
    Af = postfilter.flag_harmonics(A.copy(), parts).set_index("animal")
    pw = postfilter.animal_power(parts); pw_db = {a: 10 * np.log10(v) for a, v in pw.items()}
    fund = parts[parts.am_harm_of < 0].sort_values("f_c").groupby("animal").first()
    ax = fig.add_axes(rect(XR[0], XR[1], top, h))
    flagged_order = list(fund[Af.loc[fund.index, 'harm_of'].values >= 0].sort_values('f_c').index)
    for a, p in fund.iterrows():
        flagged = Af.loc[a, "harm_of"] >= 0 or Af.loc[a, "sub_of"] >= 0
        ax.scatter([p.R], [p.f_c], s=16, facecolors="none" if flagged else "0.25", edgecolors="0.25", lw=0.8, zorder=3)
        if Af.loc[a, "harm_of"] >= 0:
            root = int(Af.loc[a, "harm_of"]); q = fund.loc[root]
            ax.annotate("", xy=(q.R, q.f_c), xytext=(p.R, p.f_c), arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.7, shrinkA=3, shrinkB=3), zorder=2)
            k_lab = flagged_order.index(a); dx = (1.08, 1.45, 2.0)[k_lab % 3]; dy = (1.0, 1.06, 0.94)[k_lab % 3]
            ax.text(p.R * dx, p.f_c * dy, f"{Af.loc[a, 'harm_m'] or ''}x, {pw_db[a] - pw_db[root]:+.0f} dB", fontsize=5, color="tab:red", va="center")
    ax.set_xscale("log"); ax.set_xlim(2, 100); khz_log(ax, 1500, 22000, ticks=(1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000))
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30, 50, 100])); ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel("rate (Hz)"); ax.set_ylabel("carrier (kHz)"); ax.grid(alpha=0.25)
    title(ax, "E", "Harmonic post filter flags peaks that have a peak at least 6 dB stronger at 2F, 3F, F/2 or F/3.", h, TITLE_DROP["DE"], chars=48)

    # ---- (F) a chirper: the map finds the chirp rate, not the syllable rate ----
    import chirp_channel as CC
    top, h = ROWS["F"]
    ch = Aidx.loc[CHIRP_ANIMAL]
    sig, sr = X["sig"], X["sr"]
    dch = per_blob[int(pc.blob)]
    run = max(dch["cols"], key=lambda sl: sl.stop - sl.start); t_a, t_b = run.start * dt, min(run.stop * dt, run.start * dt + 60)
    res = CC.syllable_rate_from_chirps(sig, sr, float(ch.F_lo), float(ch.F_hi), t_a, t_b)
    ax = fig.add_axes(rect(X0, 3.65, top, h))
    ax.plot(fr, ex14, color=C_PURPLE, lw=0.9)
    ax.axvline(R14, color=C_PURPLE, lw=1.0)
    for m in range(2, int(fr[-1] / R14) + 1):
        ax.axvline(R14 * m, color=C_PURPLE, lw=0.5, ls=":", alpha=0.8)
    if np.isfinite(res["Rs"]):
        ax.axvline(res["Rs"], color="tab:orange", lw=1.0, ls="--")
    comb_axis(ax)
    ax.set_ylim(min(-1.0, float(ex14.min()) - 0.5), float(ex14.max()) * 1.15)
    title(ax, "F", "A chirper: modulation map identifies chirp rate (purple), not syllable repeat rate (orange).", h, drop=0.17)
    chirper_excerpts(fig, [rect(4.05, 5.55, top, h), rect(6.05, 7.55, top, h)], X, ch, C_PURPLE)

    # ---- (G) the time-domain reader on the same chirper ----
    top, h = ROWS["G"]
    tr = chirp_trace(sig, sr, float(ch.F_lo), float(ch.F_hi), t_a, t_b)
    ax = fig.add_axes(rect(X0, 3.65, top, h))
    c0 = tr["t"][tr["segs"][len(tr["segs"]) // 2][0]] if tr["segs"] else t_a + 1
    w0, w1 = c0 - 0.6, c0 + 0.9
    m = (tr["t"] >= w0) & (tr["t"] <= w1)
    ax.plot(tr["t"][m], tr["fast"][m], color=C_CHIRP, lw=0.6)
    ax.plot(tr["t"][m], tr["slow"][m], color="k", lw=0.6, alpha=0.7)
    ax.axhline(tr["thr"], color="k", ls=":", lw=0.5)
    for sg in tr["segs"]:
        ta_, tb_ = tr["t"][sg[0]], tr["t"][sg[-1]]
        if tb_ >= w0 and ta_ <= w1:
            ax.axvspan(ta_, tb_, color="0.9", zorder=0)
    pk_t = np.array(tr["peak_t"]); pk_v = np.array(tr["peak_v"]); mm = (pk_t >= w0) & (pk_t <= w1)
    ax.plot(pk_t[mm], pk_v[mm], ".", color="tab:red", ms=2.5)
    ax.set_xlim(w0, w1); ax.set_xlabel("seconds into the window"); ax.set_ylabel("band envelope"); ax.set_yticks([])
    ax.set_title("chirps (shaded) where envelope clears threshold", fontsize=BODY_PT, loc="left")
    title(ax, "G", "Direct detection of chirp syllable repeat rate", h, drop=0.17)
    ax = fig.add_axes(rect(4.05, 5.55, top, h))
    sg = tr["segs"][len(tr["segs"]) // 2]; ta_, tb_ = tr["t"][sg[0]] - 0.02, tr["t"][sg[-1]] + 0.02
    m = (tr["t"] >= ta_) & (tr["t"] <= tb_); ax.plot(tr["t"][m], tr["fast"][m], color=C_CHIRP, lw=0.8)
    mm = (pk_t >= ta_) & (pk_t <= tb_); ax.plot(pk_t[mm], pk_v[mm], "o", color="tab:red", ms=3)
    ax.set_xlim(ta_, tb_); ax.set_yticks([]); ax.set_xlabel("seconds into the window"); ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_title(f"one chirp: {int(mm.sum())} pulses in {1000 * (tb_ - ta_ - 0.04):.0f} ms", fontsize=BODY_PT, loc="left")
    ax = fig.add_axes(rect(6.05, 7.55, top, h))
    iv = np.array(tr["intervals"]) * 1000
    ax.hist(iv, bins=np.arange(5, 40, 1), color=C_CHIRP, alpha=0.8)
    ax.axvline(np.median(iv), color="k", lw=0.8)
    ax.set_xlabel("syllable interval (ms)"); ax.set_ylabel("count")
    ax.set_title(f"{len(iv)} intervals: median {np.median(iv):.1f} ms = {1000 / np.median(iv):.1f} Hz", fontsize=BODY_PT, loc="left")

    page.finish(fig, args.out)
    print(f"wrote {args.out}.pdf / .png  (chirper {CHIRP_ANIMAL}: chirp rate {R14:.2f} Hz, syllables {res['Rs']:.1f} Hz from {res['n_chirps']} chirps)")


if __name__ == "__main__":
    main()
