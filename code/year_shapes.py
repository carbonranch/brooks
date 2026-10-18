"""Shape descriptors per singer-period from the quiet-level fold, for the within-streak
shape test (does one line hold more than one shape at one position?).

For every singer-period of the streaks with >= MIN_N members (capped at CAP per streak,
deepest first), the same measurement as the gallery: the track's deepest window with a
syllable rate, the loudest 30 s for its band, the band +/- 600 Hz folded at the track's
rate (48 phase bins), dB above each row's quiet level. From the fold:
  duty      fraction of the period the modulated-power envelope is above half its max
  glide_hz  excursion of the power ridge across the unit (signed, Hz; fold_shape's rule)
  bw_hz     rows whose modulated power exceeds half the maximum, in Hz
  snr_db    peak of the fold in dB above the quiet level
  on_rows   rows above half max (count)
Writes year_shapes<suffix>.csv (one row per singer-period, resumable) and
year_shapes<suffix>.png: per streak, the distribution of glide and duty, with a
two-component split flagged when the glide distribution is bimodal (dBIC >= 10, both
components >= 20 %).

Usage:
    python analysis/callers/year_shapes.py analysis/callers/runs/v2/year_nov24_nov25 --suffix _v6 \
        --unit-roots analysis/callers/runs/v2/y2024 analysis/callers/runs/v2/y2025 --jobs 12
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import streak_gallery as G  # noqa: E402

MIN_N, CAP = 100, 400
DF = 44100 / 512


def descriptors(x: dict) -> dict:
    Fdb, fk, env = x["Fdb"], x["fk"], x["env"]
    nph = Fdb.shape[1]
    on = env >= 0.5
    duty = float(on.mean())
    # modulated power per row (linear), ridge = power-weighted frequency per phase bin
    Flin = 10 ** (Fdb / 10) - 1.0
    Flin = np.clip(Flin, 0, None)
    row_pw = Flin.mean(axis=1)
    on_rows = row_pw >= 0.5 * row_pw.max() if row_pw.max() > 0 else np.zeros(len(fk), bool)
    bw = float(on_rows.sum() * DF)
    ridge = np.array([np.average(fk, weights=Flin[:, k] + 1e-12) for k in range(nph)])
    # contiguous unit: start at the longest run of 'on' (circular)
    k0 = int(np.argmax(np.convolve(np.concatenate([on, on]).astype(float), np.ones(max(int(on.sum()), 1)), "valid")[:nph]))
    idx = [(k0 + i) % nph for i in range(max(int(on.sum()), 2))]
    r, w = ridge[idx], env[idx]
    if len(r) > 2:
        slope = np.polyfit(np.arange(len(r)), r, 1, w=np.sqrt(w + 1e-12))[0]
        glide = float(slope * (len(r) - 1))
    else:
        glide = float(r[-1] - r[0])
    return dict(duty=duty, glide_hz=glide, bw_hz=bw, snr_db=float(Fdb.max()), on_rows=int(on_rows.sum()), R=x["R"])


def _one(args):
    an, unit_roots = args
    w = G.window_of(an, unit_roots)
    if w is None:
        return dict(an=an, status="no window")
    try:
        x = G.analyse(w)
        d = descriptors(x)
        d.update(an=an, status="ok", F_fix=w["F_fix"], t=str(w["t"]))
        return d
    except SystemExit as e:
        return dict(an=an, status=f"no audio: {str(e)[:40]}")
    except Exception as e:  # noqa: BLE001
        return dict(an=an, status=f"{type(e).__name__}: {str(e)[:40]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--suffix", default="_v6"); ap.add_argument("--unit-roots", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--cap", type=int, default=CAP); ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    root = args.root
    A = pd.read_csv(os.path.join(root, f"animal_nights_streaks{args.suffix}.csv"))
    S = pd.read_csv(os.path.join(root, f"streaks_coarse{args.suffix}.csv"), index_col=0)
    S = S[(S.harmonic_of < 0) & (S.n >= args.min_n)] if "harmonic_of" in S.columns else S[S.n >= args.min_n]
    out = os.path.join(root, f"year_shapes{args.suffix}.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["an"])
    todo = []
    for sid in S.index:
        g = A[A.streak == sid].sort_values("depth", ascending=False).head(args.cap)
        todo += [an for an in g.an if an not in set(done.an)]
    print(f"{len(S)} streaks, {len(todo)} singer-periods to measure ({len(done)} done)", flush=True)
    rows = done.to_dict("records")
    if todo:
        with Pool(args.jobs) as pool:
            for k, d in enumerate(pool.imap_unordered(_one, [(an, args.unit_roots) for an in todo], chunksize=4), 1):
                rows.append(d)
                if k % 200 == 0:
                    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.3f"); print(f"  {k}/{len(todo)}", flush=True)
    D = pd.DataFrame(rows); D.to_csv(out, index=False, float_format="%.3f")
    D = D[D.status == "ok"].merge(A[["an", "streak", "T"]], on="an")
    # per-streak distributions and a two-component test on glide
    from sklearn.mixture import GaussianMixture
    rows_s = []
    sids = [s for s in S.sort_values("n", ascending=False).index if (D.streak == s).sum() >= 30]
    ncol = 4; nrow = int(np.ceil(len(sids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, sid in zip(axes.ravel(), sids):
        g = D[(D.streak == sid) & (D.snr_db >= 6)]
        X = g.glide_hz.to_numpy().reshape(-1, 1)
        g1 = GaussianMixture(1, random_state=0).fit(X); g2 = GaussianMixture(2, random_state=0, n_init=3).fit(X)
        dbic = g1.bic(X) - g2.bic(X)
        wts = np.sort(g2.weights_); mu = np.sort(g2.means_.ravel()); sd = np.sqrt(g2.covariances_.ravel()).mean()
        split = bool(dbic >= 10 and wts[0] >= 0.2 and (mu[1] - mu[0]) >= 2 * sd)
        rows_s.append(dict(streak=sid, n=len(g), glide_med=g.glide_hz.median(), glide_iqr=g.glide_hz.quantile(.75) - g.glide_hz.quantile(.25),
                           duty_med=g.duty.median(), bw_med=g.bw_hz.median(), dbic=dbic, mu_lo=mu[0], mu_hi=mu[1], w_lo=wts[0], split=split))
        ax.scatter(g.glide_hz, g.duty, s=6, alpha=0.4, c=g["T"], cmap="jet", vmin=10, vmax=30, edgecolors="none")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{sid}  {S.loc[sid, 'F25'] / 1000:.2f} kHz @ {S.loc[sid, 'R25']:.0f} Hz  n {len(g)}" + ("  SPLIT" if split else ""), fontsize=8, loc="left",
                     color="tab:red" if split else "k")
        ax.set_xlim(-900, 900); ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(sids):]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        ax.set_xlabel("glide across the unit (Hz, signed)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("duty", fontsize=8)
    fig.suptitle("Shape within streaks: glide and duty per singer-period (deepest window, fold at the track's rate), colored by temperature; "
                 "SPLIT = glide is two-component (dBIC >= 10, minor >= 20 %, 2 sd apart)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(root, f"year_shapes{args.suffix}.png"), dpi=130); plt.close(fig)
    R = pd.DataFrame(rows_s)
    R.to_csv(os.path.join(root, f"year_shapes_streaks{args.suffix}.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
