"""Put SINA's reference recordings on our own map.

SINA species pages carry Walker's recordings as WAV files, each captioned with
the recording temperature. Running the window analysis on those clips
measures each species with exactly the instrument used on the field data --
carrier, rate, modulation depth, harmonic set, fold shape -- at a known
temperature, so the comparison with the month's streaks needs no conversion
between thermometers: a reference point either lies on a streak's line in
(log F, log R) or it does not.

Steps (each cached):
    --fetch    parse recording captions from the cached species pages, download the calling-song WAVs
    --analyse  run the analysis on every clip, write sina_reference_measurements.csv
    --match    place each measurement against the month's streaks

Usage:
    python analysis/callers/sina_reference.py analysis/callers/runs/month --fetch --analyse --match
"""
from __future__ import annotations

import argparse
import html
import os
import re
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

import window_blobs as W

BASE = "https://orthsoc.org/sina/"
UA = {"User-Agent": "Mozilla/5.0 (research; caller-inventory matching)"}
# small crickets whose range data for Arkansas are too thin to filter on
SMALL_CRICKETS = {"trigonidiinae", "nemobiinae", "mogoplistinae"}


def plausible(df: pd.DataFrame) -> pd.Series:
    return (df.n_near > 0) | df.supplement.astype(str).str.lower().eq("true") | df.subfamily.astype(str).str.lower().isin(SMALL_CRICKETS)


def strip(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def parse_recordings(sdir: str) -> pd.DataFrame:
    names = pd.read_csv(os.path.join(sdir, "sina_songs.csv")).set_index("page")
    geo = pd.read_csv(os.path.join(sdir, "species_geography.csv")).set_index("page")
    sup = pd.read_csv(os.path.join(sdir, "supplement_species.csv")) if os.path.exists(os.path.join(sdir, "supplement_species.csv")) else pd.DataFrame(columns=["page"])
    rows = []
    for page in names.index:
        fn = os.path.join(sdir, page)
        if not os.path.exists(fn):
            continue
        h = open(fn, encoding="latin-1").read()
        for m in re.finditer(r'<div class="recording">(.*?)</div>', h, flags=re.S | re.I):
            block = m.group(1)
            wav = re.search(r'href="(media/[^"]+\.wav)"', block, flags=re.I)
            if not wav:
                continue
            cap = strip(block.split("<audio")[0])
            t = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|Â°|&deg;|deg)\s*C", block)
            kind = ("calling" if re.search(r"calling", cap, re.I) else
                    "courtship" if re.search(r"court", cap, re.I) else
                    "aggressive" if re.search(r"aggress|fight", cap, re.I) else "other")
            rows.append(dict(page=page, name=names.name.get(page, ""), subfamily=names.subfamily.get(page, ""),
                             wav=wav.group(1), caption=cap[:200], T=float(t.group(1)) if t else np.nan, kind=kind,
                             n_near=int(geo.n_near.get(page, 0)) if page in geo.index else 0,
                             supplement=page in set(sup.page)))
    return pd.DataFrame(rows)


def fetch(root: str, only_plausible: bool = True) -> pd.DataFrame:
    sdir = os.path.join(root, "sina")
    R = parse_recordings(sdir)
    R.to_csv(os.path.join(sdir, "sina_recordings.csv"), index=False)
    sel = R[(R.kind == "calling") & R["T"].notna()]
    if only_plausible:
        sel = sel[plausible(sel)]
    print(f"{len(R)} recordings on {R.page.nunique()} pages; {len(sel)} calling songs with a temperature "
          f"from {sel.page.nunique()} plausible species")
    adir = os.path.join(sdir, "audio")
    os.makedirs(adir, exist_ok=True)
    for r in sel.itertuples():
        fn = os.path.join(adir, os.path.basename(r.wav))
        if os.path.exists(fn):
            continue
        try:
            req = urllib.request.Request(BASE + r.wav, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as u:
                open(fn, "wb").write(u.read())
            time.sleep(0.2)
        except Exception as e:
            print(f"  {r.wav}: {e}")
    print(f"{len(os.listdir(adir))} audio files present")
    return sel


def analyse(root: str) -> pd.DataFrame:
    sdir = os.path.join(root, "sina")
    R = pd.read_csv(os.path.join(sdir, "sina_recordings.csv"))
    R = R[(R.kind == "calling") & R["T"].notna()]
    R = R[plausible(R)]
    adir = os.path.join(sdir, "audio")
    songs = pd.read_csv(os.path.join(sdir, "sina_songs.csv")).set_index("page")
    geo = pd.read_csv(os.path.join(sdir, "species_geography.csv")).set_index("page")
    rows, folds = [], {}
    for r in R.itertuples():
        fn = os.path.join(adir, os.path.basename(r.wav))
        if not os.path.exists(fn):
            continue
        try:
            sig, sr = sf.read(fn, dtype="float32")
        except Exception as e:
            print(f"  {r.wav}: unreadable ({e})")
            continue
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        dur = len(sig) / sr
        if dur < 3:
            continue
        try:
            X = W.analyse_signal(sig, sr, dur, fmax=min(22050.0, sr / 2), fold_s=min(30.0, dur))
        except Exception as e:
            print(f"  {r.wav}: analysis failed ({e})")
            continue
        parts, A = X["parts"], X["A"]
        khz = songs.kHz_first.get(r.page, np.nan)
        base = dict(page=r.page, name=r.name, subfamily=r.subfamily, wav=os.path.basename(r.wav), T=r.T,
                    dur_s=round(dur, 1), sr=sr, n_blobs=len(X["B"]), n_parts=len(parts), n_animals=len(A),
                    sina_kHz=khz, n_AR=int(geo.n_AR.get(r.page, 0)) if r.page in geo.index else 0)
        if A.empty:
            rows.append(base)
            continue
        # the recording is of one male, but tape hum and rumble also make
        # "animals": take the fundamental component nearest the page's stated
        # dominant frequency, else the deepest one above 1.5 kHz
        cand = parts[(parts.am_harm_of < 0) & (parts.f_c >= 1500)].copy()
        if cand.empty:
            rows.append(base)
            continue
        # among components at the page frequency, the modulation fundamental is
        # the LOWEST rate that is still strongly modulated (a 2x harmonic can be
        # deeper than the fundamental, e.g. Conocephalus strictus 64 Hz vs 32)
        pool = cand
        if pd.notna(khz):
            cand["dk"] = np.abs(np.log(cand.f_c / (1000 * khz)))
            near = cand[cand.dk < 0.35]
            if len(near):
                pool = near
        # the deepest non-harmonic component. Preferring the lowest rate in the
        # band was tried and rejected: strong trills often carry a half-rate
        # component, and O. texensis, O. latipennis and G. rubens then measured
        # at R/2 and left their streaks. The one known 2x case (C. strictus,
        # 64 Hz for a 32 Hz song, its fundamental too weak to be a part) is
        # documented rather than special-cased.
        fund = pool.sort_values("depth", ascending=False).iloc[0]
        a = A[A.animal == fund.animal].iloc[0]
        comp = parts[(parts.animal == a.animal) & (parts.am_harm_of < 0)].sort_values("f_lo")
        rows.append(dict(**base, F_c=fund.f_c, f_lo=fund.f_lo, f_hi=fund.f_hi, R=fund.R, depth=fund.depth,
                         on_frac=fund.on_frac, n_components=len(comp), n_am_harm=int((parts.animal == a.animal).sum() - len(comp)),
                         matches_page_kHz=bool(pd.isna(khz) or abs(np.log(fund.f_c / (1000 * khz))) < 0.35),
                         components=";".join(f"{c.f_lo:.0f}-{c.f_hi:.0f}" for c in comp.itertuples())))
        d = X["per_blob"][fund.blob]
        folds[os.path.basename(r.wav)] = (X["f"][d["p_lo"]:d["p_hi"] + 1], d["F"], fund.R, fund.f_lo, fund.f_hi)
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(sdir, "sina_reference_measurements.csv"), index=False, float_format="%.3f")
    np.save(os.path.join(sdir, "sina_reference_folds.npy"), folds, allow_pickle=True)
    ok = M.dropna(subset=["R"]) if "R" in M else M.iloc[0:0]
    print(f"{len(M)} clips analysed, {len(ok)} with a measured caller, {ok.page.nunique()} species")
    return M


def match(root: str) -> None:
    sdir = os.path.join(root, "sina")
    M = pd.read_csv(os.path.join(sdir, "sina_reference_measurements.csv")).dropna(subset=["R"])
    M["matches_page_kHz"] = M.matches_page_kHz.astype(str).str.lower().eq("true")
    site_fn = os.path.join(sdir, "site_species.csv")
    site = set(pd.read_csv(site_fn).species.str.strip()) if os.path.exists(site_fn) else set()
    M["site_obs"] = M.name.str.split("|").str[0].str.strip().isin(site)
    A = pd.read_csv(os.path.join(root, "animal_nights_streaks.csv"))
    S = pd.read_csv(os.path.join(root, "streaks_coarse.csv"), index_col=0)
    S = S[S.n >= 15]
    # each streak as a line in (log R, log F) with its perpendicular scatter
    lines = {}
    for sid in S.index:
        g = A[A.streak == sid]
        x, y = np.log(g.R.to_numpy()), np.log(g.F.to_numpy())
        if len(g) >= 8 and x.std() > 0.02:
            b, a = np.polyfit(x, y, 1)
        else:
            b, a = 0.3, np.log(g.F.median()) - 0.3 * np.log(g.R.median())
        resid = y - (a + b * x)
        lines[sid] = dict(a=a, b=b, sd=max(resid.std(), 0.03), x_lo=x.min() - 0.15, x_hi=x.max() + 0.15,
                          n=len(g), F25=S.loc[sid, "F25"], R25=S.loc[sid, "R25"])
    rows = []
    for m in M.itertuples():
        x, y = np.log(m.R), np.log(m.F_c)
        best = []
        for sid, L in lines.items():
            if not (L["x_lo"] <= x <= L["x_hi"]):
                continue
            z = (y - (L["a"] + L["b"] * x)) / L["sd"]
            best.append((abs(z), sid, z))
        best.sort()
        common = dict(name=m.name.split("|")[0].strip(), common=m.name.split("|")[-1].strip(), wav=m.wav, T=m.T,
                      n_AR=m.n_AR, site_obs=m.site_obs, on_page_kHz=m.matches_page_kHz, F_meas=m.F_c, R_meas=m.R,
                      depth=m.depth, n_am_harm=m.n_am_harm)
        for k, (az, sid, z) in enumerate(best[:2]):
            L = lines[sid]
            rows.append(dict(**common, rank=k + 1, streak=sid, streak_n=L["n"], streak_F25=L["F25"], streak_R25=L["R25"], z=z))
        if not best:
            rows.append(dict(**common, rank=0, streak=-1))
    X = pd.DataFrame(rows)
    X.to_csv(os.path.join(root, "sina_reference_matches.csv"), index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 34)
    good = X[X.on_page_kHz]
    hit = good[(good["rank"] == 1) & (good.z.abs() < 2.5)]
    print(f"{M.page.nunique()} species measured; {M.matches_page_kHz.sum()} recordings measured at the page's dominant frequency; "
          f"{hit.wav.nunique()} of those land within 2.5 sd of a streak line")
    cols = ["streak", "streak_n", "streak_F25", "streak_R25", "name", "common", "n_AR", "site_obs", "T", "F_meas", "R_meas", "depth", "n_am_harm", "z"]
    fmt = lambda v: f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
    print("\n== species seen at the site ==")
    print(hit[hit.site_obs].sort_values(["streak", "name"])[cols].to_string(index=False, float_format=fmt))
    print("\n== species recorded in Arkansas ==")
    print(hit[(hit.n_AR > 0) & ~hit.site_obs].sort_values(["streak", "name"])[cols].to_string(index=False, float_format=fmt))
    print("\n== other candidates (neighbouring states, or small crickets with no useful range data) ==")
    print(hit[(hit.n_AR == 0) & ~hit.site_obs].sort_values(["streak", "name"])[cols].to_string(index=False, float_format=fmt))
    print("\n== measured at the page frequency but on no streak (rank 0 or > 2.5 sd) ==")
    miss = good[(good["rank"] == 0) | ((good["rank"] == 1) & (good.z.abs() >= 2.5))]
    print(miss.sort_values("F_meas")[["name", "n_AR", "T", "F_meas", "R_meas", "depth", "streak", "z"]].to_string(index=False, float_format=lambda v: f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"))
    print(f"\n{(~M.matches_page_kHz).sum()} recordings measured away from the page frequency (artefact or rate outside 8-100 Hz):")
    print(M[~M.matches_page_kHz][["name", "T", "sina_kHz", "F_c", "R", "depth"]].sort_values("name").to_string(index=False, float_format=lambda v: f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"))

    # figure: the month's animal-nights as a cloud, reference recordings as labelled points
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.scatter(A.R, A.F, s=3, color="0.75", alpha=0.5, label="month animal-nights")
    fam_col = {"oecanthinae": "tab:red", "gryllinae": "tab:orange", "gryllidae": "tab:orange", "nemobiinae": "tab:green",
               "trigonidiinae": "tab:purple", "eneopterinae": "tab:brown", "gryllotalpidae": "tab:pink",
               "tettigoniidae": "tab:blue", "tettigoniinae": "tab:cyan", "copiphorini": "navy", "phaneropterinae": "teal"}
    for m in M.itertuples():
        c = fam_col.get(str(m.subfamily).lower(), "k")
        ax.scatter(m.R, m.F_c, s=30 + 60 * m.depth, color=c, edgecolors="k", lw=0.5, zorder=3)
        ax.annotate(f"{m.name.split('|')[0].strip().split()[0][0]}. {m.name.split('|')[0].strip().split()[-1]} {m.T:.0f}C",
                    (m.R, m.F_c), fontsize=6, xytext=(3, 2), textcoords="offset points", color=c)
    for fam, c in fam_col.items():
        if (M.subfamily.str.lower() == fam).any():
            ax.scatter([], [], color=c, label=fam)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rate (Hz)")
    ax.set_ylabel("carrier (Hz)")
    ax.set_title("SINA reference recordings measured with the same pipeline (labelled: species, recording temperature) "
                 "over the month's animal-nights")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(root, "sina_reference_on_map.png"), dpi=110, bbox_inches="tight")
    print(f"wrote {os.path.join(root, 'sina_reference_on_map.png')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--match", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.root)
    if args.analyse:
        analyse(args.root)
    if args.match:
        match(args.root)


if __name__ == "__main__":
    main()
