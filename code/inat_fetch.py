"""Fetch iNaturalist Orthoptera sound observations for Arkansas and its neighbors and download the recordings (MBE
2026-10-17: "quite a few orthopteran ones from Arkansas and surrounding states that might be interesting to put on our
plane" / "build it").

API: api.inaturalist.org/v1/observations with taxon_id 47651 (Orthoptera), sounds=true, quality_grade=research, per state
(place ids looked up by name: AR 36, LA 27, TX 18, OK 12, MO 28, MS 37, TN 45). Writes <root>/observations.csv (one row
per sound: observation id, taxon, time observed, coordinates, place, license, attribution, sound url and format) and
downloads every sound into <root>/raw/audio/<obs>_<sound>.<ext>, converting anything that is not WAV/FLAC to 48 kHz
mono WAV with ffmpeg into <root>/raw/wav/. Downloaded files are untrusted data: own directory, read with scripts only.

Usage:
  python analysis/callers/inat_fetch.py analysis/callers/runs/inat [--states AR LA TX OK MO MS TN] [--no-download]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import pandas as pd
import requests

API = "https://api.inaturalist.org/v1/observations"
PLACES = {"AR": 36, "LA": 27, "TX": 18, "OK": 12, "MO": 28, "MS": 37, "TN": 45}
TAXON = 47651
HEAD = {"User-Agent": "BROOKS orthopteran song study (eisenm@hhmi.org)"}


def fetch(states: list[str]) -> pd.DataFrame:
    rows = []
    for st in states:
        pid = PLACES[st]; page = 1
        while True:
            r = requests.get(API, params=dict(taxon_id=TAXON, sounds="true", place_id=pid, quality_grade="research", per_page=200, page=page, order_by="id", order="asc"), headers=HEAD, timeout=90)
            r.raise_for_status(); j = r.json(); res = j.get("results", [])
            for o in res:
                tax = o.get("taxon") or {}
                for s in o.get("sounds") or []:
                    rows.append(dict(obs_id=o["id"], sound_id=s.get("id"), state=st, taxon=tax.get("name"), rank=tax.get("rank"), taxon_id=tax.get("id"),
                                     observed=o.get("time_observed_at") or o.get("observed_on"), observed_on=o.get("observed_on"), lat=(o.get("location") or ",").split(",")[0], lon=(o.get("location") or ",").split(",")[-1],
                                     place=o.get("place_guess"), license=s.get("license_code"), attribution=s.get("attribution"), url=s.get("file_url"), fmt=s.get("file_content_type"), user=(o.get("user") or {}).get("login")))
            print(f"  {st}: page {page}, {len(res)} observations, {len(rows)} sounds so far", flush=True)
            if len(res) < 200:
                break
            page += 1; time.sleep(1.0)
        time.sleep(1.0)
    D = pd.DataFrame(rows).drop_duplicates("sound_id")
    return D


def download(D: pd.DataFrame, root: str) -> pd.DataFrame:
    adir, wdir = os.path.join(root, "raw", "audio"), os.path.join(root, "raw", "wav"); os.makedirs(adir, exist_ok=True); os.makedirs(wdir, exist_ok=True)
    out = []
    for i, r in enumerate(D.itertuples()):
        if not isinstance(r.url, str):
            out.append(""); continue
        ext = r.url.split("?")[0].rsplit(".", 1)[-1].lower()[:4]
        fn = os.path.join(adir, f"{r.obs_id}_{r.sound_id}.{ext}")
        if not os.path.exists(fn):
            try:
                with requests.get(r.url, headers=HEAD, timeout=120, stream=True) as q:
                    q.raise_for_status()
                    with open(fn, "wb") as fh:
                        for ch in q.iter_content(1 << 16):
                            fh.write(ch)
            except Exception as e:  # noqa: BLE001
                print(f"  {r.obs_id}: download failed {type(e).__name__}", flush=True); out.append(""); continue
            time.sleep(0.3)
        wav = os.path.join(wdir, f"{r.obs_id}_{r.sound_id}.wav")
        if ext in ("wav", "flac") and not os.path.exists(wav):
            os.symlink(os.path.abspath(fn), wav) if ext == "wav" else subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", fn, "-ac", "1", wav])
        elif not os.path.exists(wav):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", fn, "-ac", "1", "-ar", "48000", wav])
        out.append(wav if os.path.exists(wav) else "")
        if (i + 1) % 100 == 0:
            print(f"  downloaded {i + 1} of {len(D)}", flush=True)
    return D.assign(wav=out)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--states", nargs="+", default=list(PLACES)); ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args(); os.makedirs(args.root, exist_ok=True)
    p = os.path.join(args.root, "observations.csv")
    D = pd.read_csv(p) if os.path.exists(p) and not args.no_download and "wav" not in pd.read_csv(p, nrows=0).columns else None
    if D is None or args.no_download:
        D = fetch(args.states); D.to_csv(p, index=False)
        print(f"{len(D):,} sounds from {D.obs_id.nunique():,} observations, {D.taxon.nunique()} taxa; licenses {D.license.value_counts().to_dict()}", flush=True)
    if args.no_download:
        return
    D = download(D, args.root); D.to_csv(p, index=False)
    print(f"done: {int((D.wav != '').sum()):,} wav files in {args.root}/raw/wav", flush=True)


if __name__ == "__main__":
    main()
