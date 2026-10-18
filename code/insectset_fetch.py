"""Fetch the InsectSet459 clips of North American Orthoptera (species on SINA) straight out of the Zenodo zips with range
requests (remote_zip.py), so that the 84 GB archive is not downloaded (MBE, 2026-10-14: "we need to analyze these data
and see if they are useful"; Faiß, Ghani & Stowell 2026, Sci. Data 13:499, Zenodo 10.5281/zenodo.18554693, CC BY 4.0).

Writes <root>/raw/audio/<file_name> and <root>/selection.csv (the annotation rows taken, with SINA page, n_AR, n_near).

Usage:
  python analysis/callers/insectset_fetch.py analysis/callers/runs/insectset459 [--all-orthoptera] [--jobs 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import RemoteZip  # noqa: E402

BASE = "https://zenodo.org/records/18554693/files/"
SINA_GEO = "analysis/callers/runs/v3/year_nov24_nov25/sina/species_geography.csv"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--all-orthoptera", action="store_true"); ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    raw = os.path.join(args.root, "raw"); adir = os.path.join(raw, "audio"); os.makedirs(adir, exist_ok=True)
    A = pd.read_csv(os.path.join(raw, "InsectSet459_Train_Val_Test_Annotation.csv"))
    O = A[A.group == "Orthoptera"].copy(); O["species"] = O.species_name.str.replace("_", " ")
    G = pd.read_csv(SINA_GEO); G["species"] = G.Genus + " " + G.SpecificName; G = G.set_index("species")
    O["page"] = O.species.map(G.page); O["n_AR"] = O.species.map(G.n_AR); O["n_near"] = O.species.map(G.n_near)
    sel = O if args.all_orthoptera else O[O.page.notna()]
    sel.to_csv(os.path.join(args.root, "selection.csv"), index=False)
    print(f"{len(sel)} files of {sel.species.nunique()} species to fetch ({sel.temperature.notna().sum()} with a temperature)", flush=True)
    todo = sel[~sel.file_name.map(lambda f: os.path.exists(os.path.join(adir, f)))]
    print(f"{len(todo)} not yet local", flush=True)
    for subset in ("Test", "Validation", "Train"):
        names = list(todo[todo.subset == subset].file_name)
        if not names:
            continue
        t0 = time.time(); z = RemoteZip(f"{BASE}{subset}.zip?download=1")
        members = {i.filename.split("/")[-1]: i for i in z.infolist()}
        got = 0; nbytes = 0

        def fetch(n):
            info = members.get(n)
            if info is None:
                return n, 0
            data = z.read(info)                           # the shared RangeFile is not thread safe: one reader at a time
            with open(os.path.join(adir, n), "wb") as fh:
                fh.write(data)
            return n, len(data)
        # sequential reads keep the range cache coherent; the zips are listed in member order so reads are mostly forward
        order = sorted(names, key=lambda n: members[n].header_offset if n in members else 0)
        for i, n in enumerate(order):
            _, b = fetch(n); got += b > 0; nbytes += b
            if (i + 1) % 100 == 0:
                print(f"  {subset}: {i + 1}/{len(order)} files, {nbytes / 1e6:.0f} MB, {time.time() - t0:.0f} s", flush=True)
        print(f"{subset}: {got} of {len(names)} files, {nbytes / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
