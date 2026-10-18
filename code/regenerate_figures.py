"""Regenerate the paper figures from a run tree (2026-10-16: the switch to v4 with the carried v3 types; MBE: "switch
everything to v4 with the carried types"). The commands are the manifest's with the run paths swapped; they run in
sequence and log to <figures>/regenerate_<tree>.log. The type table's by-eye shape columns (shape_rule, shape,
shape_source, shape_note) are saved before the type-reference build and merged back after it.

Groups: catalog (Figures 7, 7s, 8, 9, 10, 11, 12, 13, 14, 15), types (type reference build, table, Figure 16),
nights (Figures 4b, 5, 6 from the focal night's unit), grids (the supplement's letter pages with audio, trill and chirp).

Usage:
  python analysis/callers/regenerate_figures.py --tree v4 [--groups catalog types nights grids] [--jobs 8]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

PY = sys.executable
C = "analysis/callers"; FIG = f"{C}/runs/figures_paper"
def trill_ids(Y: str) -> str:
    """streak ids of the trill types with >= 40 singer-nights, in paper-number order (the catalog changes: 2026-10-16 splits)."""
    P = pd.read_csv(f"{Y}/paper_ids_v9.csv"); P = P[(P.n >= 40) & (P.form != "chirp")].copy(); P["k"] = P.paper_id.astype(int)
    return " ".join(str(x) for x in P.sort_values("k").streak)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="v4"); ap.add_argument("--groups", nargs="+", default=["catalog", "types", "nights", "grids", "singers"]); ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    R = f"{C}/runs/{args.tree}"; Y = f"{R}/year_nov24_nov25"; U = f"{R}/y2024 {R}/y2025"; WX = f"{C}/runs/v2/y2025/weather_hourly_hrrr.csv"
    log = open(f"{FIG}/regenerate_{args.tree}.log", "a")

    def run(cmd, must: bool = False) -> int:
        argv = cmd if isinstance(cmd, list) else cmd.split()          # a list keeps arguments with spaces whole
        t0 = time.time(); log.write(f"{time.strftime('%H:%M:%S')} === {' '.join(argv)}\n"); log.flush()
        p = subprocess.run([PY, "-W", "ignore"] + argv, capture_output=True, text=True)
        out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not l.startswith("Note:"))
        log.write(out[-6000:] + f"\n--- rc={p.returncode} in {time.time() - t0:.0f} s\n"); log.flush()
        if p.returncode != 0 and must:
            log.write("STOP\n"); sys.exit(1)
        return p.returncode

    if "catalog" in args.groups:
        run(f"{C}/year_plane_figure.py {Y} --suffix _v9 --trillers-only --out {FIG}/F07_year_plane")
        run(f"{C}/type_panels_figure.py {Y} --suffix _v9 --number 7-S1 --out {FIG}/F07s_type_panels")
        run(f"{C}/coupling_traces_figure.py {Y} --suffix _v9 --roots {U} --examples 4301 3601 5501 5201 4601 4801 --number 8 --out {FIG}/F08_coupling")   # scatter + night traces per singer (MBE 2026-10-17)
        run(f"{C}/chirp_year.py {Y} --roots {U} --skip-build --min-syl-share 0.5 --exclude 13 --out {FIG}/F09_chirp_plane")   # v3 QC exclusion (METHOD 19 AP), same ids in the carried catalog
        run(f"{C}/type_mix_figure.py {Y} --suffix _v9 --roots {U} --top 16 --chirp-root {Y}/chirp --sites House Back9 NorthSide --out {FIG}/F11_type_mix")
        run(f"{C}/day_plane_figure.py {Y} --suffix _v9 --roots {U} --number 15 --out {FIG}/F15_day")
        run(f"{C}/diel_pattern_figure.py {Y} --suffix _v9 --roots {U} --number 16 --out {FIG}/F16_hour_of_day")
        run(f"{C}/temperature_panels_figure.py {Y} --suffix _v9 --var R --within --number 18 --out {FIG}/F18_rate_vs_temperature")      # within-singer versions are the production figures (MBE 2026-10-17)
        run(f"{C}/temperature_panels_figure.py {Y} --suffix _v9 --var F --within --number 19 --out {FIG}/F19_carrier_vs_temperature")
        run(f"{C}/streak_id_figure.py {Y} --suffix _v9 --number 13 --out {FIG}/F13_streak_id")
        run(f"{C}/reference_vs_streak_grid.py {Y} --suffix _v9 --unit-roots {U} --z {FIG}/F13_streak_id_reference_z.csv --streaks {trill_ids(Y)} --any --ignore-form --per-page 6 --paper-number 14 --out {FIG}/F14_names_evidence")
        run(f"{C}/sina_chirp_rates.py {Y}")
        run(f"{C}/chirp_reference_z.py {Y} --out {FIG}/explore/F12_chirp")
        run(f"{C}/reference_vs_streak_grid.py {Y} --suffix _v9 --unit-roots {U} --z {FIG}/explore/F12_chirp_reference_z.csv --chirp-plane --any --ignore-form --dur 1.0 --per-page 6 --paper-number 14 --page-start 6 --out {FIG}/F14_names_evidence_chirp")
        run(f"{C}/grid_sample_figure.py {Y} --suffix _v9 --unit-roots {U} --streaks 3601 4601 3151 --out {FIG}/F10_grid_sample")
        ov = f"{FIG}/exploratory/anaxipha_published_points.csv"
        if os.path.exists(ov):
            run([f"{C}/year_plane_figure.py", Y, "--suffix", "_v9", "--trillers-only", "--rlo", "2", "--a-only", "--overlay", ov, "--number", "12", "--title", "Published Anaxipha recordings on the year's plane.",
                 "--legend", "The year's trill singer-hours colored by air temperature, as in Figure 7A, with 1,318 recordings of 13 Anaxipha species from Walker and Funk (2014; their supplementary song table) as rimmed circles colored by the recording's own temperature on the same scale, each species with its fitted line and name; gray numbers mark the warm end of each type's line.",
                 "--out", f"{FIG}/F12_anaxipha_overlay"])
    if "types" in args.groups:
        tp = f"{FIG}/F15_type_reference_table.csv"
        shapes = pd.read_csv(tp, dtype={"type": str})[["type", "shape_rule", "shape", "shape_source", "shape_note"]] if os.path.exists(tp) else None
        run(f"{C}/type_reference_figure.py {Y} --suffix _v9 --unit-roots {U} --jobs {args.jobs} --out {FIG}/F15_type_reference")
        if shapes is not None and os.path.exists(tp):
            T = pd.read_csv(tp, dtype={"type": str}).drop(columns=[c for c in ("shape_rule", "shape", "shape_source", "shape_note") if c in pd.read_csv(tp, nrows=0).columns])
            T = T.merge(shapes, on="type", how="left"); T.to_csv(tp, index=False)
            md = tp.replace(".csv", ".md")
            if os.path.exists(md):                       # the trill table's shape cell (5th column) from the saved mapping
                lines = open(md).read().split("\n"); out = []; sm = dict(zip(shapes.type, shapes["shape"]))
                for ln in lines:
                    cells = ln.split(" | ")
                    if ln.startswith("| ") and len(cells) > 5 and cells[0].strip("| ").strip() in sm and cells[4].strip() in ("flat", "downslur", "upslur", "hockey stick", "chevron", "broadband", "indeterminate", "downslur / hockey stick", ""):
                        cells[4] = sm[cells[0].strip("| ").strip()]; ln = " | ".join(cells)
                    out.append(ln)
                open(md, "w").write("\n".join(out))
            log.write("shape columns restored\n"); log.flush()
        run(f"{C}/type_exemplar_figure.py {Y} --suffix _v9 --unit-roots {U} --number 17 --out {FIG}/F17_type_exemplars")
    if "nights" in args.groups:
        unit = f"{R}/y2025/BARLT_00008662_House_20250920_night"
                run([f"{C}/night_streaks_figure.py", "--unit-dir", unit, "--window", "2025-09-20 23:05", "--weather", WX, "--out", f"{FIG}/F05_one_night_streaks"])
        run(f"{C}/night_linking_figure.py --unit-dir {unit} --weather {WX} --track 51 33 --out {FIG}/F06_night_linking")
    if "grids" in args.groups:
        run(f"{C}/streak_grid.py {Y} --suffix _v9 --unit-roots {U} --inset --letter --audio --out-dir {FIG}/S_grids_{args.tree} --jobs 6")
        run(f"{C}/streak_grid.py {Y} --suffix _v9 --unit-roots {U} --chirp-plane --dur 1.0 --inset --letter --audio --out-dir {FIG}/S_grids_{args.tree}_chirp --jobs 6")
    if "singers" in args.groups:                     # MBE 2026-10-16: "you also didn't remake the singer nights, which was the whole point of this exercise"
        P = pd.read_csv(f"{Y}/paper_ids_v9.csv"); types = [str(t) for t in sorted(P[(P.n >= 40) & (P.form != "chirp")].paper_id.astype(int))]
        run(f"{C}/singer_night_grid.py {Y} --suffix _v9 --unit-roots {U} --types {' '.join(types)} --out-dir {FIG}/exploratory/singer_nights_{args.tree}")
        run(f"{C}/singer_paths_grid.py {Y} --suffix _v9 --roots {U} --types {' '.join(types)} --out-dir {FIG}/exploratory/singer_paths_{args.tree}")
    log.write(f"{time.strftime('%H:%M:%S')} REGENERATE DONE\n"); log.close()


if __name__ == "__main__":
    main()
