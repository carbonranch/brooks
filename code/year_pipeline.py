"""The year chain, end to end, as one command: from the v2 detections (runs/v2/y2024,
y2025; detection-only output of run_corpus2.py) to every table and figure the paper
uses. Each stage is skipped when its outputs exist unless --from <stage> or --redo is
given. Logs to <v2>/year_pipeline.log.

Stages (in order):
  lines      fixed_rate_lines.py on both roots (machine lines; postfilter reads them)
  link       night_global.py --all on both roots, nights and days (postfilter inside)
  hours      year_hours_figure.py, nights and days (singer-hours with HRRR T, Rc, rc_frac)
  streaks    year_streak_inputs -> build_streaks (_k55b) -> split by T (_v4) -> merge (_v6)
             -> track_form (chirp-channel form, bandwidth, rate steadiness per singer-period)
             -> track_form_audio (song form measured from the audio: bout-rhythm contrast)
             -> streaks_by_form --audio-form (_v9: the chain re-run for the firm chirpers;
             firm trillers keep the reviewed geometry; mixed assigned to the nearest streak;
             unsteady rates and chirp-rate harmonics dropped)
             -> streak_overlay_figure -> assign_day_to_streaks; v8 aliased to the
             unsuffixed names the month-scale scripts read
  names      match_sina (position), sina_reference --match (recordings on the line)
  types      type_mix, year_types (stability, thermal limits, phenogram, day/night,
             site occupancy), year_coupling, year_chirpers, year_harmonic_ref
  pulses     pulse_shapes over every member of every streak >= 40 (pulse SNR, shape)
  coherence  streak_coherence (all members) and --clear-only (pulse SNR >= 6 dB)
  catalog    type_catalog (the per-type table)
  galleries  streak_gallery --mode fixed; streak_grid raw and --sync 2

Usage:
    python analysis/callers/year_pipeline.py [--from STAGE] [--only STAGE ...] [--redo] [--jobs 12]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
C = os.path.relpath(HERE, REPO)
R = f"{C}/runs"
# the run tree: runs/v2 (the corpus run) by default; BROOKS_RUN_TREE=v3 for the chirp-repass relink tree
# (make_v3_tree.py: v2 detections by symlink, night links redone with BROOKS_SYLV2_LINK=1; METHOD 19 AH)
V2 = f"{R}/{os.environ.get('BROOKS_RUN_TREE', 'v2')}"
OUT = f"{V2}/year_nov24_nov25"; FIGS = f"{V2}/year_figs"
ROOTS = [f"{V2}/y2024", f"{V2}/y2025"]
WX_ROOTS = [f"{R}/y2024", f"{R}/y2025"]        # HRRR hourly tables live in the old run roots
FIRST, LAST = "2024-11-01", "2025-11-30"
SFX = "_v9"
STAGES = ["lines", "link", "hours", "streaks", "names", "types", "pulses", "coherence", "catalog", "galleries"]
log = None


def say(s: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    log.write(line + "\n"); log.flush(); print(line, flush=True)


def run(args: list[str], must: bool = True) -> int:
    t0 = time.time(); say("=== " + " ".join(args))
    p = subprocess.run([PY] + args, cwd=REPO, capture_output=True, text=True)
    log.write(p.stdout[-30000:]); log.write(p.stderr[-5000:]); log.write(f"--- rc={p.returncode} in {time.time() - t0:.0f} s\n"); log.flush()
    if p.returncode != 0:
        say(f"FAILED rc={p.returncode}: {(p.stderr.strip().splitlines() or [''])[-1][:200]}")
        if must:
            sys.exit(1)
    return p.returncode


def exists(*rel: str) -> bool:
    return all(os.path.exists(os.path.join(REPO, p)) for p in rel)


def main() -> None:
    global log
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STAGES, default=None, help="run this stage and everything after it, regardless of existing outputs")
    ap.add_argument("--only", nargs="*", choices=STAGES, default=None)
    ap.add_argument("--redo", action="store_true", help="run every stage")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(os.path.join(REPO, FIGS), exist_ok=True); os.makedirs(os.path.join(REPO, OUT), exist_ok=True)
    log = open(os.path.join(REPO, V2, "year_pipeline.log"), "a")
    forced = set(STAGES[STAGES.index(args.start):]) if args.start else set()
    if args.redo:
        forced = set(STAGES)
    todo = set(args.only) if args.only else set(STAGES)

    def want(stage: str, *outputs: str) -> bool:
        if stage not in todo:
            return False
        if stage in forced or not outputs:
            return True
        return not exists(*outputs)

    J = str(args.jobs)
    if want("lines", f"{V2}/y2024/fixed_rate_lines.csv", f"{V2}/y2025/fixed_rate_lines.csv"):
        run([f"{C}/fixed_rate_lines.py", *ROOTS, "--jobs", J])
    if want("link"):
        # --all skips nights already linked unless --redo; a forced stage re-links everything
        extra = ["--redo"] if "link" in forced else []
        procs = []
        for root in ROOTS:
            for tag in ("_night", "_day"):
                a = [PY, f"{C}/night_global.py", root, "--all", "--jobs", "5", "--tag", tag, *extra]
                say("=== " + " ".join(a)); procs.append(subprocess.Popen(a, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
        for p in procs:
            _, err = p.communicate(); say(f"link rc={p.returncode}" + (f" {err.strip()[-200:]}" if p.returncode else ""))
    if want("hours", f"{FIGS}/nov24_nov25_night_hours.csv", f"{FIGS}/nov24_nov25_day_hours.csv"):
        for per in ("night", "day"):
            run([f"{C}/year_hours_figure.py", *ROOTS, "--first", FIRST, "--last", LAST, "--out", f"{FIGS}/nov24_nov25_{per}", "--periods", per, "--suffix", "_global"])
    if want("streaks", f"{OUT}/streaks_coarse{SFX}.csv", f"{OUT}/animal_days_streaks{SFX}.csv"):
        run([f"{C}/year_streak_inputs.py", f"{FIGS}/nov24_nov25_night_hours.csv", OUT, "--roots", *WX_ROOTS])
        run([f"{C}/build_streaks.py", OUT, "--bR", "0.0154", "--bF", "0.0046", "--coupling", "0.55", "--coupling-hf", "0.25", "--along-tol", os.environ.get("BROOKS_ALONG_TOL", "0.6"), "--perp-tol", "0.06", "--suffix", "_k55b"])   # v3: 0.8 (METHOD 19 AK: 0.6 cut the long lines into temperature segments after the relink)
        run([f"{C}/split_streaks_by_T.py", OUT, "--in", "_k55b", "--out", "_v4", "--coupling", "0.55", "--slope", "2.4", "--sep-sd", "2.5", "--dbic", "50", "--max-k", "2"])
        run([f"{C}/merge_streaks.py", OUT, "--in", "_v4", "--out", "_v6", "--coupling", "0.55"])
        run([f"{C}/track_form.py", OUT, "--roots", *ROOTS, "--jobs", J])
        run([f"{C}/track_form_audio.py", OUT, "--unit-roots", *ROOTS, "--jobs", J])
        run([f"{C}/streaks_by_form.py", OUT, "--ref", "_v6", "--out", SFX, "--audio-form"])
        run([f"{C}/streak_overlay_figure.py", OUT, "--suffix", SFX, "--coupling", "0.55", "--out", f"streak_overlay{SFX}"])
        run([f"{C}/assign_day_to_streaks.py", OUT, "--day-hours", f"{FIGS}/nov24_nov25_day_hours.csv", "--suffix", SFX])          # positions first (writes the id list)
        run([f"{C}/track_form_audio.py", OUT, "--unit-roots", *ROOTS, "--jobs", J, "--ids-from", f"{OUT}/animal_days_streaks{SFX}.csv", "--out", f"{OUT}/track_form_audio_day.csv"])
        run([f"{C}/assign_day_to_streaks.py", OUT, "--day-hours", f"{FIGS}/nov24_nov25_day_hours.csv", "--suffix", SFX, "--audio-form", f"{OUT}/track_form_audio_day.csv"])
        for a, b in ((f"animal_nights_streaks{SFX}.csv", "animal_nights_streaks.csv"), (f"streaks_coarse{SFX}.csv", "streaks_coarse.csv")):
            shutil.copy2(os.path.join(REPO, OUT, a), os.path.join(REPO, OUT, b))
    if want("names", f"{OUT}/streak_sina_matches.csv", f"{OUT}/sina_reference_matches.csv"):
        sina = os.path.join(REPO, OUT, "sina")
        if not os.path.exists(sina) and os.path.exists(os.path.join(REPO, R, "month_v2", "sina")):
            shutil.copytree(os.path.join(REPO, R, "month_v2", "sina"), sina)
        run([f"{C}/match_sina.py", OUT], must=False)
        run([f"{C}/sina_reference.py", OUT, "--match"], must=False)
    if want("types", f"{OUT}/type_mix{SFX}.csv", f"{OUT}/year_thermal_limits{SFX}.csv", f"{OUT}/year_coupling{SFX}.csv", f"{OUT}/year_chirpers{SFX}.csv"):
        run([f"{C}/type_mix.py", OUT, "--suffix", SFX, "--roots", *ROOTS, "--period", "night"], must=False)
        run([f"{C}/year_types.py", OUT, "--suffix", SFX, "--roots", *ROOTS], must=False)
        if "types" in forced and os.path.exists(os.path.join(REPO, OUT, f"year_coupling_tracks{SFX}.csv")):
            os.remove(os.path.join(REPO, OUT, f"year_coupling_tracks{SFX}.csv"))
        run([f"{C}/year_coupling.py", OUT, "--suffix", SFX, "--roots", *ROOTS, "--jobs", J], must=False)
        run([f"{C}/year_chirpers.py", OUT, "--suffix", SFX], must=False)
        run([f"{C}/year_harmonic_ref.py", OUT, "--suffix", SFX, "--roots", *ROOTS, "--jobs", J], must=False)
    if want("pulses", f"{OUT}/pulse_shapes_streaks{SFX}.csv"):
        if "pulses" in forced and os.path.exists(os.path.join(REPO, OUT, f"pulse_shapes{SFX}.csv")):
            os.remove(os.path.join(REPO, OUT, f"pulse_shapes{SFX}.csv"))
        run([f"{C}/pulse_shapes.py", OUT, "--suffix", SFX, "--unit-roots", *ROOTS, "--cap", "100000", "--min-n", "40", "--jobs", J], must=False)
    if want("coherence", f"{OUT}/streak_coherence{SFX}.csv", f"{OUT}/streak_coherence_clear{SFX}.csv"):
        run([f"{C}/streak_coherence.py", OUT, "--suffix", SFX], must=False)
        run([f"{C}/streak_coherence.py", OUT, "--suffix", SFX, "--clear-only"], must=False)
    if want("catalog", f"{OUT}/type_catalog{SFX}.csv"):
        run([f"{C}/type_catalog.py", OUT, "--suffix", SFX], must=False)
    if want("galleries", f"{OUT}/gallery_fixed{SFX}.pdf", f"{OUT}/grid{SFX}.pdf", f"{OUT}/grid_sync{SFX}.pdf"):
        run([f"{C}/streak_gallery.py", OUT, "--suffix", SFX, "--unit-roots", *ROOTS, "--mode", "fixed", "--jobs", "6"], must=False)
        run([f"{C}/streak_grid.py", OUT, "--suffix", SFX, "--unit-roots", *ROOTS, "--jobs", "6"], must=False)
        run([f"{C}/streak_grid.py", OUT, "--suffix", SFX, "--unit-roots", *ROOTS, "--sync", "2", "--jobs", "6"], must=False)
    say("=== year_pipeline done")


if __name__ == "__main__":
    main()
