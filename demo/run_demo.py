"""Run BROOKS on the demo data (no archive needed).

  1. detect the singers in one 5-minute window of audio (window_blobs2.py), from the demo audio folder
  2. draw that window as the paper's Figure 4 does (window_figure.py)
  3. re-link the included night's per-window detections into tracks (night_global.py), with the paper's settings

Usage, from the release root:  python demo/run_demo.py
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); CODE = os.path.join(ROOT, "code")
OUT = os.path.join(HERE, "demo_out"); os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, BROOKS_AUDIO_ROOTS=os.path.join(HERE, "audio"), BROOKS_TZ="America/Chicago", BROOKS_SYLV2_LINK="1", BROOKS_FIX_PAD_REL="0.06")
PY = sys.executable
for t in ("2025-09-20 21:05", "2025-09-20 23:05"):
    stem = os.path.join(OUT, f"window_{t.replace(' ', '_').replace(':', '')}")
    subprocess.run([PY, os.path.join(CODE, "window_blobs2.py"), "--recorder", "BARLT_00008662", "--site", "House", "--time", t, "--out", stem, "--no-plot"], env=env, check=True)
    subprocess.run([PY, os.path.join(CODE, "window_figure.py"), "--recorder", "BARLT_00008662", "--site", "House", "--time", t, "--out", stem + "_figure"], env=env, check=True)
night = os.path.join(HERE, "units", "y2025", "BARLT_00008662_House_20250920_night")
subprocess.run([PY, os.path.join(CODE, "night_global.py"), night, "--redo"], env=env, check=True)
print("demo done:", OUT, "and", os.path.join(night, "night_night_global_tracks.csv"))
