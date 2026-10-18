# BROOKS

**Bioacoustic Records of Orthopteran Oscillators, Keyed by Song.** Release 1.0.0, software and paper materials.

BROOKS decomposes continuous field recordings into individual singing insects without reference recordings or training
data, follows each singer through a night, and assembles a year of singers into call types: lines in the plane of
carrier frequency against syllable rate along which a population slides with temperature. This repository accompanies
the paper "A comprehensive inventory of discrete insect songs from long-term continuously operating field recorders" (M. B. Eisen, 2026) and holds the code, a demonstration data set, the year-scale tables behind the
paper's figures, the figures, and the per-type quality-control pages. It is named for Margarette Brooks, who in 1881
first timed a cricket's song against a thermometer.

The companion package for localizing birds on the same recorder arrays is
[cardinal](https://github.com/cardinal-acoustics/cardinal).

## Contents

- `code/`  97 Python modules: the detector (`window_blobs2.py`), the linker (`night_global.py`, `night_animals.py`), the
  corpus and year pipelines (`run_corpus2.py`, `year_pipeline.py`, `make_run_tree.py`, `carry_types.py`), the chirp plane
  (`chirp_year.py`), the reference-recording comparison (`sina_*.py`, `chirp_reference_z.py`, `streak_id_figure.py`,
  `reference_vs_streak_grid.py`, `anaxipha_published.py`, `inat_*.py`), the figure scripts (`*_figure.py`, `streak_grid.py`,
  `singer_night_grid.py`, `type_exemplar_figure.py`) and their helpers. `regenerate_figures.py` is the driver that drew
  every figure in the paper.
- `demo/`  two 5-minute windows of audio from the night of 20 September 2025 at the House station (the night of
  Figures 3 to 6), the detector's per-window output for the whole of that night with its tracks, the HRRR temperatures,
  and `run_demo.py`.
- `data/`  the tables behind Figures 7 to 19 and Table 1 (see `DATA_DICTIONARY.md`): singer-nights and singer-days with
  their streaks, the streaks, every singer-hour with its temperature, the paper's type numbers, the fitted lines, the
  within-singer coupling and temperature responses, pulse measurements, the song-form test, the chirp plane (typed
  singer-nights, clusters, the untyped bout-only class), our measurements of the SINA reference recordings and their
  tests against the lines, the Walker & Funk (2014) and iNaturalist tests, the type-reference folds, recorded hours,
  and every track of the year in two compressed tables.
- `figures/`  `Eisen_Orthoptera_Figures.pdf`, the figures as the preprint carries them, and the generated Figures 1 to
  19 as PDF and PNG with their tables (`fig07_year_plane_type_panels` is Figure 7-S1), Table 1, and `MANIFEST.md`, which
  gives the command that produced every figure.
- `supplement/`  the S7 pages, one letter page per type (37 trill types, 8 chirp types; raw and rate-filtered views):
  twelve positions along the type's line by five singers, with the detected band in color; the S8 pages, individual
  singers of each type through their nights; and the preprint's supplementary figure file. The 5 s listening clip behind
  every S7 cell is in the separate archive `brooks-v1-listening-clips.zip` (folders `S_grids_v5/audio/<page>/` and
  `S_grids_v5_chirp/audio/<page>/`, each with an `index.csv`).
- `docs/`  `METHOD_record.md` (the running record of what was tried, rejected and why), `REFERENCES.md` (the paper's
  references with verified links).

The full release, including the 194 MB supplementary PDF and the listening clips, is attached to the GitHub release
as `brooks-v1.zip` and `brooks-v1-listening-clips.zip`.

## Installation

    conda env create -f environment.yml
    conda activate brooks

or `pip install -r requirements.txt` into a Python 3.11 environment.

## The demo (about two minutes, no audio archive needed)

    python demo/run_demo.py

This detects the singers in each of the two included windows and writes the window's peaks, parts and singers tables
and a figure of the window as in Figure 4; then it re-links the included night's 144 windows of detections into tracks
(`demo/units/y2025/.../night_night_global_tracks.csv`). The environment variable `BROOKS_AUDIO_ROOTS` tells the
detector where audio lives; the demo sets it to `demo/audio`, whose folder layout (`<date>/<site>/<recorder>_<tag>/
<recorder>/<folder>/S<start>_E<end>_<gps>.wav`, times in UTC) is the one the archive uses.

## Running on your own recordings

1. Lay the audio out as above (or point `BROOKS_AUDIO_ROOTS` at the archive and set `BROOKS_TZ`), one folder per
   recorder, hour-long WAV files named by their UTC start and end.
2. Detection: `python code/run_corpus2.py --first-month YYYY-MM --last-month YYYY-MM --jobs N` writes one directory per
   recorder-period (night 19:00-07:00, day 07:00-19:00 local) with three tables per 5-minute window.
3. Linking: `python code/night_global.py <run_root> --all` joins each period's detections into tracks. The paper's
   settings are `BROOKS_SYLV2_LINK=1` (syllable rate read inside chirps) and `BROOKS_FIX_PAD_REL=0.06` (the carrier
   window pads the reference band by 6 % of its center, with a 150 Hz floor).
4. Temperatures: an hourly table `weather_hourly_hrrr.csv` (columns t in UTC, T_C) in the run root.
5. Year: `BROOKS_ALONG_TOL=0.8 python code/year_pipeline.py` runs the hours, streaks, types, pulses and galleries stages;
   `chirp_year.py` builds the chirp plane. The paper's catalog was then refined by inspection (clusters split where
   their pages showed two populations) and the resulting partition carried by singer-night with `carry_types.py`, as
   `docs/METHOD_record.md` (sections 19 AU to 19 BA) describes. Parameters are the module constants documented at
   the top of each file.

## Reproducing the paper's figures

Figures 7, 8, 9, 11, 12, 13, 15, 16, 18 and 19 and Table 1 are drawn from the tables in `data/` alone (point the scripts'
root argument at a directory laid out as `data/`, or at the run directory of your own year). Figures 3, 4, 5, 6, 10,
14, 17 and the supplement pages read audio from the archive and need the recordings. The command behind every figure
is in `figures/MANIFEST.md`; `code/regenerate_figures.py` runs them all.

## Notes

- No species name is attached to any type. Figures 12 to 14 show reference recordings beside the types for the reader
  to judge; the SINA recordings themselves are not redistributed here (they are at https://orthsoc.org/sina/), nor
  are the Walker & Funk (2014) tables or iNaturalist audio.
- Temperatures are modeled (HRRR 2 m), not measured on site.
- Chirp types are defined only for chirpers whose syllables were read inside the chirps; the other bout-rhythm sounds
  are listed in `data/chirp/bout_only_tracks.csv` and not typed.

## License and citation

Code under the MIT license (`LICENSE`); data, figures and supplement pages under CC BY 4.0 (`LICENSE-DATA.md`). Cite
the paper and this release (`CITATION.cff`).
