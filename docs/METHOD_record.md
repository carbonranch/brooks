# Callers (BROOKS): window-first inventory of repeating insect calls

BROOKS = Bioacoustic Records of Orthopteran Oscillators, Keyed by Song (MBE,
2026-09-15), after Margarette W. Brooks, *Popular Science Monthly* 1881.

Started 2026-09-12 from scratch. The previous attempt is `analysis/soundscape/`;
its METHOD.md §0 records what was learned there and is assumed here. This file
records only what has been established on real audio in this directory.

## Idea

In a window of a few minutes a stridulating insect is stationary: carrier drift
(~-19 Hz/h) and rate drift (~-0.36 Hz/h) are both below measurement noise. In a
map of carrier frequency against amplitude-modulation rate it is one compact
blob, elongated along frequency (the band is 300-1000 Hz wide) and narrow in AM,
and its harmonics -- carrier n*F and modulation m*R -- are further blobs. Two
animals at the same rate and neighbouring frequencies land in ONE blob. Neither
the lattice nor the merger can be resolved in (F, R) alone. Both are resolved by
asking whether two pieces of the map switch on and off TOGETHER.

    blobs  --split by phase and on/off-->  parts  --link by co-variation-->  animals

Work per window first; connect windows into a night second.

## 1. `window_blobs.py` -- one window, the map, the animals

STFT 512/128 (86 Hz, 2.9 ms hop). Per carrier row, Welch spectrum of the
de-meaned dB series, 8-100 Hz, minus a 15 Hz running median of that row's own
spectrum. Blobs are 8-connected regions above 6 dB spanning >=2 rows. Each
blob's rate is refined on the central 30 s to ~0.01 Hz and the spectrogram is
folded at that period. Per row of the fold: modulation depth and phase. Per row
of the window: the amplitude of the level modulation AT R in each 0.5 s chunk
-- the R-locked on/off series, which a continuous neighbour at another rate
cannot contaminate.

Worked example: House / BARLT_00008662 / 2024-09-12 20:40, 300 s.
79 blobs -> 58 parts -> 13 animals, 35 AM-harmonic parts absorbed.

### Established

- **The blobs are the callers.** All four hand-annotated bands are found and
  the strong blobs fold to single clean units (5168-5943 Hz at 19.89 Hz: an
  arched glide inside a 50 ms period).
- **Rate resolves what frequency cannot.** Three animals share 2.3-2.8 kHz:
  42.98 Hz continuous, and two at 41.10 Hz (2.33-2.41 and 2.67-2.76 kHz) with
  different bout patterns. Rates 1.9 Hz apart are cleanly separate blobs.
- **Same-rate neighbours merge into one blob; phase and on/off separate them.**
  The 11.2-14.4 kHz blob at 20.61 Hz holds pieces of two animals. Rows of one
  animal share a phase or step through it smoothly (the glide); a jump marks
  another animal. Rows of one animal switch on and off together.
- **Harmonic flags need the refined rate.** On the 0.17 Hz Welch grid 30.41
  and 30.90 Hz look alike; refined, they are different animals.
- **90 s is too short for on/off; 300 s is enough here.** The 7.75-8.44 kHz
  animal switches on at ~190 s and is invisible as separate in 90 s. Nothing
  else about the method changes with window length: the R-locked series needs
  the rate to hold only within each 0.5 s chunk, and the fold stays on 30 s.
- **Co-variation must be high-passed and judged against a shift null.** Over
  300 s, unrelated animals co-vary through slow common trends (raw null p95
  +0.51); removing a 10 s running median drops the different-rate null to
  p99 +0.15. The circular shift of one series is the right null for the
  alignment of two series and calibrates itself to each pair's noise.
- **Chunks finer than 0.5 s are a trap.** At 0.1 s the rate resolution is 10
  Hz, so the 41.10 and 42.98 Hz callers sharing a band leak into each other's
  series and score z = 6 as "one animal". 0.5 s resolves rates ~2 Hz apart.
- **Chaining fails; a star works.** Single linkage on pairs passing r > 0.2,
  z > 4.5 chained the 8.2 kHz bout caller into the continuous 6.7 kHz animal
  through neighbouring rows, although the direct pair fails (z 3.4). Each
  part now joins only through an animal's lowest-frequency component.
- **AM harmonics are structural, not statistical.** A part at m*R (m = 2..6,
  within 0.5%) in the fundamental's rows (+/-300 Hz) is the same pulses seen
  at a harmonic of the rate. Their amplitude need NOT co-vary with the
  fundamental's (6.7 kHz fundamental vs its own 2x: r 0.04) -- the m-th AM
  harmonic measures pulse shape, not pulse presence. Testing them by
  co-variation created spurious "animals".
- **The 11-14 kHz region is mostly harmonics.** 13.0-13.7 kHz at 20.61 Hz is
  the 2nd carrier harmonic of the 6.7 kHz animal (r +0.40, z 10 with the
  fundamental), 19.7-20.4 kHz the 3rd. The glide is n times larger at the
  n-th harmonic and correspondingly easier to see.
- **A fold displayed on absolute level shows nothing in the dense 2-4 kHz
  chorus.** Display each row about its own mean.

## 2. Stability across windows (`sweep.py`, `window_stability.py`)

The per-window list run on 53 windows of 300 s at 60 s steps (80% overlap)
through House 8662, 20:00-20:53. Windows that share 80% of their audio should
give nearly the same list; how far they do measures the method, not the animals.

First version (first-harmonic phase-jump split, 0.5% rate tolerance): 442
tracks, 295 seen in one window only; a median 53% of one window's animals
reappear in the next; the 5.7 kHz animal alternated between one and two parts
and its blob merged with the 6.7 kHz animal's twice, taking its rate label.

Three causes, three fixes:
- **Phase is not single-valued through an arched glide.** The top of the arch
  sits ~0.2 period from its flanks; a per-row phase-jump threshold of 0.12 is
  right at the edge and flickers. Adjacent rows of one animal have
  near-identical FOLD PROFILES even through the arch, so the split now tests
  profile correlation between adjacent rows (< 0.3 cuts). This also keeps a
  1.5 kHz glide (2.76-4.31 kHz @ 28.8 Hz, House 02:40) whole where phase would
  have shredded it.
- **Neighbouring animals 0.85 Hz apart share one AM-map region**, so one rate
  label folded both. Regions are now divided along the AM axis by the AM bin
  of each row's own peak (gap > 2 bins = 0.34 Hz).
- **One animal's refined rate scatters ~0.1 Hz (0.5%) between windows**, the
  tolerance being used for "same rate". Now 1.5%.

After: 283 tracks, 166 seen once; median 67% of a window's animals reappear
in the next; the 5.7 kHz @ 19.9, 2.5 kHz @ 42.8 and 3.05 kHz @ 30.3 Hz animals
persist through 41-45 of 53 windows as single tracks. Counting animals, 17 ->
14 per window (median).

What is still unstable, in order of importance:
- **Which component is the fundamental.** The 6.7 kHz @ 20.7 Hz animal is
  matched across windows only because matching accepts any shared component:
  its 6.5-6.9 kHz component drops out in a third of windows and its 13 kHz or
  20 kHz harmonic is then reported as the fundamental. Component counts on
  persistent animals are constant in only ~21% of windows.
- **Faint parts.** 71% of tracks with depth < 0.25 are seen once; 47% of
  those at 0.4-0.6. Detections near the 6 dB threshold flicker, as expected;
  the 0.4-0.6 figure says the threshold is not the only reason.
- Animals per window still varies 8-20 on 80%-overlapping audio.

## 3. Night-animals (`sweep.py`, `night_animals.py`, `carrier_vs_rate.py`)

Direction chosen 2026-09-12 (MBE): define an animal across windows by its rate
and its whole component set, then use that to stabilise the per-window list.
The first half is done; the second is not.

`sweep.py` runs the window analysis every 300 s through a night (~8 s per
window, four in parallel: a night in ~5 min). `night_animals.py` links each
window's animals to open tracks -- rate within 1.5% of the track's latest,
ANY component band overlapping the track's union of bands, closest rate first,
up to 4 windows missed -- then stitches tracks end-to-start within 45 min at
3% in rate. The carrier trajectory is read from one reference band per track
(the band present in most of its windows) as the centroid of R-locked power
over a FIXED frequency window about that band (`F_fix`), from a per-row power
profile that `window_blobs.py` now stores with each part.

House 8662, 2024-09-12 19:00 -> 07:00, 145 windows: 2132 window-animals ->
661 tracks -> 387 after stitching; 11 span > 6 h, 30 span > 3 h. Back9 8613,
same night, 144 windows: 2408 -> 636 -> 313; 17 span > 6 h.

### Established

- **Rate is the night's thermometer, per animal.** Every long-lived animal's
  rate falls 2-7% from evening to ~04:00 and turns up towards dawn; on Back9
  several drop together in one step near midnight.
- **The carrier measured over a fixed window is stable enough to see hourly
  shifts.** Within-track scatter, which includes real drift: 6.8 kHz animal
  225 -> 62 Hz going from part-centroid to fixed-window centroid; 2.6 kHz
  animal 32 -> 11 Hz; a 17 kHz component 2290 -> 32 Hz (the part-centroid
  had been jumping between components).
- **Carrier follows rate in some animals and not others.**
  House, 3.10 kHz @ 29.4 Hz, 77 windows over 8.7 h: carrier 3150 -> 2990 Hz as
  rate 30.5 -> 28.3 Hz, r = +0.77, 0.64% carrier per 1% rate.
  Back9, 3.18 kHz @ 29.6 Hz (a different individual of the same type), 57
  windows: r = +0.69, 0.71% per 1%, and the carrier steps down at the same
  minute as the rate.
  House, 2.59 kHz @ 43.0 Hz, 48 windows: rate swings +/-2%, carrier flat to
  +/-5 Hz, r = -0.16. Back9, 7.0 kHz @ 42.6 Hz: 0.23% per 1%.
  So the coupling differs by type -- consistent with carriers set by a
  temperature-dependent tooth-strike rate in some taxa and by a resonator in
  others -- and agrees between individuals of one type at two sites.

Across recorder-nights (`compare_nights.py`, House 09-12, Back9 09-12, House
09-11 -- the last only 109 windows, audio ends ~04:00): the 3.1-3.2 kHz @
29.4-30.2 Hz type is present on all three with slopes 0.64, 0.71, 0.38
%/% (the last on a 3.4 h track with a third of the rate range). The
2.59 kHz @ 42.9 Hz House animal is the clean uncoupled case (-0.07, r -0.16).
Matching at 4% in rate and 6% in carrier is too loose in the 6-7 kHz @ 19-21 Hz
region, where it chains 19 tracks into one "type"; that region needs joint
(F, R) resolution at the ~2% level before types can be compared there.

## 4. The month (`run_month.py`, `month_summary.py`, `type_gallery.py`, `temperature_effects.py`)

All 18 BARLT recorders, nights of 2024-08-17 to 09-13: 498 recorder-nights,
5970 night-hours, 72,131 windows (every window that has audio), 9.7 h on this
machine with 12 window jobs plus two short redo passes. Windows with no blob
above 6 dB, or with blobs but no part clearing the depth floor (quiet or cold
stretches), had crashed on empty tables; both now write an empty inventory.
The redo passes changed none of the pooled numbers below -- the windows they
filled were empty.

Established:

- 8554 night-animals persist >= 2 h, 3702 >= 6 h; median 18 long-lived
  animals per recorder-night. Back9 8613 and EastCreek 8556 are the richest
  recorders, Northside 8051/8240 the sparsest.
- **In carrier x rate the month's long-lived animals form diagonal streaks,
  not clouds.** Each streak is one species seen across a month of
  temperatures. Point-clustering "types" over-splits along the streaks.
- **Temperature (Open-Meteo ERA5 hourly, 2 m, at the site).** Within an
  animal-night, rate rises +1.64 +/- 0.01 %/degC (R^2 0.30, Q10 1.18) and
  carrier +0.33 %/degC; per animal-night slopes are positive for 80% (rate)
  and 70% (carrier). Across nights, every type with enough data shows the
  same rate sensitivity, 1.5-1.8 %/degC -- within-night and between-night
  slopes agree. The streaks are coloured by temperature along their length.
- **The silent nights were cold, not wet.** 09-07 to 09-09 had minima of
  13-15 degC and 45-125 long-lived animals across all recorders against 300-390
  on warm nights (Spearman +0.52 with night minimum); the two rain nights
  (08-18, 09-11) also sit low.
- **No date effect once temperature is in the model.** Per corrected type,
  animal-night medians regressed on temperature and day-of-year: the date
  coefficient is +0.00 %/day for rate and +0.01 %/day for carrier (medians
  over 159 types), with 2 of 159 types beyond |z| = 3 for each -- chance
  level. Growth or seasonal drift larger than ~0.5% over the month is excluded
  for these types.
- Caveat on the slopes: ERA5 is a 9 km grid, not the microclimate at the
  recorder; errors in T attenuate the fitted slopes, so 1.64 %/degC is a lower
  bound. On-site temperature (if the recorders log it) would sharpen it.

## 5. Names: matching streaks to SINA (`sina_fetch.py`, `sina_regressions.py`, `match_sina.py`)

Streaks (71 with >= 15 animal-nights, clustered in temperature-corrected
(F, R) at 10%/12%) were matched to Singing Insects of North America species
on rate at 25 degC and dominant frequency. Sources, in order of trust: Walker's
per-recording spreadsheets (53 species; rate regressed on temperature, kHz
where recorded), the "Song at 25 degC" prose, SINA's quotes of Weissman & Gray
2019 for Gryllus, Walker & Funk 2014 for Anaxipha. Species were kept only if
SINA's distribution table has records in AR or a neighbouring state.
Tolerances 20% in rate and 18% in carrier, because our rates were brought to
25 degC with a slope fitted against ERA5 grid temperature.

**Walker's data calibrate our thermometer.** His median rate slope is 4.8
%/degC (IQR 3.0-6.2) against our 1.64 from ERA5: grid temperature is a damped
proxy for the insect's, so our absolute slopes are lower bounds. The carrier/
rate slope ratio survives the damping; his tree crickets give 0.54-0.76 and
ground crickets ~0.5, where our tree-cricket-matched streaks sit (0.35-0.60).

**What can be named with some confidence** (rate and carrier both within
~10%, range plausible, coupling consistent):
- 2769 Hz @ 45.3 Hz (206 animal-nights): Oecanthus latipennis, broad-winged
  tree cricket (2.87 kHz @ 45.6).
- 3644 @ 36.4 (183): Oecanthus texensis (3.71 @ 40.3, Walker slope 5.19,
  coupling 0.55); O. pini and O. quadripunctatus are alternatives.
- 3445 @ 42.4 (48): O. pini, pine tree cricket (3.4 @ 45).
- 3857 @ 57.2 (36, coupling 0.60): O. nigricornis / forbesi group (4.04 @ 53,
  coupling 0.65).
- 3567 @ 50.4 (40): Velarifictorus micado, Japanese burrowing cricket (3.6 @
  49.3), the introduced species; its low rate slope (1.1 %/degC) is testable.
- 4425 @ 44.0 (31): Gryllus fultoni, southern wood cricket (4.5 @ 45).
- 4212 @ 29.5 and 4482 @ 31.7 (205): Gryllus, fall field cricket group
  (G. pennsylvanicus / vernalis, 4.7 @ 25-31).
- 7363 @ 46.9 (85): Anaxipha exigua, Say's trig (7.2 @ 44).
- 8864 @ 30.8 (59): Conocephalus strictus, straight-lanced meadow katydid
  (9 @ 31.6); 12.1 kHz @ 19-21 (178): C. nemoralis (14 @ 18.4), weaker.

**What cannot be named from SINA, and dominates the data:** the
5.4-7.6 kHz @ 17-25 Hz complex (~3200 animal-nights, the largest streaks) and
5621 Hz @ 42.4 Hz (661). These are small ground crickets and trigs
(Allonemobius, Eunemobius, Anaxipha, Falcicula) whose Walker spreadsheets are
offline (Anaxipha) or absent (A. fasciatus, the obvious common candidate at
7.4 kHz, has no published rate on SINA). Anaxipha thomasi (7.8 @ 21) is the
closest published song to the largest streak but is known only from OH, PA,
GA. Also unnamed: 21.6 kHz @ 23.4 (173), 19.3 kHz @ 20.3, and three
low-frequency streaks at 2.2-2.4 kHz.

### Reference recordings measured with the same instrument (`sina_reference.py`, `reference_gallery.py`)

SINA's species pages carry Walker's recordings (WAV, 44.1 kHz, median 20 s)
captioned with the recording temperature. 185 calling songs of 141 species
recorded in AR or a neighbouring state were run through `analyse_signal`
(the window analysis on a clip; fold over the whole clip). Two things had to
be handled: tape hum and rumble make "animals" at 250-1100 Hz that are often
the deepest thing in a clip, so the component nearest the page's stated
dominant frequency is taken (or the deepest above 1.5 kHz when the page gives
none); and species whose rate is outside 8-100 Hz (coneheads at 150-200 p/s,
tickers below 8) measure nonsense and are excluded by that rule. 134 clips
measure at the page frequency; 64 fall within 2.5 sd of a streak's line in
(log R, log F) -- a comparison that needs no thermometer, since the reference
point either lies on the streak's own temperature line or does not.

Confirmed by shape as well as by numbers (reference fold beside the field
exemplar's fold):
- 3644 Hz @ 36.4 Hz (183): Oecanthus texensis at 24.8 degC lands at z -0.08
  and its fold is the same descending S-glide.
- 3023 Hz @ 49.7 Hz (247): O. latipennis at 22.0 degC, z -0.77, same
  down-glide at the fundamental.
- 5010 Hz @ 50.5 Hz (97): Gryllus rubens at 24.0 degC, z -1.8, and the same
  narrow flat burst -- the resonant field-cricket signature -- where the
  alternative Orocharis saltator is a steep glide.
- The other Oecanthus land on their own streaks: exclamationis (z +0.3),
  celerinictus (-0.8), niveus, nigricornis (+0.8), argentinus (-0.5),
  quadripunctatus (-1.5, -1.8), pini (+0.1).

The dominant streaks:
- 7049 Hz @ 22.7 Hz (1135): Anaxipha thomasi at 25.5 degC lands at z +0.7 and
  its fold -- a compact flat 20 ms burst -- is the closest in shape; A.
  walkeri (z +0.7) is a broad glide, A. tinnulus has a second band at 11 kHz.
  A. thomasi is known from OH, PA and GA only; if this is it, Arkansas is a
  large range extension, so the name is provisional.
- 7554 @ 24.6 (849), 6447 @ 21.1 (729), 5621 @ 42.4 (661), 5115 @ 38.1
  (271): candidates land on the lines (A. walkeri, Amblycorypha carinata,
  Phyllopalpus pulchellus, Anaxipha vernalis) but none reproduces the field
  fold; 5115 @ 38.1 is a glide, 5621 @ 42.4 a flat burst, and the references
  for both are the other shape. Unnamed.
- 13576 @ 22.3 (187) and 21625 @ 23.4 (173): narrow flat bursts with no
  reference of the right shape among the species SINA records for the region.

### Shape (`fold_shape.py`)

Position on a streak line says a reference *could* be the animal; the fold
says whether it is the same kind of call. Each fold is put on a common grid
(phase as a fraction of the period x frequency offset from the band's power
centroid, +/-517 Hz) and compared by normalised correlation over all circular
phase shifts, with duty (fraction of the period the unit occupies at half
power) and glide (signed ridge excursion across the unit) read off the same
grid.

**Within-streak consistency first.** Six exemplars per streak from different
recorders, nights and temperature bins:

| streak | Hz @ Hz | n | mean pairwise corr | min |
|---|---|---|---|---|
| 101 | 3644 @ 36.4 | 183 | 0.82 | 0.73 |
| 63 | 3023 @ 49.7 | 247 | 0.82 | 0.61 |
| 151 | 7363 @ 46.9 | 85 | 0.82 | 0.69 |
| 88 | 5621 @ 42.4 | 661 | 0.81 | 0.69 |
| 113, 122 | | | 0.72 | 0.52-0.56 |
| 96, 78, 86, 49 | | | 0.60-0.65 | 0.26-0.32 |
| 128 | 6447 @ 21.1 | 729 | 0.57 | 0.23 |
| **120** | **7049 @ 22.7** | **1135** | **0.53** | **0.04** |

The tree crickets, the 5.6 kHz @ 42 Hz and the 7.4 kHz @ 47 Hz streaks are
one shape each. **The largest streak, 7049 Hz @ 22.7 Hz, is not one species**:
its exemplars include flat bursts at 7.2 kHz, glides from 5.8 to 6.8 kHz and
from 8 to 9.5 kHz, and a broadband unit. It is a (carrier, rate) coincidence
of several small crickets, and every earlier statement about "the dominant
type" should be read as about that mixture. 128 and the two high streaks (49,
43) are also mixed.

**Shape against references** (correlation with the field exemplar, references
pre-filtered to +/-25% carrier, +/-35% rate):
- 101: Oecanthus texensis 0.91; 63: O. texensis 0.90, O. latipennis 0.79.
  Oecanthus species all correlate 0.7-0.9 with each other -- shape separates
  genera, not congeners; position on the line does the rest.
- 122 and 128: Anaxipha thomasi 0.81 and 0.79 (a flat 20 ms burst with a
  slight down-glide), A. tinnulus 0.76 on 128.
- 86 (glide, -126 Hz): Anaxipha fultoni 0.86, A. vernalis 0.77 -- a trig-type
  glide; neither species is plausible here, so this is an unrecorded trig.
- 88: Anaxipha scia 0.76, A. fultoni 0.73, Gryllus rubens 0.71 -- trig-like.
- 78: nothing above 0.65; the field fold is a narrow flat burst, G. rubens
  0.62 remains the position match.
- 151: Neonemobius palustris 0.81, Anaxipha exigua 0.79.
- 49, 43: nothing above 0.7 with a plausible species.

### Shape through a night (`night_shape.py`)

The longest single animal-night in each of eight consistent streaks, its fold
at ~hourly spacing. Within one animal the unit shape -- the slant of the
glide, the duty -- is the same at 20:30 and at 05:55 while the period
stretches with the cooling (streak 101, O. texensis candidate: 26.3 -> 27.3
ms over 23.9 -> 21.7 degC; streak 151: rate 45 -> 38 Hz over 21.2 -> 18.5
degC with an unchanged flat burst). Duty and glide show no consistent
temperature trend across the eight animals (correlations of both signs,
|r| < 0.7 on 5-8 points). The fold is therefore a temperature-invariant
signature of the animal where carrier and rate are not, which is what makes
it usable for typing across a season.

The same plots expose night-linking errors that the numbers hid: in three of
eight tracks one window has a different shape and band -- streak 96 jumps
from 4.0 kHz to 3.2 kHz at 01:40, streak 122's 19:05 window is a wide glide
where the rest are flat bursts, streak 113's dawn windows are another animal.
Shape continuity should be a criterion in `night_animals.py`.

### The fold display was hiding the edges of the unit (MBE, 2026-09-14)

Every fold so far was drawn as dB about each row's own mean. That divides a
row's in-phase modulation by whatever steady power shares the row, so where a
louder continuous neighbour sits (Northside 8259 09-04 20:30: rows at 3.1 and
3.85 kHz are 5-15 dB louder than the animal's and swing ~1 dB) the animal's
energy in that row vanishes from the picture although the fold contains it.
MBE's point: out-of-phase signal averages to a constant, in-phase signal
survives at full linear amplitude, so the field should not be flat. Drawn as
the in-phase residual (fold minus row mean, linear, or in dB below its own
peak) the same fold shows the glide continuing above the annotated band to
~3950 Hz and, at 3.0-3.3 kHz under the loud neighbour, in-phase energy 4-7 dB
below the peak at the phase the glide would reach if it kept descending.
The unit may sweep ~900 Hz over most of the period; the dB display showed the
~400 Hz that out-competes the neighbours. Consequences: the annotated bands
are where the animal wins the row, not the call's extent; extent and duty
from the dB fold are truncated by neighbours; the shape descriptors and the
displays should use the in-phase residual, with a phase-coherence test to
separate a faint continuation from the speckle of far rows.

Site observations (MBE, visual IDs at the recording location, not
comprehensive): Schistocerca obscura, S. americana, Melanoplus
differentialis, M. scudderi complex, Chortophaga viridifasciata (grasshoppers,
not nocturnal singers), Ellipes minuta (pygmy mole cricket, no song), and
Conocephalus strictus -- which supports 8864 Hz @ 30.8 Hz (59) = C. strictus
(Walker's data 9 kHz @ 31.6 at 25 degC; the reference clip measured at its
2x rate, 12.6 kHz @ 64, and so did not land on the line). Range data for
trigs and ground crickets in Arkansas are thin on SINA and iNaturalist alike,
so "no AR records" is weak evidence against A. thomasi or any other candidate
in that group; the state filter should not be applied to them.

### Fix: rows join a part on absolute in-phase amplitude (2026-09-14)

The truncation above was not only in the display. A row entered a part only
if (i) the blob finder had it -- the AM map is the Welch spectrum of the
row's dB series, i.e. modulation relative to the row's own power -- and (ii)
its relative depth 2|c|/mean exceeded DEPTH_MIN = 0.15. Both tests divide the
animal's in-phase power by whatever else is in the row, so a louder steady
neighbour removed rows the fold plainly contained. `window_blobs.py` now:

- `row_phase` also returns the absolute first-harmonic amplitude 2|c|/n
  (linear power), which a steady neighbour cannot dilute;
- `split_rows` still seeds parts as before (blob rows, depth >= 0.15, profile
  continuity), then extends each part row by row into the padded rows
  (+/-300 Hz) while the row's amplitude is within AMP_FLOOR_DB = 15 dB of the
  part's peak, its depth is >= DEPTH_EXT_MIN = 0.02 (modulated at all), and
  its residual profile correlates >= 0.3 with the row it extends from --
  phase continuity, so a same-rate neighbour with a different phase stops
  the walk;
- the profile used for continuity is the in-phase residual (fold minus row
  mean, unit peak) rather than dB about the row mean;
- `carrier_estimates` and the stored per-row `prof` use the amplitude. This
  is algebraically the old weight (mean x depth = 2|c|/n), so F_c and F_fix
  are unchanged for parts that did not extend -- verified to 0.01 Hz;
- a part's `depth` is the mean over its seed rows only, so extension does not
  dilute it.

Checks. House 8662 09-12 20:40: 22 -> 21 fundamental parts, 13 -> 12
animals; the lost part is 9561-9647 Hz @ 60.82, the (3F, 2R) fragment listed
below under "Not yet right", now absorbed because the 3.05 kHz @ 30.41 Hz
animal's 3F component extends to 9475 Hz and the +/-300 Hz nesting bound is
met. Bands widened by 1-6 rows; the two 41.1 Hz animals at 2.3-2.5 and
2.6-2.75 kHz stayed apart. Northside 8259 09-04 20:30: 14 parts both ways;
the 38.0 Hz glide (streak 101) is 3445-3790 Hz (was 3445-3704) and its
38.6 Hz neighbour 3618-4048 (was 3790-4048): the two calls overlap by two
rows, which is what the folds show, and each fold sees only its own animal
because 1.5% in rate over a 30 s fold is 17 periods of drift. The 3.0-3.3
kHz tail seen in the residual display is below the 15 dB floor here.

The 2025 run (`runs/y2025`) uses this rule from the start; `runs/month`
(the 2024 September analysis above) was made with the old rule, and its 492
old-rule 2025 units are in `runs/y2025_oldband`. The month should be rerun
before the shape descriptors and extents are compared across years.

### Not yet right (night level)

- Tracks in a dense band (2.3-2.8 kHz, three animals at 39-43 Hz) can
  exchange members; the House 2.6 kHz @ 40.9 Hz track's carrier falls 100 Hz
  with no rate change and is probably a mixture.
- The per-window list is not yet informed by the night (direction (b)).
  Faint components still come and go, and a track's reference band is a
  majority vote rather than a model.
- Stitching at 3% in rate over 45 min could join different individuals of one
  type calling in relay; nothing yet checks the join for carrier continuity.

### Not yet right

- Combined harmonics (n*F, m*R) with both n, m >= 2 are absorbed only when
  the part is alone in its animal and nests in n x a component within 300 Hz;
  9216-9647 Hz at 60.76 Hz is the 3.05 kHz @ 30.41 Hz animal's (3F, 2R) by
  eye and misses the nesting bound by 44 Hz.
- Faint same-rate parts far apart in frequency (9.9 and 16.5 kHz at 45.1 Hz)
  with no detectable fundamental stay as separate animals.
- Two different animals of one species at the same rate can co-vary weakly
  (6.7 vs 8.2 kHz: z 3.4). The 4.5 threshold separates them here; whether it
  does in general is untested.
- One window, one recorder. Nothing has been run on Back9 or a second night.

## 6. Thermometers (2026-09-14/15; `streaks_by_recorder.py`, `streak_position_vs_T.py`, `residual_decomposition.py`, `streak_thermometer.py`, `chorus_thermometer.py`, `reweather.py`, `meter_stations.py`, `sm4_temperature.py`, `sm4_vs_grid.py`)

- **The lines are the same on every recorder.** Carrier x rate drawn per
  recorder (18 panels, hourly points): every recorder's points lie on the
  pooled streaks with the same temperature gradient; only occupancy differs
  (tree-cricket lines dense on Back9, thin on Northside 8051; 8.5 kHz @ 59 Hz
  strong at House, absent at Northside).
- **Position along the line vs T.** Projecting each animal-night (or
  recorder-hour) onto its streak's principal axis and expressing it as a
  rate: slopes 1.0-1.7 %/degC, r 0.45-0.78 for single-species streaks, sd
  about the fit 3-5 %, off-line sd 2.3-4.4 %. Hourly resolution (4-5x the
  points) changes neither slope nor scatter; ERA5's one value per hour for
  the property stacks all recorders into columns 3-4 % tall. Mixture streaks
  (88, 122) are flat clouds.
- **The residual is a per-animal-night offset (85 % of variance; same-animal
  r 0.73) and, for real species, it is shared property-wide by the hour**:
  same-hour correlation on OTHER recorders equals that on the same recorder
  (101: 0.83 vs 0.77; 88: 0.63/0.64; 63: 0.56/0.52; 61, 86 ~0.45). Not
  microclimate, not individuality: ERA5's hourly error. For 120/122/128/124
  the offset is shared with nobody (0.1-0.2) -- a mixture diagnostic that
  agrees with shape. Hourly residuals form two blocks, {63,101,88,122,43,
  154,156} and {61,86,87,128,124} (0.6-0.8 within, 0.2-0.4 between):
  probably two strata. 61 and 63 (both "O. latipennis" by position) are in
  different blocks.
- **One reference animal is no thermometer** (its own offset is noise in x;
  3.8 -> 3.9 %). **The clean chorus is**: ERA5 + the hourly mean implied
  error of the other single-species streaks (common slope 1.5 %/degC, median
  over streaks, >= 4 animals and >= 2 streaks) takes left-out streaks from
  sd 4.1 -> 3.5 %, r 0.58 -> 0.72 (101: 0.39 -> 0.74); mixtures unchanged.
  Correction sd 1.96 degC, +/-5, uncorrelated with T, multi-day structure.
  First attempt divided by each streak's own slope and blew up (88's slope
  0.55 turned its offsets into +/-9 degC); use the common slope.
- **METER ATMOS 41 stations (five, installed 2026-09-09).** ZENTRA export
  clock is the account's zone (Pacific): solar first/last light 2 h early;
  shift +2 h. Against the 9 km ECMWF analysis (the archive's fill for the
  last ~5 days; true ERA5 arrives later): day -0.4 +/- 2.0, night -3.4 +/-
  2.4 degC (station minus grid), r night 0.68; hourly error sd 2.3 degC --
  the chorus said 2.0. Between stations at night sd 0.8, range 2.0 degC;
  Northside coldest (-1.0 vs Back9), House and Snake Palace (z6-38205)
  warmest (+1.0).
- **HRRR (3 km, Open-Meteo historical-forecast API) beats ERA5 at night**
  against the stations: bias +2.3 vs +3.4, sd 1.7 vs 2.2, r 0.88 vs 0.71.
  Re-temperaturing the month with HRRR (streaks fixed; `reweather.py`):
  animal-nights 0.8 +/- 1.3 degC colder; along-line slopes 1.43 -> 1.65
  %/degC (hourly median), r and sd unchanged; within-animal slope 1.63
  (was 1.64). ERA5 outputs kept in `runs/month/era5_outputs/`;
  `weather_hourly.csv` is now HRRR; HRRR for 2025 fetched to `runs/y2025/`.
- **SM4 housing temperature** (`*_Summary.txt`, 1.47 M readings, 12
  units, 2023-2026). Dawn-to-dusk schedules in 2024 and most of 2025: no
  readings 22:00-04:00. 24 h coverage only Back9 S4A20237 Mar-Jun 2025 and
  Northside S4A20258 Sep-Dec 2025. Housing sheds the day's heat by ~22:00;
  after that r 0.98-0.99 vs HRRR, sd 1.2-1.4, +0.4-0.9 degC above HRRR, so
  ~1.5-3 degC above air. No overlap with the stations; MBE will co-locate a
  unit with a station on the next visit. Frontier Labs WAV metadata has no
  temperature. Camera traps take their temperature from the network.
- Other sources checked: nearest archived ASOS are Arkadelphia and Texarkana
  (55-57 km; Hope AWOS not archived); ERA5-Land no better than ERA5 here.

## 7. The month rerun with the corrected band rule (`runs/month_v2`, 2026-09-15)

498 recorder-nights, 20.3 h at 6 jobs alongside the 2025 run. Directories
were written with a `_night` suffix and mixed NorthSide/Northside case (the
audio path's case varies by day); renamed to the `runs/month` convention so
the downstream scripts run unchanged. `build_streaks.py` now holds the
streak clustering that had been an inline command. Weather = HRRR.

- **Bands tripled.** Median fundamental band of a window-animal 172 -> 517
  Hz. Window-animals per night 1838 -> 1737, tracks per night 306 -> 248
  (fragments joined), components per animal 1.59 -> 1.67, long-lived
  animal-nights 8554 -> 8481, >= 6 h 3702 -> 3432.
- **Carrier slope 0.32 -> 0.46 %/degC; rate 1.63 -> 1.54 (HRRR).** Every
  coupling rose: O. texensis 0.45 -> 0.73, O. latipennis 0.39 -> 0.55; streak
  lines d log F / d log R 0.6-0.7 for tree crickets (were 0.4-0.6). The
  truncated bands measured the carrier where the animal won the row, an edge
  that moves less with temperature than the unit.
- **63 streaks** (were 71). The two largest v1 streaks (120: 7049 @ 22.7 and
  128: 6447 @ 21.1) merged into v2 125 (1612 animal-nights). Map old -> new in
  `streak_map_old_to_new.csv` (v1 -> v2: 120/128 -> 125, 122 -> 128, 88 -> 62,
  156 -> 106, 86/87 -> 61, 63 -> 43, 61 -> 46, 124 -> 129, 49 -> 31, 101 -> 75,
  43 -> 21, 154 -> 102, 53 -> 32, 96 -> 72, 151 -> 101, 78 -> 60).
- **Names.** O. texensis on 75 at z -0.56 (shape consistency 0.80);
  O. latipennis on 46 at z +0.21 (0.79) -- in v1 the latipennis clip had sat on
  the 3023 @ 49.7 streak (now 43), where one O. fultoni clip (30.3 C) lands at
  -0.8 and another (24.5 C) misses by 6: the two fultoni clips disagree with
  each other. G. rubens on 60 now equivocal (two clips at +4.6 and -4.2).
  A. thomasi on 128 at +1.3. Position-only candidates: Allonemobius fultoni
  for 125, A. vernalis for 62, O. quadripunctatus for 61 (which merged the
  unrecorded trig glide 86 with 87).
- **Shape consistency moved.** Tree crickets 0.75-0.80 (fine). But 62 (was 88:
  0.81 -> 0.36) and 101 (was 151: 0.82 -> 0.48) collapsed: the shape grid
  (+/-517 Hz about the band centroid) was tuned on narrow bands; with bands
  three times wider the centroid and the patch move. Re-examine before using
  shape to count species.
- **Thermometer results reproduce**: position-vs-T hourly slopes 1.0-2.2
  %/degC (HRRR), residual 89 % per-animal-night offset, same-hour-other-
  recorder r 0.81 (75), 0.70 (62), 0.55 (46); chorus correction takes clean
  streaks r 0.54 -> 0.70; correction sd 2.0 degC.
- **Artefact tests** (`clock_check.py`, `harmonic_coupling_check.py`,
  `sm4_hardware_check.py`): sample clock -25 ppm, -0.45 ppm/degC (firmware
  X1_PPM tracks it) -- 1e-5 of the effect and species-blind; 2F couples like F
  (0.44 vs 0.50, r 0.85 across 20 tracks, log-ratio slope -0.03); depth adds
  nothing (partial coupling 0.33 -> 0.33; depth _|_ rate within tracks); peak
  coupling 0.46 > centroid 0.33. SM4 64 m from BARLT 8662: shares too few
  animals for a same-animal test (unambiguous pairs still spread over the full
  1.5 % rate tolerance); needs co-location. Do it with the SM4-beside-station
  deployment: SM4 + BARLT + station at one spot. **At the population level
  the SM4 draws the same streaks** (`sm4_vs_barlt_streaks.png`): its 1354
  window-animals from the 202 shared windows lie on the BARLT's lines with
  the same slopes and temperature gradient. The plane is not the recorder's.
- **Coupling by subfamily** (`coupling_by_streak.py`). Within-animal
  coupling (median rel_slope of tracks with >= 0.3 Hz rate range) per streak:
  tree crickets 0.40-0.66 (76-96 % of tracks > 0.15); trigs/ground crickets
  0.24-0.43; Gryllus (72, 70) 0.25-0.34; katydid-candidate streaks at 8-22 kHz
  (11, 16, 21, 31, 91, 115, 116) -0.04 to +0.17 -- uncoupled species exist
  here and sing in September, as a minority. Between-night line slopes
  (0.33-0.44 for the small crickets) are steeper than within-animal
  couplings: between-individual covariance. Walker's own kHz slopes exist
  for 5 Oecanthus (0.54-0.76) and 2 ground crickets (0.51-0.56) only; no
  carrier column for Gryllus or katydids. Corollary: log F - s log R with s
  per group is a temperature-invariant species coordinate.
- **Every track** (`streaks_all_tracks.py`): 121,825 night-animals with a
  carrier; streaks visible with no persistence filter; the >= 2 h filter
  removes a diffuse background, not other lines; only the < 15 Hz / < 3 kHz
  corner and the > 15 kHz band are populated solely by brief tracks.

## 8. References found (2026-09-16/19)

- Hayward 1901, Psyche 9:179 (not vol. 3): 31 counts of Pterophylla phrase
  rate vs T, 13.5 %/degC. `refs/Hayward_1901_*`.
- Fulton 1915, NY Agr. Exp. Sta. Tech. Bull. 42: first statement that
  tree-cricket PITCH varies with temperature (niveus C -> D warm evening;
  latipennis half a note lower on cool nights). `refs/Fulton_1915_*`.
- **Riede 1993, Ambio 22:546** -- Figure 3 IS the carrier x pulse-rate
  plane, 54 hand-recorded songs from Ecuador grouped into ethospecies; the
  origin of the representation (I had wrongly dropped it as unverifiable).
  `refs/Riede_1993_notes.md`; scan in `Orthoptera Papers/`.
- **Walker & Funk 2014, J. Orthoptera Res. 23:1** (read in full 2026-09-22;
  had been used only for Anaxipha rates). Fig. 8 is the carrier x pulse-rate
  plane with six species as temperature LINES, identified "with no need that
  the temperature ... be known" -- the direct precedent for our reading of
  the streaks. Fig. 4: pulse-rate lines converge at 2.7 +/- 2.2 degC (implies
  ~5 %/degC at 25 degC; our 1.67 against HRRR is damped ~3x). Table 4: CF
  temperature coefficients by taxon -- Gryllus ~0.05-0.07 (escapement, carrier
  fixed) vs Oecanthus / Anaxipha / Nemobiinae 0.23-0.49 (tooth-impact control,
  carrier follows rate): the published basis for coupled vs uncoupled streaks.
  Names: 7.4 kHz @ 47 matches A. exigua (in range, in season); thomasi
  matches the 7 kHz complex numerically but is Appalachian; vernalis is a
  spring singer. `refs/Walker_Funk_2014_notes.md`.

## 9. Fold the loudest 30 s, not the central 30 s (2026-09-19)

`best_segment()`: per peak, the R-locked on/off series at the COARSE rate
(0.5 s chunks, averaged over the peak's rows), a running sum over 30 s, and
the maximum -> the stretch that is folded and on which the rate is refined.
Stored as `fold_t0` in the peaks table and `cols` in `per_blob`; night_shape
refolds the same stretch. Reference clips (< 30 s) fold whole, as before.

Why: a caller silent in the central 30 s gave a flat fold, depth < 0.15, no
part -- it vanished from the window although its peak was in the map; or
the fold locked onto a same-rate neighbour that WAS singing.

House 20:40: 12 -> 16 animals; existing animals' depths rise (3.13 kHz @
30.4: 0.20 -> 0.52; 9.5 kHz @ 30.9: 0.46 -> 0.71; 3.3 kHz @ 46: 0.23 ->
0.52); the new ones are bout callers (on_frac 0.17-0.5). Northside 20:30:
9 -> 12.

Stability sweep (53 windows at 60 s steps, House 20:00-20:53; `_seg` tag):
fraction of a window's animals present in the next 0.67 -> 0.76 (min 0.28
-> 0.50); animals/window 14 -> 19 (range 8-20 -> 10-26); tracks seen once
166 -> 135; tracks in >= half the windows 2 -> 4, in >= 90 % 0 -> 2; faint
tracks (depth < 0.25) seen once 71 % -> 55 %. The `_v2` comparison sweep
predates the band fix, so the two changes are confounded there; a
central-fold sweep with the current bands (`BROOKS_FOLD=central`, tag
`_cen`) isolates the segment choice -- see below.

The 2025 run (81 % done at this point) uses the central 30 s. The
full-corpus run will redo 2025 with the loudest-30 s rule, so all years are
consistent.

A/B with the current bands (`_cen` = central fold, `_seg` = loudest fold,
same 54 windows): next-window persistence 0.67 -> 0.76 (min 0.33 -> 0.50);
animals/window 14 -> 19; tracks seen once 141 -> 135 of 252 -> 279; in
>= 90 % of windows 0 -> 2; faint tracks (depth < 0.25) seen once 70 % ->
55 %. The band fix alone had changed none of these (the `_v2` sweep gives
the same 0.67 / 14); the segment choice is what moved them.

## 10. The 2025 run finished; reconciling with the reorganised audio (2026-09-21)

`runs/y2025`: 12,305 recorder-periods (night and day, >= 6 hour-files),
146,866 hours, 35.3 h wall time at 16 jobs, central-30 s fold. No
processing failures: the 37,512 windows the log counts as "failed" (2.1 %)
are 5-min centres with no covering file (deployment ends, restarts); in the
40 units checked, failed == uncovered exactly (`post_run_checks.py`,
scratchpad).

MBE reorganised the audio folders during the run. Because each unit is a
fresh enumeration keyed by (recorder, site, date, period) from the path,
the relaunch processed the relabelled units under their new site and left
the old-label directories behind. `reconcile_runs.py` compares the run
directory with the audio now on the volumes: 340 stale (BARLT_00008563
House Oct 9-Dec 31 x168, complete duplicates of Back9; BARLT_00008693 House
same dates x168, empty, now EastMeander; BARLT_00008240 NorthSide Jan 1-2
x4, now EastCreek), 0 missing, 0 unfinished; deleted with `--delete`.
BARLT_00008655 "is Back9 and always should be" (MBE): its Apr 8-May 3 2025
deployment folder was moved from `Audio/20250515/NorthSide/` to
`.../Back9/` on 8M2C and the 49 run directories renamed (the site appears
only in the directory name). EastMeander is a new site (first audio Oct
2025). `inventory_files.py` rebuilds the hour-file inventory
(`runs/inventory_2025_files.csv`, 148,920 files, 47.0 TB) for
`recorder_uptime.py`; totals are unchanged (32 recorders, 158,454 h), only
the site attribution moved. Site plots (Back9, NorthSide, House),
`site_streak_comparison_night` (67 streaks; between-site offset median
0.87 %, 90th pct 1.90 %; slope range median 0.11) and the Back9 hour-of-day
plots were regenerated.

The volumes are now the reference layout; the full-corpus run (2023-2026,
loudest-30 s fold) can start from this state.

## 11. Night linker: chimera tracks (2026-09-21)

A >= 2 h track on House 8662, 12 Sep 2024 (rate 19 Hz) wandered from 5.7 to
18 kHz. Cause: `night_animals.link` kept an all-time union of a track's
bands, so one broadband burst window (7 components, 6-22 kHz) let every
later 19 Hz singer join; chains also ran through legitimate harmonic bands
(6 kHz singer -> its 12 kHz harmonic -> a 12 kHz singer -> its 18 kHz
harmonic); and the reference band was assembled from touching burst bands
(11197-22050 Hz). Fix: band memory = union over the last 6 windows
(`BAND_MEMORY`); a window-animal with > 6 components (`MAX_COMP_BANDS`) is a
burst and contributes only its lowest band; among same-rate candidates the
one whose lowest band continues the track's recent lowest bands is preferred
(penalty R_TOL in the sort key); bursts and bands wider than 1.5x their lower
edge are excluded from the reference census. Test unit: worst within-track
F_fix scatter 21 % -> 5 %, widest reference 1.97x -> 1.31x, median share of
windows with F_fix 0.56 -> 0.65, 21 -> 20 tracks >= 24 windows.

Scale of the problem before the fix (Sept 2024, 8,481 tracks >= 2 h): 1
track with F_fix scatter > 15 %, 20 with a reference band wider than 1.5x.
Of the 44 % of windows without F_fix, 39 % sit at 2-3.3x the reference
(harmonic flips: fundamental missed, harmonic found), 11 % within 12 %
(near misses of the +/-100 Hz overlap rule), the rest scattered. The
fixed-window carrier was already catching the symptom.

`relink_run.py` re-linked all 498 Sept 2024 units in 56 min (0 failed):
8,481 -> 8,442 singer-nights; slopes 1.54 -> 1.67 %/degC (rate), 0.46 ->
0.48 (carrier); reference bands > 1.5x: 20 -> 0; isotropic streaks 63 -> 66,
ridge-frame v4 47 -> 44. Downstream tables rebuilt (month_summary,
temperature_effects, build_streaks default/_k55b/_v4, hourly points; old
cache kept as `streak_hourly_points_all_prelink.csv`). The corpus run
imported the new linker from unit 649 of the Dec 2024 stage onward (one unit
failed mid-edit and was re-linked); the whole corpus is to be re-linked
uniformly at the end. `NAMES` in `streak_story_figure.py` refer to the
pre-relink isotropic ids and must be re-mapped before use.

Also established: the flat track at 8.84 kHz on Back9 8525, 23 Aug 2024
(rate 23.7 -> 20.4 Hz through the night, carrier sd 1.4 %, corr(t, R)
-0.97) is a real uncoupled singer, seen on four other recorders the same
night (7.6-8.5 kHz); it is the 8.4 kHz @ 22.6 Hz streak (coupling 0.1-0.2).

## 12. Modulation floor test: 2 Hz vs 8 Hz (2026-09-22)

Prompted by Walker & Funk 2014 (Anaxipha tinnulenta 5 p/s, tinnula 8 p/s):
`run_amlo_test.py` re-ran Back9 8613, 23 Aug 2024 with `BROOKS_AM_LO=2`
(`runs/amlo_test/..._lo2`). 73 singer-windows below 8 Hz (2 % of 3,476),
53 of them at 3-4 Hz.

**First reading (wrong, kept for the record):** every slow rate divided
some faster singer's rate in the same window, so I called them bout
rhythms of known trillers. The test had not required the faster singer to
be in the same BAND. Restricted to +/-300 Hz, 62 of the 73 slow detections
have NO 8 Hz-pipeline singer in their band and window.

**What the main slow track is** (3.7-3.9 Hz @ 3.7 kHz, 35 windows,
01:55-06:30; `bout_fold_3p9Hz_0600.png`, `chirper_0600_trace.png`): a
chirping cricket. Level trace of rows 3.6-4.0 kHz at 06:00: chirps of 104
ms containing 5-6 pulses (median 6, IQR 5-6) at 47.6 Hz, repeated 3.8-4.2
times a second, 972 chirps counted in the window, for hours. In the
production (8 Hz) modulation map those rows peak at 47.6 Hz but only 3.6-4.9
dB over baseline -- below the 6 dB threshold -- because a 6-pulse chirp
gives a broad pulse-rate peak split into +/-4 Hz sidebands; the chirp rate
itself is a sharp 20-25 dB peak at 3.8 Hz. The 8 Hz run found nothing at
3.3-4.2 kHz after 00:35. Candidate: Velarifictorus micado (Japanese
burrowing cricket, introduced; SINA 3.6 kHz @ ~49 p/s at 25 degC, chirps
of several pulses), which was already a numbers-only candidate for a small
streak (3567 @ 50.4). At 02:00 the same rows show the chirp rate at only
+5 dB while the texensis trill at 33.5 Hz dominates the band: the chirper
was singing quietly under the triller.

**Conclusion:** the 8 Hz floor is biased toward trillers. Species that
chirp with few pulses per chirp (Gryllus, Velarifictorus, O. fultoni /
exclamationis, Anaxipha delicatula-type) have a weak, broadened syllable
peak and a strong chirp-rate peak, and are under-detected or seen only in
their loudest windows.

**Correction on the "fragmentation" side effect.** The 2,768 -> 3,476
singer-window rise I first blamed on the 2 Hz floor was the fold rule: the
production month_v2 run used the central 30 s, the test the loudest 30 s.
Against a proper baseline (same night, 8 Hz floor, current fold:
`..._lo8`), the 2 Hz floor changes little (3,425 -> 3,476 singer-windows,
421 -> 450 tracks, >= 2 h 39 -> 41). The objection to a lower floor is not
fragmentation but that it puts the steep 1/f region and the bout rhythms of
trillers into the syllable band; a chirp needs its own channel.

**The chirp channel** (`chirp_channel.py`, `BROOKS_CHIRP=1`; night linker
links on chirp rate when the syllable rate is unknown): 1.5-10 Hz map with
a log-rate running-median baseline, peaks >= 6 dB over >= 2 rows, chirp
rate refined on the loudest 30 s, fold at Rc (96 bins) for duty / chirp
length, within-chirp syllable rate from the row-summed 8-100 Hz spectrum
(accepted with sidebands at Rs +/- Rc >= 2 dB, or >= 5 dB with no trill
singer owning the rate in the band), chirp-rate harmonics absorbed,
spill-over merged, matches to trill-pass singers attached rather than
duplicated. Same night: +121 singer-windows (3.4 %), tracks 421 -> 447,
>= 2 h 39 -> 40. The addition is the Velarifictorus candidate: one track,
01:55-06:35 (36 windows, 5 h), 3744 Hz @ 47.6 Hz syllables, chirp rate
3.9 Hz, 89 % of its windows from the chirp channel, invisible to the trill
pass. A second chirper at Rc 3.36 (9 windows, syllable rate unread) may be
a second individual. Caveat: 8 chirp windows were linked into the
O. latipennis track (2.9 kHz @ 46 Hz) on rate + a wide band union; chirp
singers should prefer chirp tracks (to do). Chirp rate is now the second
axis Walker & Funk use (their Fig. 6); the census still favours trillers
where a chirper's syllable rate cannot be read. Next: run a full night set
with the channel on, then the frog setting (CHIRP_LO ~0.3 Hz, shorter
tracks) on a spring night.

## 12b. A machine in the census: House 8604 in cold weather (2026-09-22)

The Nov 2024 - Nov 2025 placeholder figures show vertical columns at 59 Hz
and 21 Hz in Dec-Jan at 0-5 degC. All from one recorder, BARLT_00008604 at
House, the one nearest the house and the power-line breakout (MBE). 747 of
749 cold (< 8 degC) singer-hours at 55-60 Hz are its; rate 59.2 +/- 0.2 Hz,
carrier anywhere from 0.3 to 7.3 kHz (broadband mechanical noise), Nov-Jan
only, and at that recorder the 55-60 Hz singer-hours are bimodal in
temperature (618 below 5 degC = the machine; 937 at 20-25 = the real 8.4 kHz
@ 59 Hz insect line). 59.2 Hz is the shaft speed of a two-pole induction
motor on 60 Hz mains (synchronous 3600 rpm, ~3550 rpm under load = 59.2
Hz): the heating system's compressor or blower, running when it is cold.
The 21.2 Hz column (2.9-5.0 kHz, 37 hours) is a second machine rhythm.
Handled by `artefacts.csv` (recorder, rate band, carrier band, dates,
T_max), applied in `year_hours_figure.py`; other year-scale scripts should
call the same filter. After the two rate-specific rules, 69 cold night
singer-hours remained at 8604, all at <= 2 degC and at several other
machine rates (15-19, 25-40, 55-58, 70-100 Hz); a third rule drops
everything at that recorder at <= 5 degC in Nov-Feb (no other recorder has
singer-hours there). Uptime per recorder-month (from the inventories) is
now in the per-recorder figure titles, so an empty panel reads as "not
recording" when it is. The artefact is a useful negative control: a
temperature-independent periodic source appears as a vertical column, the
one shape no insect line takes.

## 13. External test: A2O recordings beside TERN flux towers (2026-09-21/22)

Purpose: run BROOKS where an on-site thermometer exists. Australian Acoustic
Observatory sensors (BAR-LT, continuous 2-h FLAC, 22.05 kHz mono, CC-BY) at
TERN SuperSites; OzFlux L3 tower meteorology (30-min, QC-flagged; h5py).
Access, layout and site choice: `refs`-level notes in memory file
`project_a2o_tern_validation`; scripts `a2o_fetch.py`, `tern_weather.py`,
`run_a2o.py`, `a2o_quicklook.py`. Two pipeline fixes came out of the first
nights, both neutral for 44.1 kHz data: the analysed band is capped at
0.98 x the source Nyquist when resampling (resampler leakage had mirrored
strong low bands into the empty top half), and carrier rows below 300 Hz
are dropped (wind/rumble had formed 'singers' on the 0 Hz row).

Sites, 15-28 Jan (14 nights + days each, 168 2-h files, ~45 GB WAV per
site under `/Volumes/8M2C/Data/A2O/`):
- **Calperum Mallee Dry A** (SA, semi-arid; 2021; tower 230 m, Ta at 2 m).
  1,074 tracks, 97 >= 12 windows, 21 >= 2 h. One dominant line 4-6.5 kHz @
  22-45 Hz spanning 16-38 degC of tower T (1,198 windows, 75 tracks).
  Against the 2 m tower: **within-track pooled slope +1.66 %/degC (rate),
  +0.19 (carrier)**; per-track median +1.51, 85 % positive; **between-night
  slope of the same line +4.26 %/degC** (12 night medians), all-window
  +3.89; carrier +1.71 between nights; line slope d log F / d log R 0.38;
  linear R vs T through night medians reaches zero at 3.1 degC (Walker &
  Funk 2014: 2.7 +/- 2.2 for Anaxipha). So the within-hours slope is the
  same ~1.6 %/degC as Arkansas against HRRR, while the between-night slope
  is ~4 %/degC: the gap between our within-singer slope and Walker's
  values is NOT (only) grid damping; the within-night response is genuinely
  shallower than the between-night one even with a thermometer at insect
  height. Candidate causes: thermal lag of the singer/microhabitat vs 2 m
  air within a night; singers at fixed rate switching off rather than
  slowing; between-night selection (which individuals sing). This changes
  the thermometer story and needs its own analysis.
- **Cumberland Plain Dry B** (NSW woodland; 2021; tower 110 m, Ta at 30 m).
  One chirping species fills 3.9-4.5 kHz as a wall of sound (22 Hz in every
  row, 2-3 dB excess, below threshold, no individual separable); BROOKS
  finds only the sparser 6-10 kHz singers: 470 tracks, 42 >= 12 windows.
  Within-track slope +0.94 %/degC, coupling ~0. A stated limitation: a
  species that saturates a band is not resolved into singers.
- **Robson Creek Wet A** (Qld rainforest; 2023; tower 330 m, Ta at 40 m).
  609 tracks, 67 >= 12 windows; tower T only 15.8-25.2 degC over the
  fortnight (rainforest nights cool little), so few tracks span >= 1.5 degC
  (9). **Within-track pooled slope +4.17 %/degC (rate), +2.16 (carrier)**,
  per-track median +3.05 (100 % positive), coupling 0.50; the two longest
  tracks (3.0 kHz @ 13.5 Hz, 52-60 windows) run at 6.2-6.5 %/degC. So here
  the within-night slope against the tower is Walker-sized, the opposite of
  Calperum's 1.66. Preliminary (9 tracks), but it says the within-night
  slope is not a universal ~1.6: it depends on site / species / how the
  thermometer tracks the singer's air. Canopy-top air in rainforest may
  follow the understorey more faithfully at night than a 2 m mallee sensor
  follows a cricket in litter after a 40 degC day.
- Also fetched tower tables: Great Western Woodlands (Dry A 30 m from
  tower, 2024-25 season, Ta 36 m) for a later pull.

## 14. Stage 4 of the sonospecies definition: shape does not split the year streaks (2026-09-22)

Refold of the full Nov 2024 - Nov 2025 set finished: 32,130 syllable
patches (deepest window of each singer-period >= 2 h; 13 rows x 24 phase
bins), 2,014 periods with no matching part, 1 error. Three passes over them:

- **Pass 1, patch correlation** (`shape_clusters.py`: normalised patch,
  correlation maximised over all 24 phase shifts and +/-3 carrier rows,
  average linkage cut 0.5, medoid merge 0.5): every one of 43 streaks
  came out as ONE shape cluster. Not evidence of anything: the diagnostic
  (`scratchpad/shape_corr_diag.py`) gave within-streak median best
  correlation 0.62-0.81 and between-streak 0.59-0.78; e.g. the 2.6 kHz@41
  line against the 4.5 kHz@33 line scores 0.77, the same as either
  against itself. Shift-maximised correlation of a 13x24 normalised blob
  is blind to width, slope and sign. The partial-year splits seen earlier
  (7 kHz complex, 5.6 kHz ridge) were noise in this measure.
- **Pass 2, fixed frame + PCA + Gaussian mixture** (`shape_clusters2.py`:
  centre phase on the envelope centroid and rows on the energy centroid,
  clip, unit energy; shared 8-component PCA (67 % of variance); per streak
  k=1..3 by dBIC >= 50, min fraction 0.08, 2 SD separation along the line
  between means): 18 of 42 streaks split into 2-3 clusters with huge dBIC.
  Over-splitting. Clusters share T, months and recorders and differ only
  by a graded shift in glide and duty; pooled pairs of certainly different
  streaks also came out as k=3 with adjusted Rand ~0 against the streak
  label. A mixture with n in the thousands will always carve a
  non-Gaussian continuum. MBE's point stands too: different streaks need
  not have different shapes, so the pooled-pair check only confirms power
  when it passes.
- **Pass 3, look at the raw distributions** (`shape_features_figure.py`,
  `shape_samples_figure.py` with pulse centring): for the 12 largest
  streaks, glide (% of carrier over the pulse) and duty (half-max width /
  period) are UNIMODAL in every streak, against temperature they form one
  band each, and PC1 x PC2 is one cloud. Where a T-plot shows two clumps
  (3.70 kHz @ 37 Hz) they are the spring and summer seasons at the same
  glide, not two shapes. Between streaks the shapes do differ by eye (wide
  flat pulse at 6.19 kHz @ 20 Hz; long steep glide at 4.50 kHz @ 58 Hz;
  upward glide at 8.32 kHz @ 57 Hz, the only positive-glide line), so the
  patches carry information; within a streak there is no second mode.

**Conclusion for the definition.** At this resolution (300 s window, 24
phase bins, 86 Hz rows, one patch per singer-period) syllable shape gives
no grounds to split any year streak. Stage 4 is a check with a null
result, reported as such, not a confirmation that each streak is one
species. Resolution limits: duty is quantised in steps of 1/24 (visible
as histogram spikes); glides of a few tens of Hz are below one row.

**If shape is to carry weight later:** finer fold (96 bins as in the chirp
channel) on the loudest 30 s, zero-padded carrier axis, chirp grouping
(chirp rate, pulses per chirp) as hard features once `runs/month_v2_chirp`
is linked, and the strongest test of all: two simultaneous tracks on one
recorder-night inside one streak with reproducibly different shapes. The
mixture tools stay in the repo for the record; do not use their splits.

## 15. Curated examples: what the signal itself taught us (2026-09-23)

MBE: build, per streak, a set of examples we are confident in, screened by
depth and by the absence of an adjacent caller, then look at the actual
signal and derive decision criteria for "streak" and "real caller".
Scripts: `curate_examples.py` (screen + evidence pages), `example_features.py`
(audio features), `gold_sheet.py` (strict-criteria gold set, prototype
sheets), `streak_profile_gallery.py`, `streak_harmonic_audit.py`,
`window_harmonic_ghosts.py`, and the diagnostics `step_artifact_test.py`,
`two_step_lag.py`, `jitter_test.py`, `fold_direction_test.py`, `moire_demo.py`.

Screen (tables only): depth >= 0.40, >= 3 h, no same-night harmonic partner
(F/2 or F/3 at the same rate), rate CV along the track <= 0.03, |F_fix/F_c
- 1| <= 0.03, isolation <= 0.30 where a neighbour counts only if it is a
persistent track (>= 6 windows) sharing the band (+/- 150 Hz) at a nearby
rate (within 15 %) -- any band-sharer at all rejected everything in the
crowded 5-7 kHz band. Year: 34,420 night singer-periods -> 8,087 pass depth
and length -> 874 pass all screens -> 264 chosen (6 rate bins x 2, month-
spread); 8 streaks have no passing member (the harmonic ghosts and faint
high lines). Evidence row = 300 ms raw spectrogram (512/16) | fold | whole
loudest 30 s (512/128, block-max downsampled), all on one carrier axis.

Findings, each with a rule that follows:
- **Most members are faint.** Median depth 0.33; 72 % below 0.4; below ~0.4
  a fold is a 1-3 dB smudge. Shape statements rest on the readable quarter.
- **Chirpers and trillers share positions.** 4701 (4.49 kHz @ 46.5 Hz,
  "Gryllus fultoni by position") holds a field-cricket chirper (chirps of
  15-25 periods every 50-100 periods) and tree-cricket trillers at the cold
  end. The R-LOCKED singing fraction separates them (0.16-0.38 vs 0.9-1.0);
  the raw band envelope did not (other singers fill the band).
- **The rate can be a harmonic of the rhythm.** 7101 (5.5 kHz @ 11 Hz) is a
  3.4 Hz pulse train seen at its 3rd harmonic because the map starts at
  8 Hz; folding at R/3 gives one crisp 60 ms pulse at 18 dB instead of a
  smear at 6 dB. Rule: fold at R, R/2, R/3 and take the sharpest with a
  3 dB margin (8 of 264 examples flip: both 7101, all four 7001, one 8901).
- **Carrier harmonics run as parallel tracks.** Same 60 Sept nights: 2.6 %
  of window-singers are same-window ghosts, but 7.1 % of long tracks (40-48 %
  above 9 kHz) have a time-overlapping partner track at F/2 or F/3. The
  representation of one animal flips window to window (harmonic attached /
  separate / fundamental missing) and the linker threads the leftovers.
  Year audit by co-occurrence: 2,474 singer-periods (7.2 %) have a partner;
  streaks 10.6@39, 9.4@34, 19.6@21, 16.3@57, 21.6@23, 12.8@21, 14.4@36 are
  53-64 % partnered (lower bounds). The high-frequency "katydid-like,
  uncoupled" lines in the draft are mostly this. Fix: structural attachment
  of carrier harmonics in the window (same R within 1.5 %, integer F ratio,
  fold phase agreement) and merging of parallel tracks at re-link; not
  applied while the corpus run imports the pipeline.
- **The two-step syllable (8701, 6801) is real.** Same phase relation in
  every 5 s chunk, both parts at one rate within 0.05 Hz, both pause
  together; NOT an echo (lower part 240-500 Hz below, usually louder, lag
  4.6-7.5 ms varying between nights on one recorder). Why it is only
  sometimes seen is open (orientation likely).
- **Display artefacts we fell for.** (1) A 3 s spectrogram at ~380 px by
  nearest-neighbour sampling produced slanted moire stripes across syllables
  that read as rising glides; the fold (verified on a synthetic rising
  glide) and a 200 ms full-resolution view show descending syllables.
  Any spectrogram with more columns than pixels is downsampled by block
  maximum now. (2) The per-row MEDIAN baseline zeroes any row a singer
  occupies for more than half the period, cutting a long high-duty glide
  (4101, 15 ms of an 18.7 ms period) into an upper and a lower piece; a
  per-row 10th percentile instead paints every noise row +8 dB (mean vs
  low percentile of exponential noise). Baseline now = each row's MEAN
  power over the columns in which the singer's band is quietest (lowest
  20 %): noise rows come out at 0 dB, high-duty glides stay whole,
  contrasts for 4101 rose from 2-9 to 11-18 dB. The pipeline's own fold
  (row mean removal in `window_blobs.fold`/`row_phase`) has the same flaw
  and should get the same fix at the re-run.
- **Gold set** (contrast >= 6 dB at the best rate factor, isolated window,
  singing fraction >= 0.8 or >= 6 bouts): 67 examples in 27 streaks. The
  overview in carrier order shows one family of descending glides from 1.6
  to 4.8 kHz, flat pulses at specific positions, two Gryllus chirpers, the
  slow singers at their true 5-6 Hz, and two streaks (4501, 6801) with two
  forms among their gold members. Next: MBE groups the gold examples;
  prototypes from the groups; every member assigned to a prototype near its
  position or left unassigned.

## 16. Hand annotation as the test set; pipeline v2 built against it (2026-09-24)

MBE annotated 23 five-minute windows (202 boxes: 112 singers, 23 with a
hand rate, 18 harmonics, 26 non-insect, 4 "you are not picking this up") in
`annotator_app.py` (browser app: on-demand STFT at any zoom, log-frequency
option, box-drawing, band-passed playback at half speed above 10 kHz, and a
`measure` tool that reads a box's rate, fold, song structure). Scoring
(`annotation_scorecard.py`): a box is *covered* if any band of a detected
animal -- its fundamental or an attached component -- contains the box
centre or overlaps half of the narrower band; the rate agrees if a covering
animal's R is within 12 % of the hand rate or of 2x/3x it.

**What the annotations said about v1** (8 Hz floor, single loudest-30 s
fold): 91 of 112 singer boxes covered, 19 of 23 rates agree, 23 misses of
which 18 are chirpers, two-syllable callers and intermittent singers whose
chirp rate (1.7-6 Hz) is below the floor; 0 false detections on 26
non-insect boxes (the rain and crow boxes span the band and are not
informative); MBE's high-frequency "trillers" at 8.6-12.5 kHz are mostly
carrier harmonics BROOKS had correctly attached (13 boxes); one wrong
absorption (NorthSide 8051: a 12-13 kHz chirper swallowed as the 4th
carrier harmonic of a 27.6 Hz trill through the structural fallback).

**v2** (`window_blobs2.py`, a copy; `window_blobs.py` is untouched while the
corpus run imports it):
1. modulation map from 2 Hz with a running median over LOG rate as the
   baseline, 1.4 octaves wide, computed over a 1.6x wider range than kept so
   the edges are not padded. 0.7 octaves cut the broad, jittery peaks of 3-6
   Hz pulse trains (House 8634 chirpers: 3.8 dB at 0.7, 7.8 dB at 1.4);
   1.4 and 2.0 score the same.
2. a slow peak (< 8 Hz) sharing rows with a faster peak is that singer's
   BOUT rhythm only if the fast modulation switches off in the slow phase
   (Hilbert envelope of the level series' R_fast component, folded at
   R_slow: min/max < 0.4 and in phase with the level); otherwise it is a
   separate slow singer on top of a trill (the 8051 chirper).
3. on-period detection per peak (R-locked series, 0.5 s chunks, > half the
   95th percentile, 1-chunk gaps bridged, runs >= 1 s); a singer on < 75 % of
   the window has its rate refined on its longest run and is folded run by
   run, runs aligned on their envelopes by circular cross-correlation before
   averaging (a chirper's phase resets at every chirp).
4. sub-harmonic check: fold at R, R/2, R/3 over the same runs; a 1.25x
   deeper mean row depth wins (the map peak was a modulation harmonic).
   Peaks that collapse onto another peak's rate in overlapping rows are
   dropped (louder kept), else the fundamental appeared twice.
5. the structural harmonic fallback in `link_parts` requires the lone part
   to co-vary with the animal's root (r > R_MIN).

Score v1 -> v2 (same scorecard): singer boxes covered 91 -> 104 of 112;
flagged misses covered 3 -> 4 of 4; hand rates agreeing 19 -> 21 of 23;
misses 23 -> 9 (remaining: a 1 Hz repeater below the floor, a 0.9 s box,
the daytime rasp, two faint trillers at 4-5 dB excess, MidCreek's irregular
chirp complex and a faint chirper, a single chirp, and a 36 Hz box at 10.2
kHz that sits at the edge of a harmonic component); non-insect boxes with a
detection 4 -> 6 (a 99.6 Hz peak at 1.8-2.2 kHz under owl/coyote boxes --
2x the rate of the 49.8 Hz singer whose band is 2x higher, an inverted
harmonic relation the linker does not model); harmonic boxes reported as
their own singer 7 -> 10 (the gate on the fallback lets some faint
harmonics through as separate animals; all also covered via an attached
component of the true fundamental). Detected animals 296 -> 363 over the
23 windows.

Tried and not adopted: a block-wise (30 s) modulation map for sporadic
singers (max over blocks, 9 dB threshold: +64 animals, +3 false positives
on a train, no new covered singer -- the two-syllable clickers with 16
clicks in 8 s are below any Welch threshold; `am_map_blocks` kept, off);
a 1.5 Hz floor (no change); the chirp channel on top of v2 (+21 animals,
no new coverage -- v2's low floor covers its ground, so the channel can be
retired when v2 goes in).

Harness: `score_v2.py <root> [--module window_blobs2] [--set NAME=VALUE]
[--chirp] [--out singers_v2.csv]` writes per-window detections beside
`singers.csv`; `annotation_scorecard.py <root> --singers singers_v2.csv`
scores them. The annotator shows v1 (cyan) or v2 (orange) detections (`d`
cycles off / v1 / v2). Still to do at the corpus-run end: swap v2 in,
uniform re-link, merge parallel harmonic tracks, redefine streaks.

**Month test of v2** (2026-09-24, `runs/month_pipeline2`): the 498
recorder-nights of `month_v2` (2024-08-17 to 09-13) rerun with
`run_month.py --engine window_blobs2.py` (new `--engine` option on
`sweep.py`/`run_month.py`, default unchanged), 8 jobs beside the corpus
run, ~70 s per night.

**The 99.6 Hz "false positive" is not the 49.8 Hz singer's fundamental.**
Fold of the 1.8-2.2 kHz rows: depth 0.09 at 99.6 Hz, 0.02 at 49.8 Hz; its
on/off series peaks at 101-120 s while the 49.8 Hz singer peaks at 5-60 s
(r = 0.20, z = 4.1, below the link threshold). It is a weak (depth 0.17)
modulated component at the lower edge of the loud 2.3-3 kHz band, present
at neither the owl (63-66 s) nor the coyote (225-236 s) box times -- the
scorecard has no time axis, so a band overlap with a long box counts. No
"inverted harmonic" rule is warranted.

**Chirp channel, month-scale** (`compare_runs.py`): against `month_v2` the
chirp run (`month_v2_chirp`, 2026-09-22/24) shows window-animals 790k ->
1,173k and tracks >= 3 h 10.4k -> 15.1k, but that pairing confounds the
channel with the loudest-30 s fold rule (month_v2 predates it; section 12).
Against the corpus run's own nights (`runs/y2024`, same fold rule, no
channel; 164 nights in common): window-animals 310k -> 345k (+11 %; the
channel's own rows are 34.6k = 10 % of the run), tracks 47.8k -> 56.4k,
>= 1 h 11.0k -> 12.3k, >= 3 h 3,902 -> 4,347 (+11 %), >= 6 h 1,334 ->
1,484. Tracks >= 3 h with no counterpart within 300 Hz / 12 % rate: 437
added, 180 lost. Only 235 of the 4,347 long tracks (5 %) are carried by the
channel, about 1.4 per recorder-night: the channel adds many short
detections and few long singers. v2's 2 Hz floor reproduces the channel's
coverage on the annotated windows, so the month test will show whether
the same holds at scale before the channel is retired.

**Month test result** (2026-09-25; 498 nights in 13 h as four recorder
chains, `runs/tern/month_chain.py`, because the single-threaded night
linker took 3-9 min per night on busy recorders). Same linker in both runs
(last change 2026-09-22). Against the production pipeline on the 305
nights the corpus run had reached (`compare_runs.py runs/y2024
runs/month_pipeline2`): window-animals 596k -> 733k (+23 %), night tracks
91k -> 131k (+44 %), >= 1 h 20.9k -> 26.5k, >= 3 h 7,523 -> 8,645 (+15 %),
>= 6 h 2,595 -> 2,806 (+8 %); tracks >= 3 h without a counterpart within
300 Hz / 12 % rate: 2,212 added, 1,276 lost. Against the chirp run on all
498 nights: >= 3 h 15.1k -> 15.8k, >= 6 h 5.6k -> 5.4k; 3,829 added, 2,988
lost. Track quality along the track is unchanged (median R_sd 0.62 -> 0.58
Hz, F_fix_sd 100 -> 106 Hz, depth 0.32 -> 0.31).

What the added long tracks are: 425 of 2,212 are below 8 Hz (production
has none); 371 of those have no production track at 2-8x their rate in the
band and time -- new singers, not re-labelled trillers -- at 2-4 Hz, 166 in
3-5 kHz and 129 in 7-9 kHz: the field-cricket and two-syllable chirpers
the annotations said were missing. 938 of the 2,212 are above 12 kHz.

What the lost long tracks are (`lost_vs_y2024.csv`): 919 of 1,276 are
"fragmented" -- v2 has rate-compatible tracks in the band but all < 3 h,
on average 7 pieces, 45 % of the piece junctions overlapping in time
(parallel pieces, not breaks); 327 have a v2 long track at another rate
(ratios 0.6-1.5, not harmonic), 8 have nothing. 531 of 1,276 lie above 12
kHz, the harmonic-ghost region. The fragmentation is NOT a linker-input
problem: for the same bands, consecutive v2 windows have overlapping bands
100 % and rates within 1.5 % 78 % of the time, identical to production
(100 %, 77 %), same median band width (517 Hz), duplicates per window
lower (2.2 % vs 3.0 %). Two candidate explanations, unresolved: (a) v2
attaches fewer components (73 % single-component animals vs 66 %) because
the structural harmonic fallback is now gated on co-variation, which fails
for continuous singers; the fundamental's band then flips window to window
and the linker threads leftovers, most visibly above 12 kHz; (b) some
production long tracks are chimeras of neighbouring chorus members (the
3.06 kHz @ 50.5 Hz example: v2 has 13 parallel pieces at 2.86-3.27 kHz and
47-55 Hz, spacings of 2 % in rate, more than one animal's scatter) and v2's
finer parts resolve them. Deciding needs night-level ground truth or the
streak stage; proposed fix for (a): let the fallback attach a lone
harmonic part when co-variation is untestable (both parts continuous,
on-fraction >= 0.75) and block it only when the on-fractions disagree.

**Tightened after MBE looked at the night strip** (2026-09-26; annotator
night view, `annotator_app.py`): "not plausible to jump from 7 to 12 kHz in
adjacent windows". Two loopholes closed: links through a non-fundamental
component band (a 7 kHz animal whose attached 12 kHz harmonic overlapped a
lone 12 kHz detection) and the scaled-band (fundamental/harmonic) rule are
both off; links exist only between overlapping fundamentals (F_PAD 150 Hz);
and a carrier-centroid continuity term ((dF_c / 60 Hz)^2, effectively
uncapped) breaks a track at a carrier step beyond ~180 Hz -- the House 8099
relay (two individuals 55 Hz apart at the same rate trading off within a
window, level +4 dB, rate unchanged) is now two tracks. Carrier steps
inside tracks on two test nights: greedy median 49-155 Hz, 90th percentile
1.9-5.3 kHz, 42-90 tracks per night with a step > 1 kHz; global median
22-36 Hz, 99th percentile 139-148 Hz, maximum 179 Hz, none above 300 Hz.
MBE's stated priority (same day): be conservative about individual counts,
focus on the mix of call types over time -- so the linker should split
rather than merge, and this one does.

**Does it tighten the streaks?** All 498 fixed-v2 nights linked globally
(`night_global.py --all`, 2 min at 6 jobs) and `streaks_all_tracks.py`
drawn for both linkers. Tracks >= 12 windows: greedy 27,827, global 16,782;
>= 24 windows: 13,664 vs 4,306 (the global linker breaks where greedy
bridges). Concentration on the lines at EQUAL n (greedy subsampled to the
global count; 0.01-decade bins in log R x log F): fraction of tracks in
bins holding >= 5 tracks 0.68 -> 0.79 (>= 12 windows) and 0.46 -> 0.63
(>= 24); median nearest-neighbour distance 0.0016 -> 0.0012 and 0.0025 ->
0.0019 decades. So the same number of long tracks sits visibly tighter on
the streak lines under the global linker, at the price of fewer very long
tracks. Note F_fix for global tracks is the median centroid F_c (the
fixed-window carrier of night_animals is not recomputed); the streak chain
proper (month_summary -> temperature_effects -> build_streaks -> split ->
merge) has not yet been run on global tracks.

**Will the homogeneity test be clearer?** (MBE's question, 2026-09-26.)
Long tracks (>= 12 windows) of the fixed-v2 month under each linker,
brought to 25 C (bR 0.0154, bF 0.0046) and assigned to the nearest
month_v2 v6 streak in the ridge frame (perp = log F25 - c log R25, c 0.55
/ 0.25 above 10 kHz; along = log R25; d < 1.5 in units of 0.06 / 0.25);
27 streaks with >= 30 tracks in both, greedy subsampled to the global
count. Across-line sd median 0.0336 -> 0.0301 (-10 %), along-line
0.105 -> 0.096. Bimodality on the across-line coordinate (BIC of 2 vs 1
Gaussians): 11 streaks favour two components under greedy, 13 under
global; the big composite streaks sharpen (4801 6.9 kHz @ 22.9: dBIC 391
-> 412; 5101 8.0 kHz @ 23.2: 451 -> 510; 6101 5.6 kHz @ 21.2: 136 -> 190).
So tighter tracks help, but modestly. The larger gain is v2's new
variable: the share of tracks carrying a chirp rate is 74 % in 2701
(3.58 kHz @ 47.9 Hz, the Gryllus streak that section 15 found mixed with
tree-cricket trillers), 33 % in 6402 (8.5 kHz @ 59), 25 % in 6301, 12 % in
3201, and < 6 % in the tree-cricket streaks -- a separation the shape test
(section 14) never achieved. The homogeneity test should therefore run in
(across-line position, chirp rate present/value, on-fraction), not on
position alone.

Representation question raised by the run: a bout/chirp singer can appear
at its chirp rate (e.g. 13.4 kHz @ 7.7 Hz, depth 0.9, 2.6 h) and at its
syllable rate (20.9 Hz) in different windows of one night; the linker
breaks on the rate flip. 54 of the 425 slow long tracks are such
re-labellings (k = 2..8, mostly 7-8). Either the slow peak is always
recorded on the fast singer as `bout_rate` when they share rows, or the
linker matches on either rate.

**Fixes after the month test** (2026-09-25, `runs/month_pipeline2b`, rerun
of the same 498 nights): (1) the harmonic fallback attaches a lone part when
both it and the root are continuous (on-fraction >= 0.75; co-variation is
untestable there) and blocks it only when one is intermittent and they do
not co-vary; unchanged on the 23 annotated windows, and on two rerun nights
tracks 777 -> 710 with >= 3 h tracks 47 -> 52. (2) A chirper carries two
rates: `R` = syllable rate, `chirp_rate` = chirp/bout rate; a lone slow
peak is a chirp rate whose syllable rate is read inside the chirps with the
chirp channel's acceptance test (sidebands at Rs +/- Rc, or a clear peak no
row-sharing neighbour owns), else NaN. (3) `night_animals2.py` (a copy; the
production linker is untouched) matches on EITHER rate: syllable rates
within 1.5 %, or chirp rates within 6 % provided the syllable rates are
missing or within 4.5 % -- without the last clause a Gryllus chorus at
47-55 Hz was chained by chirp rate alone. `run_month.py --linker`.

Result of the rerun (`month_pipeline2b`, 2026-09-25, 13 h): on the 453
nights now shared with the production run (its August stage had advanced),
production -> v2 -> v2b: window-animals 951k -> 1,166k -> 1,161k; night
tracks 139.5k -> 200.5k -> 186.6k; >= 1 h 32.8k -> 41.4k -> 40.1k; >= 3 h
12,189 -> 14,057 -> 14,544; >= 6 h 4,427 -> 4,752 -> 5,190; production
long tracks with no counterpart 1,978 -> 1,918; added 3,561 -> 3,538. So
the fixes take 7 % of the tracks away and turn them into longer ones (+3.5 %
at 3 h, +9 % at 6 h) without changing coverage; the fragmentation relative
to production is barely touched (1,978 -> 1,918), which points at the
linker rather than the detector -- section 17.

## 17. Night linking as a fit, not a chain (2026-09-25)

MBE: linking should be a fit over the whole night rather than chaining.
Two prototypes against the greedy linker on five nights of the v2 month run
(the four "chimera" nights MBE is inspecting, plus 8051 2024-08-24).

**A fact the prototypes turned up first: the night's common rate drift is
large.** Matching every window-animal to its nearest neighbour in the next
window (150 Hz, 3 %) and integrating the median log-rate step gives
-10.6 % (NorthSide 8051, 08-24), -16.5 % (House 8634, 09-09), -23.9 %
(EastCreek 8686, 09-07), -24.8 % (House 8099, 08-28), -32.4 % (Back9 8224,
09-07) from 19:00 to 07:00, most of it before midnight (`night_global.py`,
`_drift.csv` per night). The greedy linker never sees this because it only
compares each window with the previous one (median step 0.45 %, 90 % of
steps under 1.4 %).

**Prototype 1, detrend and cluster** (`night_fit.py`): remove the drift,
cluster in (carrier, log rate) with average linkage, split clusters that
hold two members per window. Failed on its own terms: the drift was
estimated from clusters and clusters were cut by the drift (estimate came
out at +-1 % instead of -11 %), so a singer whose rate fell 55 -> 47 Hz
over the night became a ladder of nine 1-4 h clusters at 0.8 Hz spacing;
and with no time axis, sparse detections were joined across the whole
night (spans of 11 h with 4 windows). Kept for the record.

**Prototype 2, drift plus global path cover** (`night_global.py`): drift
from nearest-neighbour steps (above), detrended rate y = log R - delta(t);
candidate links between detections up to 6 windows apart with |dy| < 2 %
(+0.4 % per skipped window) and overlapping bands (fundamentals overlap:
cost 0; only a component band: +2; only after scaling a band by 2 or 3,
i.e. fundamental in one window and harmonic in the next: +3); link cost 1 +
min((dy/1 %)^2, 6) + band term + 0.6 per skipped window; a track break
costs 10. The minimum-cost path cover is a minimum-weight bipartite
matching between out-ports and in-ports with dummy start/end nodes, solved
exactly per connected component of the candidate graph (scipy
linear_sum_assignment; a single sparse LAPJV over the whole night ran >10
min, per component it is seconds). Calibration from what greedy itself
links: steps 95 % < 1.9 %; but carrier jumps between consecutive linked
windows have a 90th percentile of 1-5 kHz -- greedy tracks hop between a
fundamental and its harmonic through the band-union memory, hence the
scaling rule.

**Greedy vs global, five nights** (tracks >= 3 windows; internal scatter
of tracks >= 3 h after the same detrending):

| | greedy | global |
|---|---|---|
| tracks >= 1 h | 437 | 447 |
| tracks >= 3 h | 125 | 100 |
| tracks >= 6 h | 40 | 18 |
| detrended rate sd, median (%) | 1.5-4.2 | 1.2-1.7 |
| detrended rate sd, 90th pct (%) | 2.9-7.5 | 2.2-3.3 |

The global linker makes the same number of hour-long tracks, fewer very
long ones, and its long tracks are two to three times tighter in rate
after removing the common drift. A real animal's residual after the common
drift should be ~1-2 % (species slopes differ, so not zero); greedy's 4-7 %
90th percentiles are what chimeras look like. Not yet a verdict: the
5-hour+ tracks the global linker does not make may be real singers broken
by a rate step it refuses, and both linkers hop bands (F_sd 1-3 kHz within
tracks) through harmonic flips -- the track's carrier should be summarised
on its lowest band. The chimera windows in the annotator decide. Open
design point: a per-species drift slope (the 3 kHz chorus fell 15 % while
the common drift was 11 %) -- fit delta per rate band, or use the hourly
temperature with a per-track slope.

## 18. Post-detection flags before linking: harmonics, shadows, machines (2026-09-28)

First look at the v2 corpus run (Nov 2024 - Feb 2025 done, every sixth
recorder-period, window-level detections against the old run on the same
windows). Night windows with >= 1 caller: Nov 52 %, Dec 28 %, Jan 4 %, Feb
4-7 % -- the same collapse in both detectors. v2 finds 1.6x the old run's
window-animals in November and 1.2x in December: its gains are chorus
resolution, so they shrink as the chorus thins. Call types leave in order:
the 2-3 kHz band is 14-22 % of November night callers and 3 % by December,
when nights are almost entirely 3-5 kHz at 25-60 Hz. Chirpers are 13 % of
November night callers (20 % by day), none in the old run by construction;
a syllable rate is recovered for about half of them in the dense November
chorus and 85-90 % in winter. In January-February the chirper share rises to
~30 % and daytime rates shift to 60-130 Hz and 8-15 Hz chirp rates: the
winter residual is mostly not Orthoptera.

Three things were then checked and turned into `postfilter.py`, applied by
`night_global.link_dir` between detection and linking. Nothing is deleted:
each flag is a column, `keep` decides what the linker sees, and the dropped
rows are written back into `_global_window_animals.csv` with track -1 (the
annotator's night strip shows them as hollow grey squares, never joined).

**Harmonics reported as their own animal (`harm_of`, `harm_m`).** Above 12
kHz, v2 finds 3.3x as many November night detections as the old run; 18 %
of them have a same-window partner at half the carrier with the same rate
(old run 7 %). The rule -- carrier at 2x or 3x (8 %) of another detection's
with the same rate (3 %, or 6 % on chirp rate) -- first flagged 113 of 966
detections on 2024-11-01 NorthSide 8051, but a run of them were backwards:
a weak 2.1 kHz detection at the rate of the deep 4.2 kHz triller was taken
as its root. Band power from the parts files settles it: true harmonics
(3.8 -> 7.9 and 12.7 kHz) are 30-50 dB below their root; the 2.1 kHz
detections were 10-35 dB *weaker* than the 4.2 kHz "harmonic". So the upper
member is a harmonic only when >= 6 dB weaker (`HARM_DB`), and the lower
member is the louder animal's **shadow** (`sub_of`) when it is the weaker
one; comparable levels leave both. On that night: 69 harmonics (all at
6-16 kHz, roots at 3.7-4.2 kHz), 45 shadows (all at 2.0-2.1 kHz), and the
triller's tracks stopped fragmenting (tracks >= 3 h: 3 -> 7). Over 46
sampled November nights: 2,743 of 27,437 detections (10 %) are harmonics,
a third of everything above 8 kHz; 138 are shadows (2.5 % of the 2-3 kHz
band, so that band's December disappearance is real). 89.5 % kept.

**Fixed-rate lines (`line`).** Nearly half of the sampled January night
detections sat at 59.2 Hz, spread from 2 to 7 kHz, continuous
(on_frac_map 0.72), no drift: recorder 8604 at House, the induction-motor
line of section 12b, now found by the v2 detector as well (60 % of that
recorder's January detections), with weaker lines at 19.0 and 37.9 Hz, on
the other House recorders (8099, 8634, 8662) more faintly, and at Copse
12676 in December in the 2-3 kHz band only (MBE: the two are proximal;
the low band carries farthest). Section 12b handled this with a hand-written
`artefacts.csv`; `fixed_rate_lines.py` now finds such lines from the data,
per recorder-month: a 0.25 Hz rate bin holding >= 3 % of the recorder-
month's rated detections, >= 15x its neighbouring bins, from >= 5 recorder-
periods (animals drift 10-30 % a night, so no chorus makes a 0.25 Hz line
that lasts a month). It writes `<run_root>/fixed_rate_lines.csv`, which
`postfilter.load_lines` reads; detections within 0.2 Hz of a line of their
recorder-month are flagged. Run it on each run root before `night_global.py
--all`. Over the four finished months (819k rated detections) it finds ten
lines, all at House and Copse:

| recorder | site | month | R (Hz) | n | share of recorder-month | carrier span (kHz) |
|---|---|---|---|---|---|---|
| 8604 | House | 2024-11 | 59.23 | 1,464 | 29 % | 3.5 |
| 8604 | House | 2024-12 | 59.21 | 5,046 | 45 % | 3.9 |
| 8604 | House | 2024-12 | 18.99 | 420 | 4 % | 5.1 |
| 8604 | House | 2025-01 | 59.23 | 6,765 | 58 % | 3.9 |
| 8604 | House | 2025-01 | 37.88 | 1,087 | 9 % | 5.0 |
| 8604 | House | 2025-01 | 18.94 | 834 | 7 % | 4.5 |
| 8099 | House | 2024-12 | 59.22 | 314 | 7 % | 4.1 |
| 8099 | House | 2025-01 | 59.23 | 534 | 19 % | 4.2 |
| 12676 | Copse | 2024-12 | 59.20 | 97 | 7 % | 1.2 |
| 12676 | Copse | 2025-01 | 59.23 | 155 | 65 % | 2.9 |

No line in February; no line at any other site. The 37.9 and 18.9 Hz lines
are the same motor's rhythm read at 2/3 and 1/3 (v2's sub-harmonic check
prefers the lower rate when the fold supports it).

**Fragments (`frag`, informational only).** The isolated winter detections
(no match in the neighbouring window) are deep, brief and broken: on
Feb-2025 days 82 % of detections are isolated, versus 31 % on November
nights, and the isolated ones have depth 0.60 vs 0.33, on_frac 0.11 vs
0.22, 6 runs vs 2. But no per-window rule separates them cleanly: the best
(on_frac < 0.10 and n_runs >= 4 and on_frac_map < 0.25, or depth > 1.2)
fires on 4 % of persistent November callers and only 36 % of isolated
February ones, and in November the isolated detections have the same
feature distributions as the persistent ones (chorus dropout, not junk).
So the non-insect residual is handled by persistence -- tracks of >=
MIN_WINDOWS -- not by per-window features; `frag` is recorded for review.

**Corpus linked (2026-10-03).** The v2 run finished 2026-10-02 20:19 after
151 h (14,641 recorder-periods, no detector failures; every window v2 lacks
is an audio gap the old run also skipped). `fixed_rate_lines.py` over the
whole corpus (10.5 M rated detections) finds only the ten House/Copse lines
above -- nothing from February to November, nothing at any other site. All
nights and days were linked with `night_global.py --all` (filter on) in
about ten minutes per set, zero failures. First tally of the nights (6.63 M
window-animals): 85 % kept, 14.6 % harmonics-as-own (21 % in April-May, the
spring Gryllus chorus with strong harmonics), 0.3 % shadows, 0.2 % on a
machine line (but 67 % of January's night detections). Tracks >= 3 h per
night by month: Nov 2.3, Dec 0.6, Jan 0.05, Feb 0.1, Mar 0.3, Apr 1.5, May
2.0, Jun 2.9, Jul 3.6, Aug 7.4, Sep 8.1, Oct 4.7, Nov 1.8.

That September figure is a quarter of what the greedy two-rate linker made
on the September 2024 month (16,301 tracks >= 3 h over 498 nights vs 4,720
for the global linker on the same detections). Same filtered detections,
both linkers, four nights: the global linker makes as many or more 3-hour
tracks than the production greedy linker, so the gap is against the
two-rate greedy's long tracks, and those have band-centroid sd of 214 Hz
(median; 90th percentile 427) and fixed-window carrier sd of 109 Hz,
scaling with the carrier (55 Hz below 3 kHz, 158 Hz above 12 kHz, ~2.5 %).
The global linker's carrier term used a fixed 60 Hz scale and broke at
~180 Hz: it was shredding real animals on centroid noise, not only
refusing chimeras. The scale is now relative (`SIGMA_FC_REL` 0.03 of the
carrier, floor 60 Hz). On four test nights this raised tracks >= 3 h by
about half and the share of detections in them from 0.10-0.20 to
0.15-0.31, with carrier sd inside tracks 125-146 Hz (greedy 0.8-2.2 kHz)
and detrended rate sd unchanged at 1.1-1.4 % -- no chimera signature. 0.05
gave slightly more tracks with the rate sd starting to rise, so 0.03. The
7 -> 12 kHz hops MBE objected to remain impossible: fundamentals must
overlap regardless of this scale. The corpus was re-linked with it.

After the re-link (all 14,641 units, zero failures): night tracks >= 3 h
18,277 -> 33,123 (>= 6 h 1,635 -> 4,762; >= 1 h 155k -> 190k), day tracks
>= 3 h 2,085 -> 10,866 -- the day gain is larger because most long day
tracks sit at 6-13 kHz, where 60 Hz was under 1 % of the carrier. Tracks
>= 3 h per night by month: Nov 3.2, Dec 0.9, Jan 0.06, Feb 0.15, Mar 0.5,
Apr 2.8, May 3.9, Jun 5.1, Jul 7.7, Aug 15.5, Sep 14.6, Oct 7.7, Nov 2.4.
Inside the long tracks (night): carrier sd median 134 Hz, 2.3-2.8 % of the
carrier in every band; detrended rate sd median 1.34 % (90th pct 2.8 %);
fill 0.57. Day: 3 % and 1.37 %. Both read as single animals. Only 15 of
the 33k long night tracks are chirp-only (no syllable rate): chirpers enter
the track population through their syllable rate, so the chirp-rate share
of a streak is the variable to use, not chirp-only tracks.

**Fixed-window carrier and the year chain on v2 (2026-10-03).** The global
linker now computes `night_animals.reference_carrier` on its tracks, so every
linked window-animal carries F_ref / F_fix / ref band and each track's F_fix
is the median fixed-window carrier (F_fix_sd ~ half of the band-centroid
F_sd). The corpus was re-linked a third time with it (nights and days, zero
failures). `year_hours_figure.py --suffix _global` reads the global files and
now carries the hour's median chirp rate (Rc) and the share of its windows
with a chirp rate (rc_frac); `year_streak_inputs.py` passes both through.

The Nov 2024 - Nov 2025 chain on v2 (same parameters as the old run's:
bR 0.0154, bF 0.0046, coupling 0.55 / 0.25 above 10 kHz, _k55b -> _v4 ->
_v6), in `runs/v2/year_nov24_nov25`: 141,754 night singer-hours in 26,743
singer-periods >= 2 h (old run: 31,472 singer-nights), 27,292 day
singer-hours in 5,443 periods. 57 broad streaks -> 62 after 4 splits -> 57
after 4 merges, 8 flagged as carrier harmonics of another streak, 49
sonospecies-level streaks (the old year also had 49). The six largest are
the same lines as before within 3 %: 5.22 kHz @ 39, 6.88 @ 22, 3.64 @ 37,
8.49 @ 58, 2.61 @ 41, 6.04 @ 20. Below n ~ 1000 about half the streaks
shift or re-split relative to the old definition (24 of 49 have no old
counterpart within 6 % carrier / 8 % rate, and vice versa): the carrier is
now the fixed-window value, membership changed with the harmonic/shadow
filter, and streak ids are not stable by construction. 89 % of day
singer-periods join a night streak.

**One merge to review (MBE).** merge_streaks folded 2012 Hz @ 26.5 Hz
(n 2759 hours), 3187 @ 52 and 2920 @ 52 into 2626 @ 40 (across 4.1-4.8 %,
along-line separation 0.8-1.9 sd), giving one streak (6101) that runs from
1.8 kHz @ 20 Hz to 3.3 kHz @ 60 Hz along the coupling line. The old year
kept 1908 @ 25.8 (n 395) separate. A 3x rate range at 1.5 %/degC needs ~70
degC, so this is probably two or three species on one line that the
along-line test could not separate at this density; the overlay figure
(`streak_overlay_v6.png`) shows the band continuous in temperature colour
from 12 to 30 degC. Left as the chain produced it.

**Type mix (`type_mix.py`).** Singer-hours per streak per week, divided by
the recorder-nights recorded that week (`type_mix_v6.csv`, `_streaks.csv`,
`.png`). The succession is clean: 3.64 kHz @ 37 Hz carries Nov-Dec 2024 and
returns Jun-Jul; 5.22 @ 39 dominates Apr-Aug (peak week 11 May); 6.88 @ 22
takes over from August and dominates Sep-Nov (peak 14 Sep) together with
2.61 @ 41 and 6.04 @ 20 (both peak late Sep); 8.49 @ 58 peaks 10 Aug; the
high-frequency katydid lines (10-22 kHz) are Aug-Oct only. January to mid
March is silent (occupied weeks shown grey). Chirp share per streak (mean
rc_frac of its hours): 3.50 kHz @ 45 Hz 0.47 and 3.72 @ 45 0.42 (the
Gryllus-like chirpers, as in the month test), 6.51 @ 67 0.34, 7.19 @ 67
0.33, 6.48 @ 36 0.24, 7.56 @ 55 0.09; every other streak < 0.07. The two
chirper streaks sit beside the 3.64 @ 37 and 3.50 @ 45 trillers on the
plane and are separated by this variable, not by position.

## 19. The year as the paper: call types across thirteen months (2026-10-03)

MBE: the paper is the year; the September month was a placeholder. Rather
than re-running the month-scale analyses on the year, the set to present
was rethought around what a year can show and a month cannot. New scripts
`type_mix.py`, `year_types.py`, `year_coupling.py`; `match_sina.py` and
`sina_reference.py --match` run on the year root as they are (the v6 tables
aliased to the unsuffixed names they read; SINA tables copied from
`runs/month_v2/sina`). All on `runs/v2/year_nov24_nov25`, 49 sonospecies
streaks, 141,754 night singer-hours.

**A. Are the lines the same all year?** (`year_line_stability_v6.png`)
Two temperature-free tests per streak, by month. (i) Offset ACROSS the
coupling line, log F - c log R with c 0.55 (0.25 above 10 kHz): temperature
cannot move it if the coupling is right. For the 16 largest streaks the
monthly median stays within +/- 3 % in nearly every month over 10-11 degC of
seasonal range (largest: 7701 at 6 % in Nov-Dec 2024, 5501 at 5 %). The
lines hold position for thirteen months. (ii) Offset ALONG the line after
the streak's own year-wide fit of position on HRRR temperature: here the
monthly medians wander by 5-15 % with seasonal structure -- summer months
sit above the fit (faster at a given grid temperature), spring and autumn
below; 6901, 6101 and 5101 reach 30-45 %. So position along a line is not a
single function of grid temperature across seasons. Candidate causes, not
yet separated: a seasonal bias in HRRR night temperature at this site
(the stations showed a 2-3 degC clear-night warm bias in September; it need
not be constant), nonlinearity of rate on temperature, and mixtures of
seasonal species or generations on one line (6901 is a known mixture; 6101
is the merged line; 5101 has a Nov-Dec and a Jun-Jul population). The
first attempt -- comparing R25 and F25 by month after the pooled 1.54 %/degC
correction -- showed only that the correction slope is wrong: warm months
came out 10 % high and cold months 10 % low in both axes together. For the
paper: species = across-line position (stable); temperature = along-line
position (approximately, with the caveat above).

**B. Thermal limits per type** (`year_thermal_limits_v6.png/.csv`). The
5th percentile of each streak's singer-hour temperatures ranges from 7.8 degC
(4102, 4.33 kHz @ 60 Hz; coldest hour -1.2 degC, sings in December) through
14-15 degC for the bulk of the fauna to 19-23 degC for the warm-only types
(8901 5.22 kHz @ 39 Hz 18.8; 7701 8.49 kHz @ 58 Hz 19.9; 7802 21.7; 3501
23.1). The winter floor is therefore not one threshold: four or five types
(4102, 4601, 5001, 8801, 5101) sing on 8-14 degC nights when the rest of the
fauna is silent, which is what December looks like.

**C. Succession** (`type_mix_v6.png`, `year_phenogram_v6.png`,
`type_mix_v6_streaks.csv`). Rows ordered by peak week: 5101 (3.64 kHz @ 37
Hz) carries Nov-Dec 2024 and returns Jun-Jul; 8301 (4.13 @ 31, Gryllus
vernalis by position) peaks the week of 4 May; 8901 (5.22 @ 39) dominates
Apr-Aug, peak 11 May; 7701 (8.49 @ 58) peaks 10 Aug; 6901 (6.88 @ 22) and
6101 (2.61 @ 41) peak 14 Sep; 7101 (6.04 @ 20) peaks 28 Sep; the 10-22 kHz
katydid lines are Aug-Oct only; 4901 (4.94 @ 64) and 5201 peak in the last
November week. January to mid March: silent. Chirp share per streak (mean
rc_frac of its hours): 4301 (3.50 kHz @ 45 Hz) 0.47, 4101 (3.72 @ 45) 0.42,
8201 0.34, 8701 0.33, 7901 0.24; all others < 0.10.

**D. Day and night** (`year_day_night_v6.png/.csv`). 5,424 day
singer-periods >= 2 h, 89 % assigned to a night streak. Three high-frequency
lines are mainly diurnal: 2401 (11.1 kHz @ 71 Hz) 87 % day, 7401 (9.33 @ 55,
Bucrates malivolans by position) 84 %, 6301 (9.22 @ 33, Conocephalus strictus
by position, the species seen at the site) 72 %. Every cricket line is
< 25 % day; the tree-cricket lines < 15 %. Day periods are 3-7 degC warmer
than the same streak's night periods.

**E. Where each type sings** (`year_site_occupancy_v6.png/.csv`): fraction of
a site's recorder-nights with a singer-period of the streak. The large lines
are present at every site at 5-30 % occupancy; what differs between sites is
the occupancy, not which lines exist (EastMeander, 53 nights from October,
is the exception by sample).

**F. Coupling within singer-periods over the year**
(`year_coupling_v6.png/.csv`): 25,982 tracks with >= 12 windows and >= 0.3
Hz of rate range. Per streak (>= 40 singer-periods), median slope of log
F_fix on log R: tree-cricket candidates 0.40-0.54 for 6101 (O. latipennis),
5101 (O. texensis), 4302, 3901, 4102, but 0.07-0.08 for 4301 and 4101 --
exactly as in the month (the "O. pini by position" streak at 0.07 is not a
tree cricket; 4101 is the 42 %-chirp streak, a chirper); field crickets
0.34-0.35 (three Gryllus lines); ground crickets 0.16-0.41; trigs 0.11-0.13
for the three Anaxipha thomasi-position lines and 0.42 for 8901; katydid
candidates 0.15-0.32 (the month gave -0.04 to +0.22). Positive-share 0.62-0.97.
The subfamily gradient of the month is reproduced in its ordering but the
katydid lines are not at zero on the year.

**G. Names** (`streak_sina_matches.csv`, `sina_reference_matches.csv`,
`sina_reference_on_map.png`). By position 44 of 57 streaks have a SINA
candidate; 13 none, among them 7101 (6.04 kHz @ 20 Hz, 1,696 singer-
periods, the late-September line) and 4801 (2.78 @ 26, 542). Reference
recordings on the streak's own line (rank 1, |z| <= 2.5, streaks >= 100):
O. texensis on 5101 (z -0.40), O. latipennis on 6101 (+0.18), O. pini on
4301 (-1.06), Anaxipha thomasi on 6401 (-0.63), O. quadripunctatus twice on
4101 (+0.57, +0.74), O. nigricornis on 4102 (-0.13), Gryllus rubens on 4901
(+0.09), Eunemobius carolinus on 4901 (+0.68) and Orocharis saltator on 4501
(+0.39). The 6.9 kHz @ 22 Hz complex (6901) has six candidates within 1.3
sd (Allonemobius walkeri, A. fultoni, Cycloptilum spectabile, two
Amblycorypha, Orchelimum pulchellum) -- position cannot choose; 7601 (7.56
kHz @ 55 Hz) has eleven. Reference tests reuse the clips analysed for the
month; nothing was re-fetched.

**Not yet done for the paper:** a chimera rate for the linker against the
annotator windows; the spring non-insect check beyond the March look
(section 18); the shape metric on the wider bands; a decision on the 6101
merge; the seasonal along-line residual (A ii) needs the 2026 station
temperatures to separate thermometer from biology.

**H. The chirpers** (`year_chirpers.py`; `year_chirpers_v6.png`,
`year_chirpers_weeks_v6.png`, `.csv`; MBE asked 2026-10-03). A chirper hour
has a chirp rate in >= 50 % of its windows: 5,931 of 138,231 singer-hours
(4.3 %), chirp rate median 3.26 Hz. In the plane of chirp rate against
syllable rate they form two clusters: the Gryllus-like group at 35-50 Hz
syllables and 2-4 Hz chirps (streaks 4301, 4101, 3901: 3.2-3.7 kHz,
13-15 syllable periods per chirp period) and a 60-70 Hz @ 3-5 Hz group at
6.5-8.5 kHz (8201, with contributions from 7701 and 7601). Chirp rate is
the steeper thermometer: on the same hours the chirp rate of 4301 and 4101
rises 5.5 %/degC against HRRR and their syllable rate 2.8 %/degC; 3901 4.2
vs 4.4; 8201 2.4 vs 1.0. Dolbear's 1897 line for the snowy tree cricket is
4.9 %/degC at 25 degC; Pires and Hoy's G. firmus chirp rate rose fourfold
and syllable rate twofold over 12-30 degC -- the same ratio.

Chirp share by week splits lines by season. 8201 (6.51 kHz @ 67 Hz) trills
in May-June (share 0) and chirps in August-September (share 1.0): two
species on one line, separated by the chirp channel and by the calendar.
3901 (3.20 kHz @ 47 Hz) trilled in November 2024 (share 0), chirped in
Aug-Oct 2025 (0.4-0.8) and trilled again in November 2025. 4301 chirps
Aug-Oct at a steady share of ~0.55 with trilling hours in the same weeks --
either the chirp channel failing in the dense chorus or a triller sharing
the line; the within-streak test needs the loudest-window folds. 7901 (6.48
kHz @ 36 Hz) is 0.2-0.4 chirping through Aug-Sep.

A detection limit: 8301 (4.13 kHz @ 31 Hz, Gryllus vernalis by position,
the May line) has a chirp share of 0.001, yet spring field crickets chirp.
Their chirp rate is ~2.5 /s at 25 degC and under 2 /s on cool spring nights,
at or below the modulation map's 2 Hz floor (AM_LO), so the detector sees
them as trillers at their syllable rate. Chirp rates below 2 Hz are not
measured anywhere in the corpus; a lower floor for the chirp channel alone
would recover them and is the obvious next detector change.

**I. Streak galleries** (`streak_gallery.py`; `gallery/streak_<id>_windows.png`,
`_profiles.png`, `gallery_v6.pdf`, `gallery_index_v6.csv`; MBE asked
2026-10-04). Per streak, ten singer-periods at even quantiles of along-line
position (cold/slow to warm/fast), deep tracks, distinct recorder-nights.
For each: the track's deepest window is read from the archive, the loudest
30 s for its band found as the run does, the 1 s with the most band-locked
power drawn as a 0-22 kHz spectrogram (every panel the same axes; dB above
the row's quiet level; the band and fixed-window carrier bracketed in
cyan), and the band +/- 600 Hz folded at the track's rate over the same
30 s (48 phase bins) drawn as dB about each row's mean against ms over two
periods, so the unit stretches along the row of panels while its shape
should not change. On 5101 (O. texensis by position and reference) the ten
profiles from 13.8 to 25.3 degC, periods 36 -> 22 ms, are the same
descending glide at every temperature.

**J. The reference band was often a harmonic (2026-10-04).** The gallery page
for streak 1101 (13.4 kHz @ 49 Hz) showed every panel's fundamental band at
3.5-5 kHz while the fixed-window carrier sat at 12-14 kHz: the night linker's
reference band -- "the component present in most of the track's windows"
(`night_animals.reference_carrier`) -- was the third harmonic, because in a
dense chorus a harmonic survives in more windows than the masked fundamental.
`year_harmonic_ref.py` (F_fix / median fundamental centroid per singer-period)
put it at 2.8 % of all singer-periods but 50-94 % of those in every streak
above 10 kHz: the "katydid lines" at 10-22 kHz were mostly second and third
harmonics of 5-8 kHz crickets, which also explains their coupling of 0.15-0.32
rather than ~0 (section 19 F). First fix -- the lowest exact (lo, hi) band
present in >= 20 % of windows -- changed nothing (2.6 %): a fundamental's
edges shift by a row (86 Hz) between windows, so no single tuple of it repeats
while one harmonic tuple does. Second fix: touching bands are merged into
groups before counting, the lowest group present in >= REF_MIN_SHARE = 20 % of
the track's windows is the reference, and its most frequent tuple keeps the
band form. Re-linked (all four sets), re-chained, all analyses and galleries
re-run (`scratchpad/v2_full_redo.py`; pre-fix outputs kept in
`runs/v2/year_nov24_nov25_v6a_harmref`).

Result: harmonic-referenced singer-periods 2.8 % -> 0.09 % (24 of 26,743).
47 sonospecies-level streaks (was 49), 26,493 singer-periods assigned (was
25,976). Above 13 kHz: 2 streaks with 33 singer-periods (was 6 with 202);
10-13 kHz: 7 with 181 (was 6 with 428); 8-10 kHz unchanged in count (7 streaks)
and now anchored by 8401, 9.14 kHz @ 58 Hz, 431 singer-periods, no harmonic
references. The high-frequency fauna of the year is therefore ~10 types with a
few hundred singer-periods in total, not the ~1,300 the first definition
implied; the large cricket lines gained their members back (the 6.9 kHz @ 22
Hz complex moved to 6.54 kHz @ 21 Hz as its harmonic-referenced members
returned to the fundamental). This changes the paper's statement about
"katydid-like singers with zero coupling": the lines that remain above 10 kHz
are small and need to be re-examined one by one in the gallery before any
such claim. Streak ids changed throughout; everything in sections 18-19 that
cites an id refers to the pre-fix definition unless stated.

**K. Fixed-position galleries and the shape test (2026-10-04).** MBE: on the
along-line galleries "basically none of the streaks look like they're all
the same thing." Two confounds in that sampling: even steps along the line
walk through whatever succession a line hosts, and weak tracks whose window
is dominated by a neighbour get in. `streak_gallery.py --mode fixed` draws
the ten deepest long tracks within one narrow slice (+/- 3 %) of along-line
position instead. With that, 8801 (4.33 kHz @ 32 Hz) is one animal in all ten
panels and 5301 (O. texensis) one glide in all ten; 9101 (5.27 kHz @ 39 Hz)
and 6901 (6.54 kHz @ 21 Hz) still show two or three fold shapes at one
temperature. MBE: "much more compelling"; the fixed-position gallery is the
standard one (`gallery_fixed/`, 402 panels, `gallery_fixed_v6.pdf`).

Shape, quantitatively, twice: (1) `year_shapes.py`, glide and duty from the
quiet-level fold for up to 400 singer-periods per streak (9,097 measured):
no streak splits by a two-component test on glide (dBIC >= 10, minor >= 20 %,
2 sd apart); 9101 is the closest (dBIC 9.8, 24 % at -210 vs -75 Hz). (2)
`pulse_shapes.py`, per-pulse glide, duration, duty and bandwidth from the raw
spectrogram of the loudest 30 s at 5.8 ms frames and a 0.7 ms hop (a 512-point
frame is longer than a 9 ms pulse), median over ~1,100 pulses per
singer-period: for the ten 9101 singers of the fixed gallery the glide is a
continuum from -32 to -229 Hz that does not sort the panels as the folds did
("steep" -82 to -184, "flat" -32 to -229); durations 7-8 ms, duty 0.3. The two
looks in the folds are therefore fold quality (a rate right to a fraction of a
percent still drifts over 30 s and smears the sweep by a track-dependent
amount), not the animals. Same null as section 14. For the paper: a line is
the type, lines may hold more than one species, and neither position nor the
shape measures available separate them; the fixed-position galleries are the
transparency device, with each panel's per-pulse numbers beside the picture.
`pulse_shapes` is being run over all large streaks for the type catalogue
(pulse duration, duty, bandwidth per type) and as a track-quality flag
(pulse-to-pulse spread).

**L. Coherence before naming (2026-10-05).** MBE: "I still want to somehow
show that the streaks are coherent before giving them a name -- and I'm not
sure we've done that." We had not. `streak_coherence.py` puts four numbers on
every streak with >= 40 singer-periods (35 streaks):

1. *Co-occurrence.* Two members of one streak on the same recorder in the same
   hour share a microclimate, so one species keeps them within its individual
   spread. For every such pair, the separation along the line in %; the null
   is one singer-period's own hour-to-hour step. incoh_frac = share of pairs
   more than 8 % apart, minus the null's share (1-8 %). No thermometer, no
   shape. Calibrated on 5301 (O. texensis, reference recording on its line,
   one shape in the fixed gallery): 0.13; the bar is 0.15.
2. *Shared response.* Residuals of along-line position about the streak's
   year fit on HRRR temperature, paired between different recorders in the
   same hour: their correlation (the month's species detector). Applied only
   where the streak's slope is >= 1 %/degC; below that the residuals are noise.
3. *Descriptor splits.* Two-component test (dBIC >= 10, minor >= 20 %, 2 sd)
   on across-line offset, per-pulse duration and sweep rate (`pulse_shapes`),
   within-track coupling, and a chirp-share split (>= 20 % of members on each
   side of 0.5). Bandwidth is quantized by the row spacing and is not tested.
4. *Seasonal residual* of the along-line fit (max monthly median, %), reported
   but not a criterion: 31 % even for the coherent O. texensis line, so it is
   the thermometer or the season, not the streak.

Six of 35 pass all three criteria: 5301 (3.60 kHz @ 36 Hz, O. texensis),
5001 (2.79 @ 26), 4002 (3.99 @ 59, O. californicus by position), 8901 (3.83 @
29), 5201 (2.72 @ 22), 8301 (6.59 @ 30, Allonemobius walkeri by position).
The two largest lines fail hard: 6901 (6.54 kHz @ 21 Hz) has 47 % of its
co-occurring pairs more than 8 % apart (null 2 %) and a bimodal across-line
offset -- two parallel lines inside one streak; 9101 (5.27 @ 39) 20 % and
bimodal across; 6101 (2.60 @ 41) 28 % with a sweep-rate split (the merged
cold end); 5101 (4.83 @ 62) 42 %; 4501 and 4202 ~45-56 % on few pairs. 8801
(4.33 @ 32, Gryllus vernalis by position) fails only the co-occurrence test,
at 23 %, with the highest shared response (0.77) and no descriptor split:
either two field-cricket species on one line or a looser individual spread
than the tree crickets; undecided. 4001 (3.50 @ 45) and 4201 fail on chirp
share alone: the chirper/triller pairs of section 19 H. The 8-9 kHz lines
(8102, 8202, 8401) pass co-occurrence but have shared response ~0.15 at
slopes of ~2 %/degC: temperature-insensitive or mixed; the galleries say
one flat burst each.

For the paper: names only for the six, with the test values beside them; the
rest are listed as lines with the criterion they failed. The co-occurrence
histograms (`streak_coherence_pairs_v6.png`) are the figure: for a coherent
line the red curve lies on the grey null; for 6901 it is a flat tail to 40 %.

**M. Grids, the co-occurrence test revisited, MBE's reading (2026-10-05).**
`streak_grid.py`: per streak, rows = 12 positions along the line, columns = 5
individual singers, 0.5 s raw excerpts zoomed to the band on one shared
log-frequency axis (`grid/`, `grid_v6.pdf`); `--sync 2` adds a labeled
rate-filtered page (period-synchronous average over five periods at the
track's rate; `grid_sync_v6.pdf`). MBE: "these are GREAT"; "the only one so far
that looks like a mix is 4001".

Reconciling that with the co-occurrence test: restricted to members with a
measured pulse SNR >= 6 dB (`--clear-only`), the share of co-occurring pairs
> 8 % apart falls from 20 % to 4 % for 9101, 15 -> 5 % for 5301, 13 -> 4 % for
4001, 23 -> 14 % for 8801 -- the faint tracks (nine in ten members in the
summer chorus) carried most of the spread, a detection-quality statement, not
a species one. 6901 stays at 33 % and 6101 at 34 %. `cooccur_pairs_figure.py`
shows 6901's far-apart co-occurring pairs: in every window both members are
the same descending hook, one at 5.5-6 kHz @ 15-19 Hz and one at 6.5-7 kHz @
20-22 Hz, at the same place and moment -- two parallel lines ~10 % apart in
carrier, which the across-line bimodality had flagged. MBE: "I can forgive
merging those things in 6901." Decision: the paper's unit is a CALL TYPE --
one song form on one line; parallel sub-lines of the same form (size classes
or sibling species) are reported as structure within the type, not split. The
coherence gates that remain are the ones that catch a different song on the
same line: a chirp-share split (4001, 4201), a sweep-rate or duration split
(6101's cold end), and the grid. Co-occurrence spread and across-line
bimodality are reported per type as descriptive numbers. Pulse SNR is being
measured for every member so the clear-only test runs on the whole set.

**N. Coherence on every clear member; the verdict recast (2026-10-05, night).**
`pulse_shapes.py` was run over every member of every streak with >= 40
singer-periods (26,227 measured; 10,662 with pulse SNR >= 6 dB). On that set
the co-occurrence excess of 9101 is 0.18, not the 0.04 quoted in section M --
that figure came from the 73 pairs among the 228 clear members of the 400
deepest tracks, too small a sample. On all clear members: 6901 0.42, 9101
0.18, 6101 0.23, 8801 0.19, 5301 0.10 (O. texensis, the calibration), 4001
0.02. So weak tracks inflate the spread but do not explain it; the large
lines carry real sub-structure along the line even among clear singers.

Per MBE's reading (section M) the verdict is now in two parts. `song_form`:
'mixed song forms' only when a descriptor says a DIFFERENT song shares the
line -- the chirp-share split, or a duration AND a sweep-rate split together.
A duration split alone fired on smooth long-tailed distributions (5001's
pulse durations run from 3 to 14 ms with no gap) even with a tail guard
(wider component > 2.5x the narrower), so it is reported, not acted on; the
same guard is in `two_component`. `position`: 'single population' when the
co-occurrence excess is <= 0.15 and the shared response >= 0.3 (where the
slope is >= 1 %/degC), else 'sub-structure', else 'too few pairs'. On clear
members: 28 of 29 streaks are one song form (the exception 4001, chirper +
triller); 5 are also single populations (5301, 3801, 5201, and two more by
too-few-pairs / shared only). On all members: 30 of 35 one song form; the
mixed ones are the chirp-share pairs 4001, 4201, 8201, 8701, 9001. 6101's
sweep-rate split (section L) disappears on clear members: its odd cold end
was weak tracks. The catalog (`type_catalog_v6.csv/.md`) carries both
verdicts and the numbers behind them. For the paper: every type with 'one
song form' can be described and, with a reference recording on its line,
named; 'sub-structure' is stated as such.

**O. What one window's eight loudest animals taught about the detector
(2026-10-06, F03).** MBE, reading F03 (House 8662, 2025-09-20 23:05): "5 and 6
are picking up the same thing ... 8 is a rate harmonic of 7". Checked:

- *8 was the 2x chirp-rate harmonic of 7.* Both chirp-only, 7.1-7.9 kHz; the
  whole-window modulation spectrum of their shared rows peaks at 2.99 and
  5.98 Hz, and their on/off series co-vary (r 0.53, shift-null z 11.8). The
  detector reported them as 3.16 and 5.97 Hz, each refined in its own 30 s
  run (97-127 s and 209-239 s): a chirp rate wanders ~5 % within a window, so
  the m x R rule at R_TOL 1.5 % misses it. Across the v2 corpus (6,521 nights,
  299,797 chirp-only animal-windows) a chirp-only animal sits at 2x or 3x
  another chirp-only animal's rate in an overlapping band in 0.8 % (3 %
  tolerance), 1.7 % (6 %) or 2.4 % (10 %) of cases -- about 0.1 % of all kept
  animal-windows. No consequence for the year: chirp-only animals (no syllable
  rate) never enter the hours table (`year_hours_figure` keeps R > 0), and no
  type in the catalog sits at 2x another's chirp rate on the same carrier.
  F03 drops such a peak from its eight (CHIRP_HARM_TOL 8 % below 10 Hz) and
  draws it dashed in (B). Fix for the next relink: a `postfilter` flag for
  chirp-rate multiples at a chirp tolerance (CHIRP_TOL 6 % already exists for
  carrier harmonics of chirpers).
- *5 and 6 were two animals, and the look-alike was the measurement.* The
  5.7-6.9 kHz band holds at least four stable rates (17.4, 18.7, 19.5,
  20.6 Hz), each present in every 30 s chunk of the window: a chorus of one
  type. Their shared rows look identical in (D) because a 0.5 s chunk
  (SLOW_S) resolves rates ~2 Hz apart, so the series at 17.75 and 18.58 Hz
  over the same rows are nearly the same series (their r is 0.38, z 7.4,
  above the link threshold). The chorus is not merged because parts link only
  at rate multiples, never by co-variation alone across rates. The caption's
  old claim "a neighbor at another rate contributes nothing" was wrong for
  neighbors closer than 2 Hz; corrected.
- *A third kind of split appeared when 8 was removed:* the next animal,
  2842-3101 Hz @ 38.96 Hz, is the upper piece of the 2.5 kHz tree cricket
  (2412-2670 Hz @ 38.98): same rate to 0.05 %, fixed phase lag (-60 to -85
  deg in the chunks where both are strong), its on-run 47-93 s inside the
  other's 0-95 s. The 42.5 Hz singer owns the rows between, and the upper
  piece was too brief for its on/off series to co-vary with the lower (r
  0.07), so link_parts left it as its own animal. Corpus scale: 4.8 % of
  syllable-rate animal-windows are the briefer member of a same-rate (0.5 %)
  pair in a band within 600 Hz of the other's (300,982 of 6.33 M; 243,544 in
  tracks; median on_frac of the briefer member 0.28). An upper bound: two
  individuals of a chorus can match to 0.5 % by chance, and one animal in F03
  (5805 Hz @ 18.79) folds to two anti-phased pulses at 5.3 and 5.8 kHz that
  may be a two-part syllable or two alternating singers. This does not touch
  the 6901 sub-lines (those differ in rate, 15-19 vs 20-22 Hz). Effect on the
  year: a share of short tracks offset in carrier from their singer's line at
  the same rate -- across-line scatter and inflated individual counts, which
  the paper does not report. Not fixed for the paper; a `split_of` postfilter
  flag (same rate within 0.5 %, gap <= 600 Hz, briefer member) is the planned
  fix at the next relink. F03 applies the same rule to its choice.

**P. Paper figures F04-F08 and the two biphasic streaks (2026-10-06, afternoon).**
F04 one night as tracks (`night_tracks_figure.py`), F05 the year plane
(`year_plane_figure.py`), F06 coupling with typical example tracks
(`coupling_figure.py`; the per-type medians now exclude tracks whose rate range
exceeds 30 % of the type's R25 -- harmonic flips -- which moves 4001 from 0.07
to 0.20 and little else), F07 the grids (`streak_grid.py --inset --vector-pdf`),
F08 the type mix (`type_mix_figure.py`). F08's denominator is recorded hours,
counted from the detector's per-window files (87,935 h on 7,321 recorder-nights,
median 12.1 h per night), so the unit is singers tracked at once. MBE's design for
the site panel: each streak's peak weeks (>= 50 % of its 3-week-smoothed pooled
activity, >= 3 weeks, weeks under 25 % of the usual recording effort excluded),
then per recorder with >= 5 recorded nights in those weeks the fraction of nights
with the streak (or, `--metric hours`, its singer-hours per recorded hour); no
data in the peak weeks = blank. Rows by peak week. `--b-top` restricts the panel to
the 14 streaks named in the year panel.

Biphasic streaks 8202 (7.88 kHz @ 56 Hz) and 9101 (5.27 @ 39) have activity
peaks in Apr-Jun and Aug-Sep with a July dip. Split at 1 July 2025 (grids
`F07_grid/streak_*_spring|autumn_grid`): the song form is the same in both
seasons for both -- 8202 a flat tick train at 7-8 kHz, 9101 short down-glided
ticks at 4.3-6 kHz. Numbers: carrier at equal rate differs between seasons by
+1.2 % (8202) and -0.7 % (9101); clear-member pulse duty 0.27/0.25 and 0.22/0.24;
sweep rate 1.2/2.2 and -12.4/-13.2 Hz/ms; position at 25 degC in autumn 3-6 %
faster in rate (R25 54.7 -> 57.9; 38.8 -> 40.0). One type in two generations, or
one population with a midsummer lull, not two types. The autumn pages carry
the 6901 hooked syllables above the band in many cells: the September chorus,
not the streak.

Follow-up (same day): `season_plane_figure.py` plots the two seasons on the plane
(`F07s_biphasic_plane`). 9101: identical (-0.6 %). 8202: the fall cloud sits
+2.8 % higher in carrier at equal rate, clearly displaced but overlapping the
spring cloud over 50-60 Hz. MBE: "definitely different for 8202 but i agree
probably the same thing - maybe a bit larger." Carrier falls with body size in
crickets, so a higher carrier at equal rate would be a slightly SMALLER fall
generation; either way a size shift between generations, within one type.
Decision: no split; reported as within-type structure beside 6901's sub-lines.

**Q. F09, the identification figure (2026-10-06).** `streak_id_figure.py`
recomputes, against the v6 lines, the z of every SINA reference clip measured
at its page's dominant frequency (182 clips, 147 species; `sina_reference.py`
measurements): a line per type fitted to its singer-periods in (log R, log F),
z = perpendicular residual / sd (floor 0.03). 96 clips lie within 2.5 sd of
their nearest line; 31 of 35 types have at least one. Within 1 sd, the catalog's
candidate by published position is itself among the clips on the line for 9
types (4202, 5101, 5301, 5701, 6101, 6401, 6701, 8701, and 8801 only by
A. vernalis/E. melodius, not G. vernalis). Dense lines (6901, 7201, 8101, 8201,
9001) carry 5-6 species each within 1 sd: position alone cannot name them.
Oddity worth MBE's eye: the five O. fultoni clips (12.5-30.3 degC) fall on four
different lines (4801, 5801, 5301, 5701) -- its reference recordings do not
follow any one of our lines, so either SINA's O. fultoni clips are heterogeneous
or our 2.5-3.5 kHz tree-cricket lines are pieces of one species with a shallower
temperature path than our lines assume.

Temperature added (MBE: "shouldn't we be using temperature there somehow?"):
per type, log rate is fitted against HRRR air temperature over its singer-hours
(slopes 2-6 %/degC, residual sd 0.10-0.16 in log rate), and each clip gets
z_along = (its log rate - the type's rate at the clip's recording temperature) /
hypot(that sd, slope x 2 degC thermometer uncertainty). A clip of the right
species must be on the line (z_across) and at the right place along it
(z_along); the best line is the smallest combined z. 95 clips are on a line
across, 87 of those also along; 29 types have one. Within 1 sd on both axes the
published-position candidate is itself on the line for 5101, 5301, 5701, 6101,
6401, 6701, 8701. Clips on a line but at the wrong place for their temperature:
G. fultoni (8101, 3.5 sd slow), G. integer (6601, 3.0 sd slow), P. camellifolia
(8901), C. exsanguis (7201), and on the fast side A. griseus and C. confusa
(6901), N. triops (7801), A. insolita (6701). The O. fultoni puzzle stands: its
five clips still fall on four lines (8901, 5801, 5201, 5701) with z_tot 0.6-2.2.

**R. F10, the reference beside the field (2026-10-06).**
`reference_vs_streak_grid.py`: one row per type, the SINA clip of the catalog's
candidate species (red frame; z across/along the type's line in its title)
beside four field singers of the type from distinct recorder-nights within
1.5 degC and 15 % in rate of the clip -- of twelve clear, deep tracks analysed,
the four whose 0.5 s excerpt has the most in-band level above quiet and the
least clutter around the band (MBE: "not just crappy examples"). F10 = the five
clean cases; F10b = all 28 types whose candidate has a measured clip (none for
N. borellii 5801 and V. micado 4201). Reading of F10b, by eye:
- Same song, reference on the line: 5301 O. texensis, 6101 O. latipennis,
  6701 A. fultoni, 5101 G. rubens, 6401 N. palustris, 8701 A. rosamacula,
  4601 E. carolinus (fast flat trill), 5701 O. fultoni (the field singers
  trill continuously where the clip chirps), 6901 vs A. fultoni (z across
  +2.1: same hooked syllable with harmonic, so 6701 and 6901 are both
  A. fultoni-like; the two lines may be size classes or siblings).
- Same song form but the clip is off the line: 7201/7401/7501 vs A. thomasi
  (hooked tick at 7.2 kHz, field at 7.7-9.7 kHz: z -1.9 to -7.2 -- the field
  types are the same form at a higher carrier, candidate uncertain); 8101 and
  8201 vs E. melodius (z -9 to -10.5: the clip sits at 4.9 kHz, our lines at
  6.5-6.9 kHz; different carrier, likely different species); 8202 and 7801 vs
  N. cubensis (clip 7.0 kHz @ 56.5 Hz between the two lines, z -4.0 / -1.7);
  8102/8401 vs B. malivolans (chirped broadband katydid clip vs a continuous
  trill in the field: not the same song); 8301 vs A. walkeri (z +4.9, field
  cells faint/ambiguous); 4501 vs A. rosamacula (z +2.8, same form as 8701).
- Clip outside the line's rate range, no field match: 4801 vs G. fultoni
  (chirper at 17 Hz vs a 45 Hz line), 8801 vs G. vernalis (chirper at 10 Hz vs
  a 32 Hz line), 9001 vs A. sparsalus (16 Hz vs 65 Hz) -- the published-position
  candidates for these three rest on SINA prose rates that our clip
  measurements do not reproduce; treat as no candidate.
- Tree crickets 4001/3801 vs O. pini and 4002 vs O. californicus, 4202 vs
  O. nigricornis: same descending-hook form as the whole Oecanthus block;
  which Oecanthus is which cannot be read from the song shape alone.

**S. MBE's reading of F10b and what it changed (2026-10-06, evening).**
Verdicts: good -- 6401, 7401, 7801, 8101, 8301, 6901, 8701, 4601, 4501, 9101,
5101, 5301, 6101, 4002 (but one chirping field cell), 4001 (field heterogeneous),
3801 (a chirper among trillers), 8202 "maybe"; the reference is something else
-- 8401 and 8102 vs B. malivolans (chirped broadband katydid vs a continuous
trill), 5701 vs O. fultoni (chirper vs trillers); the reference was analysed
wrongly -- 8201 vs E. melodius, 4202 vs O. nigricornis; field cells incoherent
-- 7201 ("we have to figure that one out"), 7501; 6701 fine but examples with
overlapping songs should be avoided; no reference -- 9001, 4801, 8801.

Fixes. (1) `sina_remeasure.py`: the detector had given E. melodius' clip a band
at the lower end of its glide (4.9 kHz against a 6.4 kHz page and a visible
6-7.5 kHz song) and O. nigricornis' clipped recording one 2.2-7.7 kHz band over
its harmonic comb. Each clip's carrier is now checked against its own long-term
spectrum with the page's dominant frequency as arbiter: the spectral peak
replaces the detector's carrier only when the detector is off the page by
> 15 % and the peak is within it (25 of 182 clips; E. melodius 4.94 -> 6.46 kHz,
O. agile 7.0 -> 8.0, O. vulgare 7.6 -> 9.6, P. pulchellus 5.8 -> 7.4,
N. variegatus 4.8 -> 6.7, G. rubens' second clip 6.1 -> 4.65); otherwise the
carrier stands and the band is redrawn as the rows within 6 dB of it (64 clips
narrower, O. nigricornis 2.2-7.7 -> the fundamental). F09 and F10 read the
re-measured table. Result: E. melodius now sits ON 8201 (z across -0.04, along
-0.11), the published-position candidate confirmed by its own recording; it is
2.4 sd off 8101. O. agile and P. pulchellus move onto 8101's line.
(2) `reference_vs_streak_grid.py`: field singers must have the clip's song form
(the clip is a chirper if its band is loud < 75 % of the time; the field window
is a chirper if it carries a chirp/bout rate), windows with another kept singer
overlapping the display range are taken last, and the four shown are the best
of twelve by in-band level less clutter. 5701's row therefore shows chirping
members or none.

7201 and 7501. The grids (`F07_grid/streak_7201_grid`, `_7501_`) show the same
hooked, down-gliding tick as 6901 and 6701 with its 2x harmonic, at a higher
carrier; the F10b rows looked incoherent because chirpers with the same 20 Hz
syllable rate were admitted as examples and the hooks are faint at 8-9 kHz. On
clear members the pulse shapes agree: 6901 / 6701 / 7501 / 7201 / 7401 have
duration 6.5 / 5.8 / 5.1 / 4.5 / 5.1 ms, duty 0.14 / 0.13 / 0.10 / 0.09 / 0.10,
glide -41 / -22 / -26 / -26 / -19 Hz, bandwidth 345 Hz throughout, and differ in
f0 (6.3 / 7.25 / 7.6 / 8.5 / 7.2 kHz). The 6.3-8.9 kHz @ 19-23 Hz complex is one
song form on four or five parallel carrier lines -- size classes, sibling
species, or the carrier of a glide landing on different parts of the sweep. The
two SINA references available (A. fultoni 7.45 kHz @ 22.9 Hz, A. thomasi
7.2 kHz @ 19.9 Hz, both at 25 degC, both hooked ticks) sit in the middle of the
complex and cannot be told apart by position; the catalog's alternation between
them across 6701/6901/7201/7401/7501 is noise. For the paper: one hooked-tick
complex (Allonemobius/Anaxipha-like), sub-lines reported as structure, no
species name per line.

**T. Song form as a third coordinate, and the unplaceable chirpers (2026-10-06,
evening).** MBE on 4001's grid: chirpers and trillers mixed -- "surely we can
separate those in some way" -- "Yes. We should do that." The streak is built
from rate and carrier only; the chirp rhythm is a third axis it ignores, and in
4001 the chirpers (359) and trillers (173) sit within 1.5 % in rate and 0.7 % in
carrier. Within singer-periods the trillers never chirp and the chirpers rarely
stop (hours with every window chirping: 975 of 1,875 for chirpers, 0 of 877 for
trillers), so they are different singers, not one animal alternating; the
in-between singer-periods (a chirp rate in 10-50 % of hours) have chirping hours
twice as deep as their silent hours -- faint chirpers whose chirp channel drops
out -- but when they were put with the chirp type, 8104 became a mixture of
8102's own trill and a few broadband chirpers, so they stay with the trill type,
flagged `form = mixed`. `split_streaks_by_form.py` (_v6 -> _v7): a streak with
>= 40 firm chirpers (chirp rate in >= 50 % of hours) and >= 40 firm trillers
(< 10 %) is split; trillers keep the id, chirpers take the next id in the
hundred (4001/4003, 4201/4203, 8101/8103, 8102/8104). Grid cells of a chirp type
come from windows WITH a chirp rate (the deepest window of a chirper is often a
trilling one -- depth favors continuous modulation -- which made the first v7
chirp pages look like trills), and chirp types are drawn with 1.5 s cells so
three or four chirps fit.

MBE then pointed at cells in 7201's grid that "look completely different than
the dominant pattern, and the numbers don't match at all". They were chirpers
whose within-chirp syllable rate the detector could not measure: one track read
71, 8, 8, 98 and 21 Hz in successive hours and its night median, 21 Hz, had put
it on the 8.9 kHz @ 20 Hz line; another's 15 Hz "syllable rate" was twice its
7.9 Hz chirp rate. A test for syllable rates at integer multiples of the chirp
rate found fewer than chance (8 % vs 18 % with chirp rates shuffled), so the
mechanism is instability, not harmonics. Singer-periods whose hourly syllable
rate scatters by more than 20 % (robust sd of log R) are dropped before typing as
UNPLACEABLE: 326 of 26,743 (1.2 %), 97 % of them chirpers, concentrated in the
chirp types -- 32 of the 41 "chirpers" split from 6101 and 33 of the 56 from
7801, which therefore dissolve. The same class of failure as F03's chirp-rate
harmonic (19 O): the chirp channel returning a rate that is not a syllable rate.

Result (grids): 4003 is a coherent chirping tree cricket at 3.0-3.7 kHz, three
to four chirps a second; 8104 is a chirping katydid-like song, broadband bursts
over 7-9.5 kHz at 3.5-4.7 chirps a second, heard almost only at House -- MBE's
"completely different" B. malivolans clip may belong here rather than with
8102's trill; its syllable-rate coordinate (59 Hz) is not meaningful for a
broadband burst. `candidate_decisions.csv` holds the curated name per type with a
status (recording / position / complex / rejected / none); the catalog, F09 and
F10b read it, so the figures carry the decisions rather than the raw position
match (MBE: "the new reference grids don't reflect things you said you were
correcting").

**U. Streaks defined by song form (_v8, 2026-10-06, night).** MBE: "to the
extent we can reliably do it, we should define streaks by type as well as
carrier and rate." `track_form.py` measures each singer-period from its own
windows: share of windows with a chirp rate (rc), median chirp rate, relative
band width, robust scatter of log syllable rate across the night. Forms: trill
(rc < 0.1; 24,343), chirp (rc >= 0.5; 1,174), mixed (1,226); unplaceable when the
rate scatter exceeds 20 % (227). Band width does not separate broadband
katydid bursts from tree-cricket glides at the track level (medians 0.06-0.18
everywhere) and is kept as information only.

`streaks_by_form.py`: the chain (build -> split by T -> merge, year-pipeline
parameters) is run on the firm chirpers alone; the firm trillers KEEP the
reviewed _v6 membership and ids. Re-running the trill chain too was tried
first: average linkage re-cut 6901 into its two sub-lines and 6101 along its
cold end -- decisions MBE had already taken the other way -- because removing a
tenth of the points changes the merges, not because of song form; so the trill
geometry stays as reviewed (`--trill-rebuild` keeps the alternative). Chirp
streaks take the id of the most-overlapping _v6 streak + 50 (4051, 4251, 8151,
8152, 8252), mixed singer-periods are assigned to the nearest streak of either
form by the builder's own anisotropic metric (1,220 of 1,223) and flagged, the
unplaceable get -1. The catalog counts a type by its FIRM members (n_firm >=
40): 37 types, 32 trill and 5 chirp; 6401 and 9001 fall below 40 firm, 8701
drops to 44.

Grids of the chirp types (1.5 s cells, windows with a chirp rate): 4051 and
4251 are coherent tree-cricket chirpers (5-8 syllables per chirp, 3-4 chirps/s)
at 3.47 and 3.73 kHz -- two lines of one song form, like 6901's sub-lines; 8152
is a broadband-burst chirper at House (7-9.5 kHz, ~4/s) against 8102's trill;
8151 and 8252 are faint and heterogeneous (the hooked-tick complex and 8102's
trill show through) and are not carried as types pending MBE's reading. The
coherence test flags 4051 and 8152 as "mixed song forms" by chirp share
because their assigned mixed members chirp less; expected, not a split.

F10 field cells now come only from firm members of the clip's form (MBE: "4001
and 4002 still seem to be mixed trill/chirp" in the v7 version, whose trill
type still held the mixed singers).

Open for the next version (MBE, 2026-10-06: "that chirper vs triller thing
needs to be refined"): v8's form comes from the detector's chirp channel,
which never fires on some audible chirpers (4 of 60 cells on 4001's
firm-triller page still chirp) and returns a stable but meaningless syllable
rate for others (a 2x-chirp-rate reading; broadband bursts). The refinement is a
per-singer-period measurement from the audio -- the band envelope's bout-rhythm
contrast and duty, the test that classifies all 17 checked SINA clips correctly
-- plus the 2x check and relative bandwidth as a broadband class; then
`streaks_by_form.py` re-runs on it.

**V. v9: song form measured from the audio (2026-10-06, late).** MBE: "you can
go ahead with that." `track_form_audio.py` reads two windows of every
singer-period (the deepest and the median-depth one, 26,743 x 2 windows, ~2 h
on 10 jobs) and measures the band envelope's bout-rhythm contrast -- the
SINA-clip test -- plus duty and band share (recorded, not used: they do not
transfer to field audio) and harm2 (the detector's rate within 8 % of twice the
rhythm). The envelope is smoothed over ONE syllable period, not two: two
periods washed out the 7.9 Hz rhythm of 7201's impostor (detector rate 15 Hz).
Rule: chirp if BOTH windows have contrast > 50, trill if both < 20, mixed
otherwise. "Any window > 50" was tried first and called 9 % of O. texensis and
11 % of 6901 chirpers -- a chirping neighbor sharing the band (4051 under 5301,
8151 under 6901) contaminates one window, rarely both; with both windows 5301
drops 184 -> 29 and 6901 514 -> 67 while 4001's missed chirpers hold (117) and
4051 keeps 324 of 422. Test set of 14 known singers all correct, including the
4001 "firm triller" of 2025-09-11 (contrast 440) and the 7.9 Hz impostor
(40,000, harm2).

Result over the corpus: trill 18,545, mixed 6,629, chirp 1,569; 6,953
singer-periods change form from the chirp channel (2,222 trillers become
chirpers); 110 harm2 -> unplaceable (332 in all). `streaks_by_form.py
--audio-form --out _v9`: firm trillers keep the v6 geometry; chirp chain on
1,412 firm chirpers; 35 types with >= 40 firm members, 29 trill and 6 chirp
(4051 542 firm, 4251 122, 8151 127, 8152 96, 6951 60, 9951 85). MBE's v8 grid
review (`grid_review_v8.csv`) is the check list for the v9 pages.

**W. v9 final: syllable-locked chirps, no mixed members (2026-10-06, late
night).** MBE on the v9 chirp pages: "9951 still a mix of chirps and trills --
i thought we weren't doing this anymore?" and "a streak is either trills or
chirps - not a mix." 9951's and 8152's "chirpers" were 8102's trillers at House
under a persistent broadband burster that shares their band (the burster
itself has no track): the bout rhythm was in BOTH windows, so the
both-windows rule passed them (contrast 57-195). Added `locked_corr`: the
correlation, over the loudest 30 s, between the bout envelope and the strength
of the band's modulation at the track's own syllable rate. A chirper's
syllables exist only inside its chirps (test set +0.30 to +0.73); a triller's
syllable modulation is steady while the neighbor's bursts move the envelope
(9951, 8152 and 6951 members -0.30 to -0.79). Chirp now requires contrast > 50
in both windows AND locked_corr > 0.2 in both. Re-measured on the 1,569
chirp-labeled singer-periods: 822 remain chirpers, 747 become mixed.

Mixed singer-periods (7,376) are no longer attached to any streak: a type
holds firm members of one form only. Result: 32 types with >= 40 firm members,
29 trill and 3 chirp -- 4051 (3.47 kHz @ 45 Hz, 234), 4251 (3.71 @ 46, 226),
8151 (6.39 @ 36, 89). 8152, 9951 and 6951 dissolve. Firm members in types: 18,636; singer-hours in types 100,032 of 142,519 (70 %); unplaceable 332; mixed 7,131 unassigned. 8151's page shows no chirping singer (a neighbor's rhythm passed both tests): rejected; two chirp types remain, 4051 and 4251.

v9 trill pages against MBE's v8 review (`grid_review_v8.csv`): 4001 and 4002
no longer show chirpers (the audio form moved them to 4051/4251 or to mixed);
4201 fell below 40 firm; 3801, 5301, 6101, 6601, 6701, 6901, 8101, 8102, 8201,
8202, 8801, 8901, 9101 unchanged and good. 4501 and 4801 still carry 20-30 % of
cells with other songs at the band (bursts, hooked ticks, a 2.5 kHz triller):
these are singers whose position lands on the line and whose form is trill, so
neither coordinate removes them -- a different trilling species at the same
position, or faint tracks the detector placed on the loudest nearby line.
Both are kept as types (MBE: "keep 4501 and 4801"); 4501 carries A. rosamacula by position only, 4801 has no name. 8151's "chirpers" were a neighbor's rhythm
(rejected). The remaining shape outliers (5301 r10c4's rising glide, 7401 r1c4,
7501 r10c3) are single cells and are left as they are.

Decisions closing the day (MBE, 2026-10-06): 4501 and 4801 kept as types
despite their interlopers; 4051 and 4251 left as two chirp types ("ambivalent
... might as well leave them separate") -- the same chirping tree-cricket song
on two carrier lines 7 % apart, like 6901's sub-lines. The v9 type list for
the paper: 29 trill types and 2 chirp types with >= 40 firm members.

**X. Day singers under v9 (2026-10-06, night).** MBE: "what about day
singers?" They had been assigned to streaks by position alone
(`assign_day_to_streaks.py`). `track_form_audio.py` now reads day units
(`night_day_global_window_animals.csv`) and any id list; run on the 5,427 day
singer-periods: 3,680 trill, 18 chirp, 1,729 mixed. `assign_day_to_streaks
--audio-form` keeps a day singer only where its form matches the streak's:
3,393 of 4,965 position-assigned singer-periods keep a streak (2,772 in types
>= 40 firm). Day share falls where the daytime band held bursts or mixed
singers: 8401 0.74 -> 0.67, 6701 0.46 -> 0.37, 8102 0.39 -> 0.30, 5101 0.37 ->
0.30. Day singers never enter the chirp chain (nights only) or the grids.
Pipeline: positions, then the day audio pass, then the form-matched assignment.

Day grids (`F07_grid_v9_day/`, `streak_grid --period day --form trill`, 13
types with >= 40 day trillers; MBE: "can we make figures for the day
trillers"). The night types sing the same song by day (6701's hooked tick,
8401's and 8102's 9 kHz trills, 5101, 5301, 6901, 9101). Three types the
night catalog barely holds are coherent daytime singers: 2201 (10.1 kHz @
64 Hz; 366 day trillers vs 19 night; House and SecretGarden, Apr-Sep at
19-38 degC) and 2301 (10.6 @ 64; 132 vs 10; House, Aug-Sep) -- the same flat
10-11 kHz trill, summer katydids, N. exiliscanorus by position but its clips
sit off the line -- and 9001 (6.2 kHz @ 51 Hz; 78 vs 6; Back9, Aug-Sep), a
looser 6 kHz trill with other songs in a third of its cells. The catalog now
counts day firm members of the type's form toward the 40-member threshold,
which brings 2201, 2301, 9001 and 4201 (36 + 23) in: 36 types.

**Y. Sub-classification by fold shape (2026-10-07).** MBE: "take one of the
big streaks and then look at the data for 5m windows for a good random
subsample -- are there criteria in the shape of the fold that we can use to
subclassify the matches". `fold_subclass.py`: 600 random firm singer-periods of
a streak, one RANDOM window each (not the deepest), the loudest 30 s folded at
the refined rate on a fine grid (256-point frames, 0.73 ms hop, 2048-point
FFT; the earlier shape work folded on 11.6 ms frames, which smears a 6 ms pulse
to twice its length), dB above the row's quiet level; nine descriptors (duty,
rise, fall, asymmetry, sub-peaks, glide, curvature, bandwidth, centroid offset)
and the aligned fold image; the along-line trend (temperature) regressed out;
PCA and Gaussian mixtures with BIC and an 8 % weight floor; clusters tested
against the across-line residual (the sub-line coordinate), position along,
temperature, site, month. About 45 % of random windows give a fold of >= 6 dB.

Results. Clustering finds no discrete sub-types: the descriptor mixture picks
one component (6901, 300 windows) or two that differ in curvature with the
same position, and the fold-image clusters that do differ in across-line
residual (6901 p = 1.5e-5, 9101 p = 0.002 on 278 and 245 windows) are a
continuum cut in pieces, not separated modes. The continuum is real: after
removing the along-line trend and controlling for fold contrast (signal
quality, which inflates duty and bandwidth and does not correlate with the
across-line residual), the across-line residual correlates with pulse duty
(rho -0.18 and -0.19, p 0.003), bandwidth (-0.20, -0.18; p <= 0.004) and
centroid offset (+0.24, +0.31) in both 6901 and 9101, and with glide in 6901
(+0.21, p 0.001). Contrast-matched halves of 6901 (across < -0.015 vs >
+0.015, 96 each at 8.9 dB): duty 0.47 vs 0.42, bandwidth 474 vs 388 Hz, glide
-128 vs -56 Hz. Singers on the lower-carrier side of the line sing longer,
broader, more steeply falling pulses -- the direction a larger animal would
give -- so the sub-line structure of 6901 is a size axis within one song form,
continuous rather than two populations in this sample.

Controls at 600 windows, detrended, contrast controlled. 5301 (O. texensis,
224 windows): duty rho -0.05 (p 0.49), bandwidth -0.14 (p 0.03), glide +0.17
(p 0.01), centroid offset +0.13 (p 0.05) -- the bandwidth/glide/offset
covariation is present weakly in a one-species line too, so those three are
partly measurement coupling between the pulse's spectrum and the carrier
estimate; DUTY is the discriminator: -0.18 and -0.19 (p 0.003) in 6901 and
9101, -0.05 in 5301, -0.12 (p 0.05) in 6701 (270 windows; across-line sd only
0.020, a narrow line). Verdict: no shape criterion assigns a singer to a
sub-line; the paper can say that within 6901 and 9101 pulse duty covaries
weakly with carrier offset from the line (longer pulses below it), and that
O. texensis shows no such covariation.

**Z. F03b, and two more things one window taught (2026-10-07).** MBE wanted
"nice, rate filtered views of the 30s windows for each animal" of F03.
`window_rate_filtered_figure.py` (F03b, unnumbered while the figures are
frozen): per animal, 30 s of the window around an excerpt (50 ms mean-level
columns, on-periods as a bar), the 1.5 s excerpt centered where the animal's
rate-locked amplitude over its band peaks inside its on-periods, and the same
excerpt period-synchronously averaged over 9 periods at its refined rate
(`streak_grid.sync_average`). The selection moved into
`window_figure.choose_animals` so F03 and F03b cannot disagree (F03 re-rendered
pixel-identical). Nine periods, not five: with a 29 Hz singer 5 dB stronger
sharing the rows, k = 4 drops it from 23 to 14 dB over the median and lifts the
45 Hz animal from 18 to 36; k >= 6 lets the neighbor back in (13 periods at 45
Hz is nearly a whole number of periods at 29 Hz).

MBE: "4 doesn't look right ... the syllable repeat rate is not what is
reported". Animal 4 (3101-3445 Hz @ 44.98 Hz) sits inside animal 3's band
(3015-3876 Hz @ 29.02 Hz) and 3's rhythm is the stronger one in every shared
row (rate-locked amplitude ratio 0.16-0.83), so the raw excerpt shows 29 Hz
syllables under a 45 Hz label. The 45 Hz rhythm is real: present in all ten
30 s chunks of the band envelope, 8-10 dB below the 29 Hz peak, concentrated in
3270-3450 Hz, not a multiple or difference of 29. Two corrections in the figure:
(i) titles state when another figure animal's rhythm is the stronger one in
shared rows (4 under 3, 2 under 1, 6 under 5), decided by rate-locked
amplitude in those rows over the animal's on-periods, because band LOUDNESS
cannot decide (animal 4 ranked second of 29 by loudness; its rows are lit by
3); (ii) the filter at k = 4 above. Lesson for the ranking in F03: loudness of
the band overstates a singer under a louder one.

MBE: "something at 7.2 kHz visible in 8 that we're not picking up". It is
detector animal 13 (6891-7321 Hz @ 19.80 Hz, 19.85 Hz in the band envelope
through the whole window), 16th of 29 by loudness and so outside F03's eight;
probably another member of the 17.4-21.9 Hz chorus type.

MBE: "why are there striations visible above the chirp in the fold for 8 that
aren't visible in the data for 8?" The fold of animal 8 (chirper, 7149-7924 Hz
@ 3.16 Hz chirps) shows six stripes per period in 8.1-8.4 kHz: a 19.0 Hz
rhythm, 6 x 3.159 = 18.95. It is NOT the chirper's syllable rate (the chirp
holds ~5 syllables in ~100 ms, about 50 Hz) and not gated by the chirps: the
19 Hz rate-locked series in 8.1-8.7 kHz runs through 68 % of the window,
uncorrelated with the chirper's on/off (r = -0.04), and across the chirp
period the band's level shows six evenly spaced bumps, i.e. a continuous
trill at 19.0 Hz in 8.1-8.3 kHz (another singer, a chorus-type rate) that
happens to lie within 0.3 % of the sixth multiple of the chirp rate and so
survives a 30 s fold (drift 0.06 Hz x 30 s = 1.8 cycles, enough to blur, not to
erase). Invisible in the raw excerpt because it is 4-6 dB of ripple on a 25 dB
scale; the fold averages ~95 periods and subtracts each row's quiet level,
which is exactly what a fold is for. The detector then ABSORBED it: part 22
(blob 22, 8096-8699 Hz @ 19.06 Hz) is a rate harmonic of part 54 (blob 81,
7838-8527 Hz @ 3.14 Hz, the chirp rhythm in the chirper's upper skirt) by the
structural rule (m = 6, within 1.5 %, overlapping rows, no bout test). With a
chirp rate of 3.16 Hz, m = 2..6 at 1.5 % covers 6.2-6.4, 9.4-9.6, 12.5-12.8,
15.6-16.0 and 18.7-19.2 Hz, so a chirper's rate harmonics form a comb that
catches trillers of the chorus type by chance. For the next detector version:
apply the bout test (co-variation of the on/off series) to rate-harmonic
absorption whenever the fundamental is a chirp rate (R < BOUT_MAX_HZ) or m >= 3,
and never absorb a harmonic whose rows extend past the fundamental's. Not
applied to the corpus run (frozen); the types are unaffected (the 19 Hz
singer at 8.2 kHz would be its own animal, inside an existing chorus line or
unassigned). Recorded with the known splits in
[[project_detector_known_splits]].

**AA. F03 redesigned (2026-10-08).** MBE, after looking at the window in the
annotator (which gained the v2 corpus run, site-local clock times, a deep link
and a row-normalized display along the way): "the image in 03A is bad - i
would change this figure so (A) is higher resolution (B) is the same then for
each chosen peak we show a zoomed in region along with a modulation map for
each -- 4 doesn't stand out at 45 Hz in the rate filtered figure - it looks
like a cross-stitched version of 3". `window_figure.py` now draws (A) at
2048-point frames in 0.093 s mean-level columns, (B) unchanged, and (C) per
animal a zoom of the map around its peak (band +/- 600 Hz, R/1.6 to 2.6 R so the
absorbed 2x harmonic is in view, (B)'s scale and outlines) beside the raw and
rate-filtered 1.5 s excerpts of F03b; the fold panel and the on/off heat map
are gone (folds remain in the companion's views, on/off in 19 O).

Selection rule added: an animal is shown only if its OWN rhythm is the
strongest one in its fundamental rows on the modulation map (median over the
rows of the excess at its rate, best of +/- 2 bins, against every other
fundamental part sharing the rows at a different rate), and only if it is >= 8
dB above its rows' quiet level with its band inside the figure's 1.5-12 kHz.
Band loudness cannot decide dominance (the 45 Hz animal ranked second of 29 by
loudness; its rows are lit by the 29 Hz animal), and 0.5 s rate-locked series
cannot either (they neither separate chorus members 1 Hz apart nor hold a 3 Hz
chirp rhythm; a first attempt with them skipped the chirper and every chorus
member). The 45 Hz animal, three chorus members inside each other's rows
(18.58, 20.61, 21.90 Hz) and a faint 79 Hz singer at 10 kHz (7.7 dB) are left
to the map; the figure says how many. MBE's cross-stitch reading is right: a
period-synchronous average at 45 Hz of rows ruled by a 29 Hz train leaves a
moire of the louder train, not the weaker one's syllables, so an excerpt
cannot show a singer under a stronger rhythm and the map is its evidence
(linear-envelope spectra confirm it: 45 and 90 Hz peaks 20 dB over the floor in
3270-3450 Hz, present all window, strongest 210-270 s, absent from the 29 Hz
animal's other rows; a 40/80 Hz pair MBE spotted in 3.5-3.9 kHz is the 2.5 kHz
tree cricket leaking at its 38.98 Hz rate, following its on/off and under the
map threshold). The seven shown: 2554 @ 38.98, 2726 @ 42.52, 3193 @ 29.02,
5805 @ 18.79, 6345 @ 17.75, 6650 @ 19.80 and the chirper 7603 @ 3.16. F03b
imports the same selection and helpers and was regenerated to match.

Then (same day) MBE: "for F03 I meant a modulation graph like you showed
recently when we were looking at the 45/90 Hz caller 4 - can we use that
instead of the zoomed in modulation map". (C)'s first panel is now the
animal's modulation spectrum: the map's excess averaged over its fundamental
rows against rate on the 2-100 Hz log axis, its rate solid, the rate's
multiples dotted (where the absorbed harmonics sit), the 6 dB threshold, and
the rates of other shown animals whose bands touch the rows dashed in their
colors. One line per animal shows what the map's two dimensions hid: the
harmonic series of a trill (29.02 with 58 and 87 Hz), a chirper's comb (3.16
with every multiple to 19 Hz), and a neighbor's rate inside the same rows
(42.52 inside the 2.5 kHz tree cricket's rows). `--no-dominant` restores the
full loudness ranking.

**AB. F02 as a data overview (2026-10-08).** MBE: "a similar resolution
improvement for F02, and drop E, F and G -- i want to use that just as a data
overview - then F03 is the analysis". `night_overview_figure.py`: every panel
from 2048-point frames (21.5 Hz rows): the night in 5 s blocks (cache keyed by
frame and block length), the hour read again from the audio in 1 s columns
(it was a slice of the 10 s night blocks), the window in 0.093 s columns; all
absolute dB with a 45 dB range under the 99.5th percentile, as the record was
recorded (F03 shows the same window as dB above each row's quiet level). The 3
s, 150 ms and fold panels are gone; the window panel's title says F03 analyzes
it. The band for (B) and the window and hour still come from track 51.

**AC. The unflagged carrier harmonics above 9 kHz (2026-10-08).** MBE, from
`night_plane_points.py` (every window-animal of the F02 night on the plane,
no lines; `--kept-only` drops the flagged): "that's a lot of things that
should have been picked up as harmonics but weren't". Numbers for House 8662,
20 Sep 2025: 905 window-animals above 9 kHz with a rate, 471 flagged by the
postfilter, 434 kept. Of the kept, 18 meet the rule's own test (same-rate
animal at F/2 or F/3 within 8 %) and slipped on the 6 dB power condition; 45
have a same-rate PART there (a component of another animal); 90 more match at
5 % rate tolerance; 281 have no same-rate part near F/2 or F/3 at all. A
sample of 30 of those, tested on the modulation map: 19 (63 %) have a >= 6 dB
peak at their own rate in rows at F/2 (17) or F/3 (3), and in every case those
rows are RULED by a nearby different rate (12.86 kHz @ 19.06 over 6.4 kHz rows
ruled by 20.0; 12.54 @ 18.58 over 20.5; 11.73 @ 17.31 over 17.2): the chorus
overlaps at the fundamental and the strongest rhythm owns the rows, while at
2F the members spread out and resolve one by one, so the harmonic has no
fundamental animal to match. The harmonic images therefore carry information
the fundamental region lost (a member counted at 2F and not at F). The other
11: real high singers (9.08 kHz @ 61.2 Hz, depth 0.54, on 94 %; 9.10 kHz @
20.7 Hz, on 93 %), carrier harmonics beyond the rule's 2x/3x (15.0 kHz @ 29.01
= the 29.02 Hz tree cricket's 4th-5th), and faint brief fragments near the
floor.

No intrinsic acoustic signature separates a harmonic band from a fundamental
(MBE asked; level variance and pulse narrowing scale with harmonic number but
only by comparison with the fundamental; the ecological prior "narrow band,
slow rate, above 10 kHz" is the one single-band cue and is site-specific;
phase coherence needs F). Next-version plan: flag harmonics against the MAP
(a peak at the animal's rate in rows at F/m, m = 2..5, whether or not an
animal was built there), drop the 6 dB power condition when the rate match
is tight, and at the night level treat a track that copies another at m x F
with the same rate and slope as its image. Not applied to the frozen run; the
year's streaks above 10 kHz at chorus rates are such images (19 J).

**AD. Is the structural rate-harmonic join backed by data? (2026-10-08).**
MBE, on the methods sentence "a peak at m x R ... is joined without any
statistical test": "is this an assumption or do we have data to back it up?"
It was an assumption (Fourier: any repeating pulse puts power at every
multiple of its rate), with one known failure (19 Z). Tested on 20 windows of
the F02 night spread over 19:20-06:45 (seed 3): for each absorbed rate
harmonic, the correlation of its rate-locked on/off series (its rows, its
rate, 0.5 s chunks) with its fundamental's; two calibrations: SELF, a
fundamental's own rows locked at mR against R (one train by construction, an
upper bound since the rows share their noise), and NULL, each harmonic
against every other fundamental in its window (chance). Files in
`runs/figures_paper/explore/absorbed_harmonics_*.csv`.

Result (271 absorbed pairs, 1096 self, 8377 null). Absorbed pairs: median
0.37, quartiles 0.15-0.64. Self: median 0.34 (m = 2: 0.41; m = 3: 0.27),
quartiles 0.11-0.60. Null: median 0.00, 95th percentile 0.31. So the absorbed
harmonics co-vary with their fundamentals as much as a train's own harmonic
does, and far above chance (KS against null p ~ 1e-100). The measure is
SNR-limited: self pairs with the mR peak under 6 dB give median 0.26, over
15 dB 0.69; 40 % of all pairs fall under the null's 95th percentile and so
would fail a bout test, self pairs included (52 % of self pairs clear it, 56 %
of absorbed). The one real difference: among STRONG harmonics (>= 10 dB) of
trill fundamentals, absorbed pairs co-vary less than self pairs at matched
strength (10-15 dB: median 0.37 vs 0.62, Mann-Whitney p = 0.002), and 37 of
102 are at chance. Those 37 are almost all chorus members at 16-22 Hz, m = 2,
19:50-03:25. Nine have another fundamental in reach whose multiple fits
within 1.5 %, and for seven of them that alternative co-varies well (median
0.58): a 2R peak attributed to the wrong member of a chorus whose members'
second harmonics lie within 1 % of each other. The other 28 have no
alternative fundamental at all; their likeliest reading is the hidden-member
mechanism of 19 AC (a member at a rate within 1.5 % of the assigned one whose
fundamental rows belong to a neighbor), compounded by the series' own limit:
in rows shared by members closer than 2 Hz the 0.5 s rate-locked series is a
composite and cannot track one individual (19 O), so the test has little
power exactly where the failures sit. Chirp-rate fundamentals (R < 8 Hz): 6
of 7 absorbed harmonics at chance (-0.16 to 0.05), one at 0.81; all four m =
6 joins fail. 19 Z's case was not an exception.

Verdict: for trill fundamentals the structural join holds as well as the
test can see, with mis-attribution among chorus members as the known
weakness (harmless for types: a harmonic is not an animal, and it moves no
band); for chirp-rate fundamentals the join is usually wrong and should carry
the bout test (19 Z). The methods text should say the join is assumed from
the mathematics and checked on a sample, with these numbers, rather than
imply it was demonstrated. Suggested sentence for MBE: "A peak at mR in the
same rows is taken to be the harmonic of the syllable rhythm rather than a
second singer, since any repeating pulse produces such peaks; on a sample of
20 windows the absorbed harmonics switched on and off with their fundamentals
as closely as a train's own harmonic does, except where the fundamental was a
chirp rate (six of seven cases at chance), for which the next version applies
the co-variation test."

**AE. Can the postfilter's harmonic flag delete a real type? (2026-10-08).**
MBE: the flag uses carrier ratio, rate and level only, no bout test, so
"doesn't that risk merging things that are different call types too?" It
cannot merge types (a type is a year-long streak; the flag drops single
window-animals before tracking), but it can delete a quieter type's detections
in windows where a louder type sings at half or a third of its carrier at the
same rate. Among the 32 types with >= 40 firm members, four pairs sit inside
the flag's tolerances at 25 C: 9101 A. vernalis (5265 Hz @ 39.3) at 2x 6101
O. latipennis (2595 @ 40.5, rates 3.1 % apart); 8101 (7168 @ 40.5) at 3x 6101
(0.1 % apart); 8202 N. cubensis (7794 @ 55.4) at 3x 5801 (2587 @ 54.2, 2.3 %);
8102 (8521 @ 58.3) at 2x 4002 O. californicus (4008 @ 59.3, 1.7 %). In the
year's singer-nights the pairs do not behave as harmonic pairs: on the 20 % of
9101 nights with a 6101 singer on the same recorder, the 9101/6101 rate ratio
is 0.72-0.89 (median 0.83), never 1 (the species' rate-temperature relations
differ, so the 25 C coincidence rarely holds at the temperatures they share),
and 1 of 3682 9101 singer-nights is harmonic-like; 8102/4002: 3 of 612 (the
kept table cannot show deleted detections, so this is the surviving relation,
not the loss). The loss, from the night tables of 150 sampled recorder-nights
where both sing: 1,287 window-animals at the vernalis position (4.6-6.2 kHz,
30-50 Hz) were flagged as 2x images of a latipennis-position singer against
25,876 kept there, an UPPER bound of 4.7 % on co-nights (about 1 % of all
vernalis detections), since a real vernalis and a true latipennis harmonic are
indistinguishable by position; 898 of the 1,287 match the fundamental's rate
within 1.5 %, as a true harmonic must. Verdict: the risk is real but costs
hours, not types, and at most a few percent of them for the one large type
exposed; the mirror risk, a comparable-level harmonic of 6101 kept at the
vernalis position, is bounded the same way. A next-version flag that tests
co-variation where the harmonic is strong enough to carry a series (>= 10 dB)
would settle the ambiguous cases individually.

**AF. The window-steps methods figure, and the chirper's syllable-rate reader
(2026-10-08).** MBE: "can we construct a figure that shows with data from our
focal night each of these steps (before tracking)?" `methods_window_steps_figure.py`
-> `methods/window_steps`: (A) all 75 thresholded peaks on the 23:05 map; (B)
peak 9 split by rows into animals 12 and 13, with the per-row fold and on/off
series; (C) the 29 Hz trill's 2R/3R and the chirper's comb; (D) the 2.5 kHz
tree cricket's 2F/4F components joined by co-variation (detector r 0.30 z 6.8,
r 0.37 z 9.0) and the same-rate piece not joined (r 0.04, z 0.8); (E) the
postfilter's 7 flags on the window plane; (F) a chirper whose syllable rate
was read; (G) the 23:05 chirper whose syllable rate was not. MBE: "captions
are WAY too busy" -> one-line titles ([[feedback_short_figure_captions]]).

Two corrections MBE forced. (1) The first (F) used the 22:05 chirper with the
detector's syllable rate 20.69 Hz; MBE: "you have the syllable repeat rate
way wrong - it's definitely not 20 as you can see from the zoom ... 20 is the
triller right below the chirp; that confound is present in both cases." The
chirp-gated band-passed envelope repeats at 41.5 Hz (22:05) and 39.8 Hz
(23:05); 20.69 is the 19.8 Hz chorus triller in the rows just below. Over the
night, 37 of 49 chirpers at 7-8.5 kHz with a syllable rate carry a rate in the
chorus range 17-27 Hz and 10 carry 35-55 Hz; the latter are right. (F) now
uses 23:50 (animal 14, 7063-7924 Hz, chirps 3.23 Hz, syllables 39.5 Hz), and
both panels mark the neighbors' rates. (2) MBE: "in F the excess dB for the
SRR in the chirp is below threshold, but it's above for G." The panels had
shown the map's row-mean excess with the 6 dB peak threshold, which is not
what decides a chirper's syllable rate. `chirp_channel.syllable_rate_in_rows`
sums the level over the chirp peak's rows inside its longest on-run, takes
the Welch spectrum, measures excess over a 15 Hz running median (threshold
SYL_MIN_DB = 3), excludes the chirp rate's multiples (+/- 0.6 Hz), takes the
single maximum, and accepts it if sidebands at Rs +/- Rc exceed 2 dB or the
excess exceeds 5 dB and no peak sharing the rows has a rate within 6 %. Both
panels now draw that spectrum. 23:50: the neighbor's 19.9 Hz peak (19.8 dB)
fell inside the exclusion around 6 x 3.23 = 19.4 Hz, so the maximum became
39.53 Hz, 16 dB, sidebands +3.5 dB: accepted, partly by luck. 23:05: the
maximum was 17.5 Hz (13.8 dB), a chorus member's rate, sidebands -0.3 dB,
owned by a 17.65 Hz peak in the rows: rejected, and the syllables' own peak at
35-40 Hz (11 dB) was never tried because the reader stops after one
candidate. Next-version fix: exclude owned rates and neighbors' rates before
taking the maximum, iterate candidates, and require in-chirp concentration
(the rhythm's power inside the chirps against between them, which a leaking
triller does not show).

Does the confound reach the catalog's chirp types? Sampled 16 members of
4051 and 4251 (`explore/chirp_types_syllable_check.csv`): in every case the
chirp-gated envelope repeats at the detector's rate (36.7 -> 36.1-38.2; 43.4
-> 43.6; 49.3 -> 49.0; 44.1 -> 44.1; 46.8 -> 46.8 ...), and the rhythm's power
is 3.6 to 2000 times higher inside the chirps than between them, so these
rates are the chirpers' own syllables, not a neighbor leaking in. Most windows
also hold a same-rate animal in overlapping rows (e.g. 3359-4134 Hz @ 44.0
beside the 3749 Hz chirper at 44.1), which is either the triller type at the
same position (4001 O. pini beside 4051) or the chirper's own syllable peak
detected once more as a trill; which one is the next thing to check.

**AG. Chirps recognized directly: the time-domain reader, and the year
re-pass (2026-10-08/09, overnight, MBE away).** Checked first: the same-rate
neighbor beside a chirp-type singer IS the chirper in 19 of 25 cases (its
rhythm 4-1,258 times stronger inside the chirper's chirps than between them;
the 6 others are continuous trillers in adjacent, non-overlapping bands). Of
the 25 duplicates, 18 never became singer-periods, 3 were mixed/unplaceable,
3 joined the chirp type (a chirper counted twice within its own type), 1 is a
real latipennis. The trill types at the position are clean: 4001 shares a
same-rate chirp-type singer on 1 of 104 recorder-nights, 3801/4201/5301 on
0-1. The bout test (`is_bout_rhythm`: the fast rhythm's envelope folded at
the chirp rate must be deep and in phase) fails for these chirpers because
the syllable peak is weak and broadened, so slow and fast peaks become two
animals.

Two spectral fixes failed. A concentration test (power at the candidate rate
inside the chirps vs between, on the row-summed level) let the chorus rate
through (ratio ~1.2-2.4 for the neighbor, < 1 for the true syllables: gating
a dB level series with the chirp envelope does not separate them), and would
have merged real trillers as twins; restricting the sum to rows the chirp
rate rules did not help because the level sum is dominated by the loudest
rows. Both are left in the code behind `BROOKS_SYL_V2` (off) only as a record.

MBE: "can we not recognize chirps directly somehow and ID the syllable rate
within". `chirp_channel.syllable_rate_from_chirps`: band-pass the audio to
the chirper's band, envelope at 4 kHz, the CHIRPS as events where the 40 ms
smoothing clears a threshold between its quiet (20th pct) and loud (97th pct)
levels for >= 30 ms (gaps < 20 ms bridged); inside each chirp the pulses are
the peaks of the 3 ms smoothing (prominence >= 25 % of the chirp's range, <=
120/s); syllable rate = 1 / median inter-pulse interval; quality = the
intervals' IQR as % of the median, accepted when <= 35 % with >= 20 intervals
over >= 5 chirps. A continuous neighbor adds no bursts and cannot win. Focal
night, 7-8.5 kHz chirpers: 75 of 75 read, median 77 Hz (quartiles 74-79), 7
pulses per 104 ms chirp at 3.0 chirps/s, the rate sliding 80 -> 72 Hz as the
night cooled; the corpus run's 20 Hz was the chorus and its 39.5 Hz half the
rate. Chirp types: where the spread is small the time-domain rate equals the
recorded one within a few % (4251: 44.5/44.1, 40.9/41.6, 46.1/46.8, 46.6/46.1,
48.9/47.6, 41.3/41.9; 4051: 39.7/36.7, 38.5/38.7, 44.1/43.4, 50.1/49.3); where
the spread is 50-85 % the reading is inflated (pulses split) and the gate
rejects it. Figure `methods/window_steps` F-G now show the reader on the
23:05 chirper and its rate through the night.

The year re-pass (`chirp_repass.py`, MBE: "launch it on whole year if this
can be done without disrupting ... the rest of the v2 year run"): every
window with a chirp-fundamental animal (chirp_rate set, bout_rate NaN) read
again, the time-domain reader on the animal's band over its longest on-run
(capped at 60 s), the twin rule (an animal at the new rate within 3 % whose
band overlaps, with no chirp rate) recorded; results in a SIDECAR per unit,
`night_night_global_window_animals_sylv2.csv`, originals untouched; 16
workers, 6.6 h, 6,521 units (3 band-edge failures fixed and rerun). Pooled in
`year_nov24_nov25/chirp_repass_sylv2.parquet`: 672,767 chirp-fundamental
animals in 285,945 windows (45 % of all windows). Spectral reader had read
372,970 (55 %); the time-domain reader reads 253,986 (38 %): 136,280 both,
236,690 lost (mostly rejected by the spread gate: 383,827 have intervals
with IQR > 35 %), 117,706 gained. Where both read they agree within 5 % in
only 18 %; the rest is the neighbor-rate problem in the old reading, as on
the focal night. Time-domain rates: quantiles 10/25/50/75/90 = 51 / 69 / 80
/ 87 / 96 Hz; 7 pulses per 96 ms chirp median. Twins marked: 42,429.
Validated on one night and the two chirp types only; the 250k readings are
otherwise untested. NOTHING downstream consumes the sidecar yet: the corpus
tables, tracks, streaks and figures are as they were. Feeding it into a
relink would give chirpers positions on the plane and change the catalog,
which is a decision for after the paper.

**AH. The v3 relink (2026-10-09).** MBE: "I think we should link them - the
previous generation of chirp data had a real flaw." Done as a new tree so
the v2 run (the paper's `_v9` numbers) stays as it is: `make_v3_tree.py`
builds `runs/v3/y2024|y2025` with every unit a real directory of symlinks to
the v2 files (6.4 M links, 5 min), night units leaving out the three global
link outputs; `night_global.apply_sylv2_sidecar` (on with
`BROOKS_SYLV2_LINK=1`) patches the window animals before linking: every
chirp-fundamental animal takes the sidecar's time-domain syllable rate (NaN
where the reader rejected it, so chirp-only and untracked), and a twin is
folded into its chirper (parts reassigned, row dropped, band widened). Relink
of all nights: 6,521 units in 8 min on 14 workers, 0 failures, the same set
of nights the v2 run had linked. Focal night: 3,062 window-animals (3 twins
folded), 228 tracks; the 8 kHz chirpers go from 24 track fragments at the
chorus's 20 Hz (49 of 75 with a rate) to 11 tracks at 71-79 Hz (62 of 75).
Then `year_pipeline.py --from hours` with `BROOKS_RUN_TREE=v3`, writing
`runs/v3/year_nov24_nov25` and `runs/v3/year_figs`; the chain's intermediate
suffixes (_k55b, _v4, _v6, _v9) keep their names inside the new directory.
Day units are unchanged (no chirp re-pass for days). Open question kept for
the catalog comparison: whether the quality gate's 10-30 % acceptance at 1-6
kHz hides real chirpers (frogs and bursts dominate there) and whether the
time-domain rates' few-percent jitter fragments chirper tracks at the
linker's 2 % rate tolerance.


### 19 AI. The chirpers' year on a second plane (2026-10-11)

The year chain places a singer by its syllable rate, so a v3 chirper whose syllables the time-domain reader refused (a
chirp rate, no syllable rate) is linked into a track but never reaches the streaks: 3,014 night tracks of >= 2 h, 4.2
percent of all long tracks, 10,460 track-hours, July to October at 3-4 and 6-7 kHz. (In v2 the spectral reader gave
almost every chirper a syllable rate, usually a neighbor's, so they sat on the main plane misplaced; only 68 long tracks
were chirp-only.) On carrier against CHIRP rate those tracks form lines as clean as the trillers' (explore/chirp_only_tracks_v3),
and chirp rate follows temperature along each line.

`chirp_year.py` builds chirp singer-hours (per track-hour: median F_fix, median chirp rate, HRRR temperature), writes
them in the main chain's format with R := chirp rate into `<year>/chirp/`, fits the within-track temperature slopes
(chirp rate +3.2 %/C, carrier +0.9 %/C over 2,879 hour pairs) and runs build_streaks (coupling 0.35, along 1.0,
across 0.10 in log units) and merge_streaks. First try with across 0.06 cut the thick 3 kHz band into three strips;
0.10 gives 8 streaks with >= 40 singer-nights, numbered C1-C8 by carrier:

| id | F25 (Hz) | chirp rate at 25 C | n | slope %/C |
|---|---|---|---|---|
| C1 | 3431 | 10.3 | 95 | +1.1 (cold-season, Dec-Mar) |
| C2 | 3562 | 3.5 | 1244 | +5.8 |
| C3 | 3703 | 9.0 | 183 | +3.5 (cold-season) |
| C4 | 5178 | 7.2 | 41 | +0.2 |
| C5 | 5656 | 5.6 | 171 | +4.7 |
| C6 | 6325 | 3.0 | 796 | +6.3 |
| C7 | 6528 | 6.4 | 83 | +4.6 |
| C8 | 8526 | 4.3 | 80 | +4.2 (the focal night's 3 Hz chirper, Fig 5G) |

Overlap with the main catalog's chirp types (syllable rate read, id + 50): C2 is the same population as 4251/4951
(3.5-3.7 kHz, chirps 3.4-3.6 Hz), C6 as 7151/8751/9951 (6.5-7 kHz, 2.8-3.2 Hz), C8 as 7351 (8.2 kHz, 3.7 Hz). The
same animals are split between the planes by whether the reader could read their syllables in that window. Open
decision for MBE: define chirp types on the chirp plane for ALL chirpers (rc_frac >= 0.5) with the syllable rate as an
attribute, which would retire the +50 types. SINA reference clips carry no chirp rate yet (sina measurements would need
the chirp channel), so Figure 10 has no chirp panel. The detector caps chirp rate at 8 Hz (BOUT_MAX_HZ): the hard edge
at 8 Hz in the plane. type_mix_figure takes `--chirp-root` (F09_type_mix_v3). METHODS draft: "chirp-only singers do
not enter the tracks" is wrong for v3 (they are tracked, not placed); fix when the chirp plane is adopted.


### 19 AJ. The chirp-rate flip, the re-pass, and the two planes (2026-10-11)

MBE: type 7151 (6.8 kHz, R 73) "seems like two things". A 2-component mixture on (log C, log R, log F) split it, but the
excerpts of both groups showed the same song (100 ms chirps of 6-8 syllables at 70-85 Hz, 6.5-8 kHz); within a track
the syllable rate held to 3 % while the chirp rate wandered 11 %, and in 13 of 46 tracks the chirp rate's upper mode
was 1.7-2x its lower. Diagnosis on BARLT_00008634_House_20251119 track 196: at 00:55 the map's peak was the chirp
rhythm's third harmonic near 20 Hz; `subharmonic_rate` divided once (m = 3) to 6.7 Hz and stopped, although the fold
at 2.24 Hz is three times deeper. The check was single-step. Not every fast reading is a flip: in some windows the
fold at 5 Hz really is the rhythm (two equal pulses per 400 ms; H1/H2 of the fold at 2.5 Hz < 1), plausibly two
individuals interleaving. The genuinely different thing inside 7151 is a Copse-only August 2025 cluster (41 h, 15
tracks, 5.1-6.6 kHz, R 42-61 at 24-25 C where the main group sings 72-87 Hz): faint 2-3 syllable chirps, swept in
because it lies on the downward extension of 7151's carrier-rate line; the temperature split needs > 15 singer-nights
to accept a component. The crude time-domain chirp count (n_chirps / run_s) was tried as a flip test and rejected:
its ratio to the map rate is broad (pauses, split chirps), with only a bump at 0.5.

Fix: `window_blobs2.subharmonic_rate` iterates (SUBHARM_ITER 3: divide again while a sub-multiple's fold is still
SUBHARM_GAIN deeper). `chirp_rate_repass.py` re-reads every window-animal with a chirp rate (6,520 v3 nights,
672,360 chirp animals, 16 workers, 3.7 h): the window's STFT, the band rows, on-runs at the old rate, the iterated
check, a refinement; sidecar `night_night_global_window_animals_crv2.csv` (t, animal, C_old, C_new, m, depths);
25,343 chirp rates changed (3.8 %). `night_global.apply_sylv2_sidecar` applies it before linking (BROOKS_SYLV2_LINK=1).
Relink of the v3 nights (the days were not relinked: the re-pass was nights only; a forced pipeline relink would have
redone all 14,641 units, so the pipeline was stopped after the night link and restarted from the hours stage) and
the year chain: 64 streaks, 35 with >= 40 singer-nights (31 trill, 4 chirp); 138,096 night singer-hours.

Two planes (MBE, 2026-10-11): Figure 7 is the trillers on carrier against syllable rate (`year_plane_figure.py
--trillers-only`: tracks with a chirp rate in >= 50 % of their windows and the chirp-form streaks dropped; 134,088
hours, 31 types numbered 1-31 by carrier, `paper_ids.py` numbers trill types only). Figure 7b is ALL chirpers on
carrier against chirp rate (`chirp_year.py`: a chirper = a track >= 2 h with a chirp rate in >= 50 % of its kept
windows, with or without a syllable rate; the syllable rate carried as `Rs`), clustered with the main chain with
R := chirp rate; types C1..Cn by carrier; panels (A) by temperature, (B) by type, (C) carrier against syllable rate
for the hours that have one, (D) syllable rate against chirp rate with the syllables-per-chirp diagonals; a second
page with chirp rate against temperature per type and the Theil-Sen slope. The trial on the pre-re-pass data found
7,897 chirping tracks and 33,178 chirp singer-hours, far more than the 3,014 chirp-only tracks of 19 AI: most
chirpers had a syllable rate read in a few windows, too few to enter the hours table (year_hours_figure requires
min_win windows with R), so they fell between the two planes. The exploratory F-C / F-R / C-R plots by type
(explore/chirp_FCR_by_type) showed which rhythm each chirp type controls: the 3.5 kHz pair's chirp rate is
temperature-locked (+5 to 6 %/C) while the syllable rate barely moves; the 7-8 kHz group's syllable rate is the clean
coordinate and its chirp rate is noisy and flip-prone. Syllables per chirp period are not a constant of a type (15
to 10 across the night for the 3.5 kHz pair): closer to a fixed chirp duration with the gap shrinking as it warms.


### 19 AK. The v3 partition: long lines cut into temperature segments; along-line tolerance 0.8 (2026-10-12)

Carrying the v2 names to v3 (carry_decisions.py: singer-nights matched on the same recorder-night by position, a
target type named only when >= 60 % of it comes from one v2 type that sends >= 50 % of itself there) carried 11 of 35
v3 types. The rest failed because the v3 streak chain, run with the same parameters as v2, cut the long tree-cricket
lines into temperature segments: v2 latipennis 6101 (1,639 singer-nights) became v3 5501 (2.0 kHz @ 28 Hz, nights at
15-19 C), 5601 (2.6 kHz @ 42 Hz, 18-25 C) and 5801 (3.4 kHz @ 64 Hz, 21-28 C); vernalis 9101 became 7701 and 7501;
texensis 5301 lost its cold end to 5301(v3). The relink changed the density of singer-nights along the lines and
average linkage at along-tol 0.6 (log-rate units; the latipennis line spans ln(78/20) = 1.36) broke where the density
thinned. Test of the stage-one tolerance on v3 (split and merge unchanged): 0.6 -> 36 types, latipennis in 3,
vernalis in 2; 0.8 -> 28 types, latipennis, vernalis and texensis whole (100 / 97 / 99 %); 1.0 -> 31 types,
latipennis 78/21 again (a merge with a neighbor pulls the linkage). The 6-7 kHz hooked-tick complex (v2 6901 chorus
and 6701 fultoni) is redistributed at every setting, consistent with its 'complex' verdict. MBE: "go". The pipeline
takes BROOKS_ALONG_TOL (default 0.6; v3 run at 0.8); the 0.6 partition is kept in <v3 year>/partition_tol06/.


### 19 AL. The streak slope is a between-individual slope (2026-10-13)

Figure 7C fits each trill type's streak (log carrier on log rate over its singer-hours): slopes 0.42-0.61, median 0.51,
nearly parallel. Figure 8 shows the within-singer coupling: ~0.5 for the tree crickets (< 4 kHz) but 0.1-0.35 for
the types above 4 kHz, so for those the streak is steeper than any singer's nightly path. Decomposition
(explore_streak_slope_origin.py, explore/streak_slope_origin.*): the between-track slope (track medians) is 0.5-0.6
and stays 0.5-0.6 with temperature regressed out (r 0.94); rate responds to temperature alike within and between
tracks (3.1 vs 3.4 %/C above 4 kHz) but carrier responds twice as much between tracks as within (1.7 vs 0.8 %/C).
Test 1, simultaneous singers of one type on one recorder (air temperature shared): carrier offset on rate offset
slope 0.55 (median over types; r 0.76), so the covariance is between individuals, not temperature or microclimate.
Test 2, syllable duration against carrier between singers with temperature removed: weak and negative (r -0.07 to
-0.14; shorter syllables with higher carrier), consistent with but not strong evidence for a size axis.
Interpretation: in the resonant singers (ground crickets, trigs) the carrier is set by the harp and barely follows
the night's temperature, while rate does; between individuals, size sets carrier and wing-stroke rate together at
slope ~0.5, and the streak takes that slope. Tree crickets, whose carrier tracks wing-stroke rate within a night, have
the two slopes equal. The within-singer coupling is the right correction for bringing singer-nights to 25 C; the
streak slope is the population's, and the two should not be conflated in the text.


**AM. A formal classification of the types and an idealized reference fold per type; the chirp types matched to
reference clips (2026-10-13).** MBE: "in the next pass through the manuscript we should formally classify each type
and have some kind of idealized reference" / "and no chirpers?" (Figures 12 and 13 had none).

`type_reference_figure.py` (F15, number provisional): for each of the 25 trill and 15 chirp types, up to 40 of the
clearest singer-nights with night temperature 21-29 C (clear pulse measurement first, then depth, one per
recorder-night; the window widened by a degree a side until 15 are available: C4 needed 12-38 C), the deepest window
without another singer in the band (streak_gallery.window_of), fine fold (256-point frames, 32-point hop, 2048-point
FFT) of the band +/- 700 Hz over the loudest 30 s: trills at the refined syllable rate into 64 phase bins, chirpers
at the hour's corrected chirp rate (chirp plane's hourly table; the detector's own chirp rate can be 2x/3x) into 160
bins. Folds in dB above each row's quiet level, onset-aligned (fold_subclass.describe), resampled to +/- 8 % of the
carrier, normalized, pixel-wise median over folds with contrast >= 4 dB (the fine frames spread a pulse's power over
more bins than the analysis frames; a random-window first pass at 6 dB kept 1-20 of 40, the deepest-clean-window
pass keeps 30-40 for most types). Classification by rule, from the per-window descriptors' medians: form = the plane
clustered on; tick = trill with R25 <= 25 Hz; glide = centroid change onset to end / carrier, flat if |glide| < 1 %,
else falling / rising; hooked = falling glide with the mid-pulse centroid more than 0.4 % of the carrier below the
chord (quadratic term x squared half-width); for chirpers syllables per chirp = duty / chirp rate x Rs. Result: all
trill types fall or are flat (tree crickets < 4 kHz glide 3-7 %, types > 6 kHz < 1.5 %); ticks 13, 14, 19, 20, 22,
24; hooked 10, 14, 18, 19 (a first rule, |bend| > 0.4 % and > 35 % of |glide|, flagged every flat high-carrier type
because a flat song's small arch passes a relative test; the rule now requires a real falling glide). Chirpers: < 5
kHz 3-9 syllables per chirp at 40-60 Hz, > 7 kHz 10-15 at 75-90 Hz; C2 (3.5 kHz @ 9.8 Hz) and C4 (3.8 kHz @ 8.4 Hz)
peak in Feb-Mar, have a syllable rate in 1-6 % of hours and 1 and 7 usable folds: not cricket songs (frogs, most
likely); they stay in the catalog as detected but are flagged in the draft. A track can be a member of a trill type
on the main plane and a chirp type on the chirp plane, so the type key travels with the job (a first run grouped
results by track and stacked 64-bin and 160-bin folds together).

Chirp references: `sina_chirp_rates.py` runs the bout test (reference_vs_streak_grid.reference_cell's, over the whole
clip) on every SINA clip with a band: 194 measured, 46 chirpers (contrast > 50x), chirp rate = the rhythm peak in
1.5-12 Hz. `chirp_reference_z.py` fits each chirp type's line (log F on log chirp rate, singer-nights) and its chirp
rate against air temperature (singer-hours), and gives each chirping clip z across and along as streak_id_figure does
on the syllable plane; 33 clips within 2.5 sd of their best line, 32 also along it, 11 of 15 chirp types with at
least one (none has C2, C3, C4 or C7 as its best line; C3's 3.6 kHz @ 3.5 Hz line has O. latipennis and O. fultoni
clips nearby but their best line is C1). `reference_vs_streak_grid.py --chirp-plane --dur 1.0 --page-start 6` draws
the rows for those 11 types as pages 13.6 and 13.7: field singers matched on the singer-night's corrected chirp
rate, the window must carry a chirp rate, 1 s cells so a cell holds a few chirps, cell heads give chirp rates to a
tenth. The clip's syllable rate (not its chirp rate) is passed to the bout test's smoother. Readable at a glance: C5
beside G. fultoni, C6 beside C. gundlachi, C9 beside C. velox and C10 beside A. socius are the same chirp structure;
the > 10 kHz chirp types' clips are katydids with broadband sweeps and the field cells are faint.


**AN. QC grids for the chirp types; grid pages named by type (2026-10-13).** MBE: "do we have quality control figures for
all the types - trills and chirps" (no: the supplement grids covered the main plane only), "why are there so many 'no
windows' in those new chirp grids?", "the s_grids_v3 aren't named by type so are hard to parse - and the insets showing
where the samples are from are overlapping", "i don't think you need the numbers - they're in order".
`streak_grid.py --chirp-plane`: members and streaks from <root>/chirp (ids C1..), along-line coordinate in log carrier
against log chirp rate with the chirp plane's coupling 0.35, window_of(form="chirp"), `--dur 1.0`. The "no window" cells
were the grid's own filter: it rejected a window without a syllable rate, and a chirper's deepest window often carries
only the chirp rate (C2 reads a syllable rate in 1 % of hours, C4 6 %, C7 10 %, C1/C3/C8 13-19 %); G.analyse already
folds such a window at its chirp rate, so the grid now accepts it and the cell head reads "no syllable rate c3.2".
Pages are named by the paper's type number (type_05_streak_3601, type_C3; the 4 chirp-form streaks of the main plane
keep streak_<id>), heads say "type 14 (streak 4601)", the combined PDF is in type order. The inset is now two: a
locator on the whole plane with a box around the streak, and a zoom on the streak with the row centers as open rings
in order, no numbers (twelve numbered rings on the whole plane overlapped). Head text wrapped to 115 characters so it
stays clear of the insets.


**AO. The chirp re-pass: type only chirpers whose syllables were read (2026-10-13, late).** MBE read the 15 chirp
types' QC grids (AN) and found only C9/C10/C11 clean, C2/C4 frogs, the rest mixed ("those are too messy to be reliable
- so do it"); his verdict tracked the share of a type's tracks with a syllable rate read inside the chirps (85-93 %
for the passes, 0-6 % for the frogs; QC_NOTES_MBE_2026-10-13.md). The chirp plane clustered on carrier and chirp rate
only, so any bout rhythm below 8 Hz in the band joined a type. `chirp_year.py --min-syl-share 0.5` (now the default)
keeps a track only if its syllable rate was read in >= 50 % of its hours: 3,943 tracks / 15,547 h typed; 3,894 tracks
/ 17,361 h written to chirp/bout_only_tracks.csv and bout_only_hours.csv as an untyped bout-only class (half of them
below 4 kHz; 10 % in Dec-Mar: the frogs; the rest birds, raspers, chirpers whose syllables are too fast or broadband
to read). The old plane is archived as chirp_v1_allbouts/. Result: 11 chirp streaks, 7 with >= 40 singer-nights
(C1-C7), holding 96 % of the typed singer-nights; 91 % of the typed hours carry a syllable rate (was 47 %). Old ->
new (chirp/chirp_type_map_v1_to_v2.csv, by shared singer-nights): old C3 (3.6 kHz @ 3.5 Hz) and old C5 (4.5 kHz @
4.8 Hz) are ONE type, new C1 at 3.9 kHz @ 4.1 Hz: the same 47 Hz syllable rate, one line from 2 chirps/s at 3.2 kHz
(17 C) to 6 at 4.3 kHz (27 C), chirp rate +6.6 %/C; the frogs and bout-only tracks had held the two apart. Old C6 ->
C2, C8 -> C3 (only 161 of 1,133 old members had syllables), C9 -> C4, C10 -> C5 (2,532 of 2,730 kept: the one old type
that was already clean), C11 -> C6, C14 -> C7; old C1, C2, C4, C7, C12, C13, C15 dissolve (under 40 syllable-bearing
singer-nights, or into the small streaks). Everything downstream redrawn (chirp_repass.py in the scratchpad: F09,
F11, chirp_reference_z -> F13 pages 6-7, F15, F10, S_grids_v3_chirp): the C1 grid is 60 cells of the same 5-9
syllable chirp from 16 to 28 C; 25 of 46 chirping SINA clips lie within 2.5 sd of a chirp line and all 7 types have
one; F15 chirp groups: C1-C3 hold 5-7 syllables at 36-55 Hz, C4-C7 11-13 at 67-84 Hz. Draft paragraphs for Figures
9, 10, 11, 12 and 15 updated.


**AP. C7 dropped by QC (2026-10-13, late).** MBE on the re-passed chirp grids: "type 7 is a mess - but other than that it
looks a lot better". C7 = chirp streak 13, 15.3 kHz @ 3.1 Hz, 63 singer-nights, 51 of them at House, tracks of 2-6
hours (median 3) and low depth (median 0.34), syllable rate 50-95 Hz; the grid holds broadband katydid bursts, faint
slivers and noise rather than one song. `chirp_year.py --exclude 13` removes it after clustering from the tables every
downstream script reads (animal_nights_streaks_c2.csv: its singer-nights -> -1; streaks_coarse_c2.csv: row dropped;
unfiltered copies *_c2_all.csv; reason in chirp/excluded_streaks.csv). C1-C6 keep their numbers (C7 was last by
carrier). Redrawn: F09, F11, chirp_reference_z -> F13 page 13.6 (6 rows, one page), F15 from the cached folds (the
excluded key is skipped), S_grids_v3_chirp (C7 page and clips removed, PDF re-merged). A count threshold alone lets
such a streak through: the paper's type list is the clustering plus a stated QC exclusion.

Note: the F14 draft edit of the same evening corrupted RESULTS_DRAFT_F07_F12.md (a slice with reversed bounds gave an
empty string, and str.replace("" , text) inserted the text between every character: 12 MB); restored from commit
95faf76 and the Figure 14 and C7 edits reapplied with the day section bounded by the end of the file.


**AQ. Figure 5 grayscale; the release (2026-10-13, late).** F05's excerpt cells had never had the grayscale-outside-the-band
treatment of Figures 10 and 13 (MBE: "did we decide not to do the grayscale thing for F05?"); night_streak_examples_figure.draw_row
now draws the detected band in magma and the rest in gray on the same scale, and the dotted band-edge lines are dropped.
`make_release.py` assembles release/brooks-v1/ (code closure over imports and subprocess calls from the pipeline and figure
entry points, 71 modules; demo = two 5-min windows cut from the Figure 3 night laid out in the archive's folder convention so
BROOKS_AUDIO_ROOTS=demo/audio works, plus that night's unit directory and the HRRR rows, and run_demo.py which detects,
draws and relinks; data = the year tables behind Figures 7-15 with the type catalog's candidate-name columns removed, the
chirp and SINA tables, the type-reference folds, recorded hours, and every track of the year as two gzipped tables; figures
F01-F15 with MANIFEST.md; the supplement grid PDFs; README, DATA_DICTIONARY, environment.yml, requirements.txt, LICENSE
(MIT, from the repo), CITATION.cff) and zips it: brooks-v1.zip 343 MB, brooks-v1-listening-clips.zip 801 MB. The demo ran
end to end in the build (2 windows detected and drawn; the night relinked: 2,562 detections -> 228 tracks). release/ is
git-ignored; the builder is committed. Not included: the per-window detection tables (2.2 GB) and the audio archive.


**AR. Geographic range in the reference comparison (2026-10-14).** MBE: "Are we restricting the SINA matches to species
that might plausibly be in Arkansas?" We were not: F12/F13 compared every measured clip (182 of 147 species); only the
withdrawn position-based candidate list (match_sina.py) had filtered on SINA's distribution tables. `sina_range.py`
now marks a clip in range when its species has records in AR or a neighboring state (species_geography.csv n_near > 0:
LA, TX, OK, MS, MO, TN) or the clip is hand-supplemented. The first run kept match_sina's exemption for trigs, ground
crickets and scaly crickets (thin AR county data) and excluded nothing, because every one of the 28 out-of-region clips
(27 species) is a small cricket: Florida/Gulf-coast Cycloptilum (ainiktos, albocircum, distinctum, irregularis,
spectabile, tardum, zebra), Pictonemobius (4), Anaxipha (calusa, delicatula, fultoni, imitator, litarena, rosamacula,
scia, tinnula), Cyrtoxipha (confusa, gundlachi, nola), Hygronemobius alleni, Allonemobius griseus and fultoni,
Neonemobius eurynotus, and the European Nemobius sylvestris. The exemption was dropped: no record in seven states is a
stronger statement than none in Arkansas. streak_id_figure and chirp_reference_z write *_reference_z_all.csv (every
clip, in_range column) and *_reference_z.csv (in-range only, what Figure 13 reads); F12A draws out-of-range clips as
hollow gray markers. Counts: trill plane 154 in-range clips, 70 within 2.5 sd across, 52 along, 24 of 25 types (was
91/67/25); chirp plane 38 in range, 15/14, 6 of 6 (was 21/19). Figure 13 rows that changed species: 7 C. distinctum ->
E. melodius, 10 N. sylvestris -> A. vernalis, 11 A. litarena -> E. confusus, 18 A. imitator -> A. maculatus, 19
C. spectabile -> A. walkeri (now shown for 14 and 19), 20 A. fultoni -> A. thomasi, C2 C. gundlachi -> H. boreale.

**AS. Syllable shape words (2026-10-14).** MBE: "Are there any accepted names for syllable shape in the field?" There
is no settled Orthoptera vocabulary for the within-syllable frequency course (the literature describes pulse shape with
oscillograms and frequency with a single carrier figure), so the words come from bird-song description, where
syllable shape on a spectrogram is routinely named: flat, downslur, upslur, hockey stick (a steep fall that levels),
chevron (a rise then a fall), broadband (no carrier to follow). MBE: "i like the words rather than abbreviations".
The type table (F15_type_reference_table.csv) gained a `shape` column assigned by rule from the fold descriptors
(|glide| < 1 % flat; falling glide downslur; rising upslur; the former hooked flag hockey stick) and then checked by eye
against Figure 16's exemplar folds, which overrode five: 13, 14, 18, 19 and C2 are chevrons (the ticks' and the
late-summer choruses' syllables rise and fall within the syllable, which the glide descriptor, onset to end, reads as
a small fall) and 17 is flat (its 1.2 % glide is at the threshold). The
"hooked" word in the draft is retired for hockey stick. Figure 16's cell heads carry the shape word on a second line
(beside the numbers it ran into the next cell); HEAD 0.21 in, row pitch 0.92 in.

MBE then read the redrawn Figure 16 and disagreed with several of the rule's labels: "1-8 downslur; 9 indeterminate;
10 hockey stick; 11 could be either downslur or hockey stick; 13, 14 chevron; 15 downslur; 16 chevron; 17 is unclear -
i wouldn't say flat; 18, 19 chevron; 20 chevron; 21 chevron; 22 flat; 23 chevron; 24 a mess; 25 flat; chirps I agree
with." His call is now the `shape` column wherever he gave one (shape_source "MBE by eye on Figure 16 (2026-10-14)",
his words in shape_note; the rule's label stays in shape_rule; type 12, not mentioned, keeps the rule's downslur).
Eight changed: 9, 17, 24 indeterminate (were downslur, flat, flat), 11 "downslur / hockey stick", 16, 20, 21, 23
chevron (were downslur, flat, flat, flat). Result: 10 downslurs (1-8, 12, 15), 9 chevrons (13, 14, 16, 18-21, 23,
C2), 7 flat (22, 25, C1, C3-C6), 1 hockey stick (10), 1 either (11), 3 indeterminate (9, 17, 24). Note what the rule
missed: the high flat types 20, 21, 23 are chevrons to the eye, so the |glide| < 1 % rule (onset-to-end centroid
change) cannot see a rise-and-fall that ends where it began; a chevron needs the mid-syllable centroid above the chord,
which is bend_pct > 0, and the rule should be revised to use it before the table is called final.

**AT. InsectSet459 measured: not a reference set as it stands (2026-10-14, late).** The full dataset (26,298 clips) was
downloaded and the 3,690 clips of the 49 North American Orthoptera species were measured with the window analysis
(insectset_measure.py, BROOKS_CHIRP=1; 10 workers, 1 h 28 min): 3,574 ok, 3,016 with a carrier and a syllable rate,
832 chirpers by the bout test, 216 with a recordist temperature. insectset_analysis.py: no species falls on a line.
Across-line sd per species 0.16-1.0 log units (median 0.45) against 0.030 for our 25 types; the species' fitted
slopes of log carrier on log rate scatter around zero. The carrier is not the problem: 1,898 of 2,054 clips with a
SINA page frequency measured within 0.35 log units of it. The syllable rate is: for O. fultoni and O. rileyi (both
chirpers) an independent envelope-autocorrelation rate on the band agrees with the detector's R in only 36-40 % of
clips (latipennis 82 %, quadripunctatus 75 %), and a 12-clip grid of O. rileyi (runs/insectset459/rate_check_rileyi.png)
shows why: the set mixes clean chirps (2.4-2.7 kHz, ~50 syllables/s inside the chirp, R and the envelope rate agree)
with clips that hold only a noise band at 1.7-2.0 kHz, several clips cut from one iNaturalist observation, and
chirp-rate harmonics read as R (16, 22, 26 Hz). The labels are per observation, the recordings are phone-quality and
compressed, and nothing marks which clip actually holds the named singer. Verdict: usable only after curation (a
depth/contrast gate plus a look at every kept clip), which is the work the SINA set already does for us with known
temperatures; parked. Tables: runs/insectset459/measurements.csv, insectset_species_table.csv,
rate_check_oecanthus.csv.

**AU. Candidate splits examined (2026-10-15).** Figure 7 supplement (type_panels_figure.py: each trill type's singer-hours
by temperature over all 25 lines) led MBE to ask whether more splitting is due: "5 is two lines for sure; 6 looks like 3;
9 has something else mixed in; 16 looks like multiple things; 23 looks like two". Diagnostics, all under
figures_paper/exploratory/: across_line_vs_rate.png (offset across each type's line against rate, by temperature: the
color stratification across the line in 10, 14, 19, 20 is the slope mismatch of METHOD 19 AL, not two populations);
across_line_vs_rate_glide.png (the same colored by the per-singer-night pulse glide: type 5's warm lower tail and 6's and
9's lower bands are steep downslurs, their upper bands near-flat); band_partitions.png and the _v9bands catalog (each
candidate type's singer-nights cut on the across-line offset by a 2- or 3-component mixture, bands a, b, c from the
lowest carrier; paper ids may be band labels, paper_ids.load tolerates strings); band_grids/ (streak grids per band,
letter pages); band_grids/band_rate_vs_T.png (each band's rate and carrier against temperature: 5b and 6c are cold,
shallow-slope bands, 16c is a spring singer at 53 Hz at 25 C against 16a's 44, 9a and 9b are parallel lines 7 Hz apart,
23a and 23b share one rate line and differ 5 % in carrier); band_folds.py + band_shape_figure.py (fine folds per band:
5a/5b the same downslur; 16a downslur against 16b/16c chevron-like; 23a/23b both broad and flat). MBE on the grids: "some
of the groups are pure downslurs and their overlapping group is a mix of downslurs and chevrons/hockey sticks - it's
just not clean so i'm not sure exactly how to deploy it". Nothing in the catalog changed; the band files are a parallel
suffix. Two new page types came out of this: singer_night_grid.py (one type, eight singers as rows, five moments through
each singer's night as spectrogram cells, the singers' paths on the streak inset; MBE: "that's actually fantastic";
drawn for all 25 types into exploratory/singer_nights/) and singer_paths_grid.py (Figure 8A for a whole type: 48
singers' carrier-against-rate paths by temperature, ordered along the streak; exploratory/singer_paths/).

**AV. The fixed-window carrier damps the within-singer slope, most for the high types (2026-10-15).** MBE: "I do wonder
whether there's a bit of an ascertainment bias in the frequency behavior of singers in big clusters, where there's a
bias to stabilize frequency?" Test on 3,035 tracks (up to 150 per trill type, >= 12 windows, rate range >= 2 % of R25;
exploratory/coupling_vs_crowding.csv, carrier_estimators.csv): the within-track slope of log carrier on log rate computed
three ways from the per-window table, F_fix (the fixed-window R-locked centroid the whole chain uses: reference band +/-
FIX_PAD = 150 Hz), F_c (the detected part's centroid) and F_pk (its peak). Crowding is not it: the number of simultaneous
same-type tracks on the recorder is uncorrelated with the slope (r -0.02) and lone singers have the same slopes. The
estimator is: F_fix gives 0.34 over all tracks, F_c 0.47, F_pk 0.47, and the gap grows with carrier because the pad is
absolute: the reference band is 11-13 % of the carrier wide for the tree crickets but 6-8 % for the types above 6 kHz,
the raw centroid leaves it in 15-33 % of their windows (0-6 % below 4 kHz), and the fixed window clips the carrier's
excursion (F_fix 5-95 % range 3.5-4.8 % against F_c 6-7.6 % for types 14-25; the ratio of the two ranges correlates with
the reference width, r 0.39). Per type with F_c (F_fix in parentheses): 2 0.60 (0.55), 5 0.58 (0.51), 10 0.57 (0.45),
14 0.40 (0.24), 16 0.49 (0.31), 18 0.50 (0.28), 19 0.41 (0.25), 21 0.42 (0.25), 23 0.54 (0.29), 25 0.50 (0.31); still
low with any estimator: 20 0.30, 22 0.16, 24 0.15 (the 7.5-9 kHz ticks), and 8, 9, 11, 12 at 0.36-0.43. Consequence:
METHOD 19 AL's "within 0.1-0.35 above 4 kHz" and Figure 8B are an F_fix result; with the raw centroid most high types'
within-singer slopes come within 0.1 of their streak slopes, and the resonant-harp reading survives clearly only for 20,
22 and 24. The coupling notes (paper_draft/COUPLING_notes_2026-10-14.md) carry a caveat. Next: year_coupling and Figure 8
should take a carrier choice (F_c, or F_fix with a relative pad), and the hourly points' F should be checked for the same
damping (it enters the streak slopes too, though between-singer spread dominates there). Type 20's singer page
(singer_nights/type_20_singers.png) shows the symptom MBE noticed: the eight singers' paths run flatter than the streak
and end above it at the cold end; by the cell heads (F_fix) their carriers fall 3-9 % over 8-10 C of cooling while
the rate falls 20-30 %.

**AW. The v4 relink: relative pad, carried types (2026-10-15).** MBE: "rerun the linker." `make_run_tree.py` built
runs/v4 from v3 (6.36 M symlinks, 16 min; night AND day link outputs left out), `night_animals.FIX_PAD_REL` = 0.06 from
BROOKS_FIX_PAD_REL (pad = max(150 Hz, 6 % of the reference center)), and `year_pipeline.py --from link` with
BROOKS_RUN_TREE=v4 BROOKS_SYLV2_LINK=1 BROOKS_ALONG_TOL=0.8 ran the whole chain (link 21 min on 4 x 5 workers, 0 failures;
chain done in 2 h 40). The pad changes only F_fix: a scratch relink of one night gave identical track ids and widened the
4-6 kHz tracks' 5-95 % carrier range from 4.6 to 6.3 % (raw centroid 6.7 %). Over the year the singer-nights are the same
26,116 and their median carriers moved by 0.00 % (IQR -0.07 to +0.15 %, mean |change| 0.4 %).

Yet re-clustering those positions re-partitioned the catalog: 23 trill types >= 40 instead of 25, type 10 cut 59/40 into
two along-line segments (5.4 kHz @ 41 and 4.6 kHz @ 33), 14 and 19 merged into one 6.5 kHz @ 21 type of 2,493 with the
rest of 14 as a 5.9 kHz @ 20 type, 5 lost 25 % to a new 2.9 kHz @ 28 cold segment, 23 split 73/23 into 7.9 and 8.8 kHz
at 55 Hz (compare_catalogs.py; exploratory/catalog_v3_vs_v4.*). Average linkage at the 0.8 tolerance is unstable to
perturbations two orders of magnitude below the type spacing (METHOD 19 AK saw the same between 0.6, 0.8 and 1.0): the
partition is a choice frozen in v3, not a measurement to be redone. So `carry_types.py` carries the v3 partition onto
the v4 measurements by singer-night id (suffix _v3types in runs/v4/year_nov24_nov25: the v3 assignments and paper
numbers, v4 F/R/F25/R25, streaks_coarse positions recomputed; 26,116 nights, 5,427 days), and the auto-partitioned
_v9 of v4 stays on disk as the instability's record.

Result on the carried types (year_coupling_tracks_v3types.csv, streak_lines_v3types.csv; exploratory/
within_slopes_v3_vs_v4.csv, F08_coupling_v4.*, F07_year_plane_v4.*): type positions at 25 C move < 1 %; streak slopes
unchanged (median 0.51 both); the within-singer slope median rises from 0.30 to 0.40 above 4 kHz (0.51 -> 0.54 below),
per type 10 0.41 -> 0.54, 14 0.27 -> 0.40, 16 0.27 -> 0.41, 17 0.36 -> 0.54, 18 0.28 -> 0.43, 19 0.27 -> 0.41, 21 0.30 ->
0.47, 23 0.32 -> 0.54, 25 0.31 -> 0.50, agreeing with the raw-centroid values of 19 AV within a few hundredths; 20 (0.29),
22 (0.16) and 24 (0.15) stay low, and 8, 9, 11 at 0.31-0.34. The paper's numbers are still v3 until MBE says v4
replaces them; the switch means re-pointing the figure scripts at runs/v4 with --suffix _v3types (the chirp plane,
pulse shapes, coherence and galleries of v4 were built on the auto partition and would need the carried one).

**AX. The switch to v4 (2026-10-16).** MBE: "switch everything to v4 with the carried types" / "run everything without
prompts." In runs/v4/year_nov24_nov25 the carried catalog (carry_types.py, METHOD 19 AW) became the run's `_v9` (its
own re-clustering moved to auto_partition_v9/), the chirp catalog was carried the same way (v3's C1-C6 assignments on the
v4 chirp measurements; the v4 re-clustering had split v3's C5 and renumbered, 25 % label agreement, same clusters; the
QC exclusion of streak 13 must be passed as --exclude 13 whenever chirp_year redraws with --skip-build, since it re-applies
the list to _c2_all), and `year_pipeline.py --from names` rebuilt names, types, pulse shapes, coherence, the catalog
and the galleries on it (18 min + galleries). `regenerate_figures.py --tree v4` then redrew Figures 7, 7s, 8, 9, 10, 11,
12, 13 (trill and chirp pages), 14, 15, the type reference (cache rebuilt, 27-40 folds per type; the by-eye shape
columns saved and merged back), 16, the focal-night Figures 4b, 5, 6 (same track ids, new carriers), and the supplement
grids with audio (S_grids_v4, S_grids_v4_chirp). Figure 12: 71 in-range clips within 2.5 sd across (was 70). All
exploratory pages drawn earlier today still show v3 carriers unless noted. The release (brooks-v1) predates all of this.

**AY. The catalog split on the v4 carriers (2026-10-16).** With the clipping gone the v4 type panels showed structure the
v3 partition had merged. MBE, type by type: "new type 6 is definitely two things", "13 also", "3 also", "5", "16 ...
it's a total mess", then "some of those are clearly different, so I guess we have to change the type catalog." A
two-component scan of all 25 types (exploratory/split_scan_v4.*: a mixture on position along the line at 25 C and offset
across it) and the question MBE asked first -- are the second components pieces of OTHER types? -- answered no: every
minor component's own 25 C position is >= 0.09 log units from any other type's line (split_scan_v4_components.csv);
types 4, 8, 11, 12 are the exceptions whose small minor components sit nearer a neighbor (2, 6, 9, 15) and were not
touched. For 1, 3, 5, 6, 13, 16, 23 each type was cut in two along or across its line, whichever separated better
(exploratory/split_v4/split_candidates.*; grid pages per half, letter format, _v9split catalog). The decisive test is
rate against temperature by half: 6, 16 and 23 show two rate lines over the same temperatures (45/60, 43/50, 40/58 Hz
at 25 C), 13's halves meet only at 23-24 C and differ there (20 vs 32 Hz), 3's overlap at 20-24 C as two stripes (48/55
Hz), 5's cold upper band gives no second rate line (its shallow slopes are what the slope mismatch does), 1's halves
have crossing slopes (not a split). The grids: 6a (3.4 kHz @ 44) faint fine downslurs, 6b (4.0 @ 59) bold long syllables
-- different songs; 13b (7.0 @ 31, 21 nights, mostly NorthSide June-July) a faint flat band unlike 13a's bright hooked
ticks; 23a/23b the same thin flat trill at two carriers.

Decision: split 3, 5, 6, 13, 16, 23. The larger half keeps the streak id and paper number (so 3601, 4601, 3151 and the
F13 id list still resolve); the split-off halves take id x10+k and new paper numbers by carrier: 26 = 3a (2.4 kHz @ 45,
109), 27 = 6a (3.4 @ 44, 90), 28 = 5b (3.8 @ 38.5, 349), 29 = 16b (7.0 @ 52, 128), 30 = 23a (7.4 @ 43, 633); 13b (21
nights) is small type 49. 30 trill types with >= 40 singer-nights. The carried-but-unsplit catalog is kept as
_v9carried. Day assignments stay with the keeping half (assign_day_to_streaks was not rerun on the halves). Type 5's
split follows MBE's eye ("5 is two lines for sure") against the rate test; 1 was left whole. Then year_pipeline --from
names, regenerate_figures (trill ids now read from the catalog; type panels grow with it) and, as the point of the
exercise (MBE: "you also didn't remake the singer nights"), singer_night_grid and singer_paths_grid for every type into
exploratory/singer_nights_v4 and singer_paths_v4. The type table's shape words exist only for the 25 old numbers; the
new five need calls.

Renumbered the same day (MBE: "if those are now new types we need to renumber" / "a little confusing in the ordering but
it's fine"): paper_ids.assign over the split catalog, every trill type by carrier at 25 C, so the old numbers shift
(old -> new: 1->1, 3a->2, 2->3, 3->4, 4->5, 6a->6, 5->7, 5b->8, 6->9, 7->10, 8->11, 9->12, 10->13, 11->14, 12->15,
13->16, 16->17, 14->18, 15->19, 17->20, 18->21, 16b->22, 19->23, 23a->24, 20->25, 21->26, 22->27, 23->28, 24->29,
25->30). The mapping is runs/figures_paper/type_renumbering_2026-10-16.csv (and type_renumbering.csv in the v4 year
dir); the type table's shape words were carried by streak id to the new numbers; every draft text written before
2026-10-16 uses the old numbers.

**AZ. Pass 2 of the split (2026-10-16).** On the pass-1 singer-night pages MBE picked out "type 8 - which has a cold and a
hot blob; type 9 - which clearly has a secondary streak; type 12 - which looks like it might be 2 streaks". Same
procedure (exploratory/split_v4_pass2/): 8 cuts along its line into a cold 2.7 kHz @ 22 Hz tick (76 nights, Oct-Nov,
a long-downslur slow tick, its own line 0.45 log units from any type) and a 3.9 kHz @ 40 Hz trill (273); 9 cuts across
into the 3.9 kHz @ 58 Hz downslur trill (167, warm) and a 4.2 kHz @ 62 Hz band of bold flat vertical bars over 4-28 C
(44), a different syllable; 12 cuts along into 4.3 @ 43 (124) and 5.0 @ 55 (62), parallel rate lines 25 % apart but the
same thin syllable on both pages. 8b's page is itself two things to the eye (cold Aug-Dec cells broad and fuzzy, warm
Jun-Sep cells crisp downslurs); a two-component cut on night temperature gives 231 cold (10-20 C) and 42 warm (22-29 C)
singer-nights on one plane position, the warm rate line continuing the cold one (38 vs 41 Hz at 25 C), so this cut is
season, not geometry, and stands on MBE's eye. MBE: "I think all should be split - and I agree that 8b looks like 2
different things." Done: 8 -> 360123 (cold tick), 360124 (8b cold), 360125 (8b warm); 9 -> 31013, 31014; 12 -> 34013,
34014 (pass-1 ids collided: 31011 was already new type 6, so pass 2 uses +3/+4/+5). Renumbered again by carrier
(type_renumbering_pass2_2026-10-16.csv; pass-1 catalog kept as _v9pass1); shape words carried by streak. Chain rerun:
names..galleries, all figures, supplement grids, singer-night and paths pages.

**BA. Pass 3 and the v5 catalog (2026-10-16).** MBE's third read of the singer pages, with two new tests, became the
final pass ("let's do it as v5"). Tests added this pass: (1) density views per type (hexbin on the plane, density of
rate and of carrier against temperature, and a histogram of each singer-night's rate residual from the type's own
rate-vs-temperature line; exploratory/split_v4_pass3/density_*.png): one population gives one residual peak, two rate
lines give two; this retired the mixture-model "two lines" flags for 17, 12 and 3 (all unimodal) and confirmed 28
(bimodal, peaks 30 % apart). (2) The bridging test (MBE: "whether we ever see individual singers bridge the gap"): the
share of singers whose own hours fall on both sides of a proposed cut, and the within-singer sd of the matched-T rate
residual against the cut's spacing (bridging_*.png). Real splits: 28, 1 % of singers cross, singers wander 1.9 % vs
26-33 % spacing; C1 (syllable rate 58 Hz), 8 %, 2.8 % vs 44 %; C3 (62 Hz), 15 %, 5.2 % vs 52 %. Not a split: 17, 18 %
cross a 13 % cut, one continuous population; 15, no night crosses its 15.5-17 C temperature gap (31 cold, 81 warm, 0
straddling), so untestable and left whole; 10, cut by pulse shape (glide < -2 %) gives parallel rate lines 10 % apart
but shape predicts position only 73 %, left whole and flagged shape-heterogeneous; 2 vs 5 (the pass-1 halves of old
3): collinear on the plane but 2's lower part sings 16 % slower than 5's line at matched temperature, kept apart.

Splits taken: 8's fast end (R >= 50 Hz, 114 nights, own line 4.8 kHz @ 64 Hz 0.4 log units from 8's, flat rate
response, a dense fine trill unlike 8's downslur) -> new type 16; 28 four ways on the two matched-T residuals (BIC
prefers 4): 28a 5.9 kHz @ 28 (54, NorthSide) -> 22, 28b 7.0 @ 38 (273) -> 28, 28c 7.8 @ 50 (300) -> 34, 28d (6) small;
29's high blob (20 nights at 9.3 kHz @ 33 Hz, 50 % faster than 29 at the same temperature, a dense flat trill not 29's
tick) -> small type; C1's fast-syllable singers (62 nights, 70 Hz inside 2.8 Hz chirps, 26 syllables per period, long
dense chirps Sep-Oct on NorthSide/Pond/FarNorth) -> C1 (3.6 kHz @ 3.1), the main group C2; C3's fast half (48 nights,
80-90 Hz syllables) -> C5, slow half C4. v5 = runs/v5: units and year_figs from v4 by symlink, the year dir's inputs by
symlink, sina/ copied, the catalog written fresh (37 trill types >= 40, 8 chirp types; _c2_all keeps streak 13 so
chirp_year --exclude 13 regenerates _c2 unchanged). Renumbered by carrier (type_renumbering_v5_2026-10-16.csv: v4 pass-2
numbers 1-15 unchanged, 16+ shift by up to three); shape words carried by streak. Chain: year_pipeline --from names
(BROOKS_RUN_TREE=v5), regenerate_figures --tree v5 incl. supplement grids and singer pages.

**BB. Walker & Funk's Anaxipha recordings on the plane (2026-10-16).** MBE: "I would like to add the Anaxipha Walker and
Fink data to the paper - it's pretty compelling as IDs for the streaks it overlaps, no?" / "as a figure - like we did
before". A first version digitized their Figure 4 (carrier against pulse rate, six species; exploratory/
anaxipha_digitized_points.csv); MBE then downloaded the whole supplement to analysis/callers/anaxipha/ (Walker & Funk
2014, J. Orthoptera Res. 23:1-38; SINA g610ms3.htm), and `anaxipha_published.py` tidies SMTbls_Songs.xlsx (13 species
sheets; 1,318 recordings with temperature, pulse rate and carrier; column names vary by sheet) and runs the Figure 12
test against the v5 type lines: per recording, z across the type's carrier-on-rate line (sd floor 0.03) and z along it
from the type's rate-on-temperature line at the recording's temperature (sd floor 0.08, 2 C uncertainty), a recording
"on" a type when both |z| <= 2.5. Best type per species (share on, n in the type's rate range): vernalis -> 18 (5.2 kHz
@ 38 Hz) 94 % of 149; litarena -> 19 (5.3 @ 54) 95 % of 60; delicatula -> 20 (5.3 @ 69) 83 %; exigua -> 27 (6.5 @ 39)
79 % of 193, with 34 and 28 close behind (exigua's 195 recordings span 17-35 C and 23-52 Hz and lie between our lines
27/28 and 34); rosamacula -> 19 78 %; fultoni -> 29 (7.0 @ 51) 69 %; thomasi -> 31 (7.5 @ 22) 62 % (30 at 61 %, the two
lines 7 % apart); scia 46 % at best; tinnulacita, tinnula, tinnulenta below the detector's rate floor, no type (their
SINA clip for tinnulacita had been matched to 36 by position, 4 sd off along: explained). Species lines from the
recordings: rate +3.9 to +6.9 %/C, carrier +2.5 to +4.0 %/C, log carrier on log rate 0.42-0.88 (0.57 vernalis, 0.63
exigua). Range (State column, AR + LA/TX/OK/MS/MO/TN): exigua 12, tinnulenta 18, rosamacula 9, scia 8, vernalis 7,
thomasi 3 recordings; delicatula 1, litarena 2, fultoni/tinnula/calusa/imitator 0. F12b_anaxipha_overlay (number
provisional): Figure 7A's plane with the recordings as rimmed circles colored by their own temperature on the hours'
scale, species lines and names, gray type numbers at the lines' warm ends (year_plane_figure --a-only --overlay; a T
column in the overlay CSV switches the coloring). Tables: anaxipha/anaxipha_songs_tidy.csv, anaxipha_vs_types.csv.

MBE (2026-10-16): "vernalis, exigua and tinnulenta are all on inaturalist from Arkansas ... pretty sparse records." SINA's
table has n_AR = 2 for vernalis (632) and 0 for exigua (616) and tinnulenta (615), all three in range already through
neighboring states. The iNaturalist presence is kept in analysis/callers/range_supplement.csv (page, species, source,
note), which sina_range.in_range_mask now adds to the SINA count, so a species with Arkansas observations outside SINA
counts as in range even with no SINA record in the seven states. No Figure 12/13 membership changed.

**BC. Temperature responses type by type (2026-10-17).** MBE: "two figures - Figure 18 and 19 for now - that show T vs
R and T vs F for the trillers" / "show me what the within-singer versions look like" / "those last two withins are the
ones to use - can you make those production figures and make a table as well?" `temperature_panels_figure.py` draws one
panel per trill type (37, shared axes). Pooled version (--var R|F): the type's singer-hours against HRRR air temperature
with the fit of log value on temperature; slopes rate 0.9-6.0 %/C (median 3.0), carrier 0.1-3.7 (1.5). Those pooled slopes
mix the within-singer response with between-singer composition along the season: type 4 (cold, few degrees sampled)
pooled 0.95 %/C but 3.7 within; types 1 and 3 pooled 5.9-6.0 but 3.6-3.7 within (warm nights carry faster singers).
Within-singer version (--within, now Figures 18 and 19): tracks with >= 4 hours and >= 2 C of range (37 types, 5-1,125
singers each), each hour as deviations from the track's own mean temperature and mean log value, the pooled within-track
slope, and the median and IQR of per-track slopes; rate 1.7-4.9 %/C (median 3.2), carrier 0.3-2.6 (1.4), carrier/rate
ratio median 0.48, lowest in 2 (0.13), 33 (0.18), 31 (0.25), 4 and 36 (0.28): the resonant-harp candidates. Table:
runs/figures_paper/Table_temperature_slopes.csv/.md (per type: both slopes within and pooled, IQR over singers, n, the
ratio). Pooled pages kept as exploratory/F18p_, F19p_.


**BD. iNaturalist recordings on the plane (2026-10-17).** MBE: "are you able to automatically download recordings on
iNaturalist? ... quite a few orthopteran ones from Arkansas and surrounding states" / "build it". `inat_fetch.py` pulls
every research-grade Orthoptera observation with a sound from Arkansas and its six neighbors (API place ids AR 36, LA 27,
TX 18, OK 12, MO 28, MS 37, TN 45; taxon 47651): 1,879 sounds from 1,780 observations and 103 taxa into runs/inat
(1,877 converted to 48 kHz mono WAV; 578 carry no license and are not redistributable; downloaded files are untrusted
data, read with python -I from outside their directory). `inat_measure.py` measures each as the SINA/InsectSet clips
were (insectset_measure._one) with the species' SINA page frequency as a band hint (68 of 103 taxa; without it the fixed
window locked onto katydids or noise and cricket carriers came out near 10 kHz), sets aside the AAC frame-rate artifact
(1024 samples at 44.1 kHz = 43.07 Hz, also its half and double; 84 measurements, ~9 % of the m4a files read it as a
syllable rate), attaches an ERA5 2 m temperature at the observation's coordinates and hour (Open-Meteo archive, cached
in era5_cache.csv; 1,843 of 1,877), and runs the Figure 12 type test. 1,769 measured, 1,386 with carrier and rate, 522
on a type's line at matched temperature (crickets 214 of 493, 43 %; katydids and others 308 of 893, a coincidence rate).
The overlay of all species with fitted lines on Figure 7A's plane (year_plane_figure --overlay-min-n) was unreadable; MBE
asked for "a separate panel for each iNat species - and no lines": `inat_species_panels.py` (exploratory/
inat_species_panels.pdf, 42 species with >= 5 recordings, type lines in gray, points colored by ERA5 temperature). What
it shows: Velarifictorus micado (58) in a tight cluster on type 7 (3.6 kHz @ 45-60 Hz); Neocurtilla hexadactyla (110) at
1.7 kHz and 40-90 Hz below every line, a song not in the catalog; Oecanthus fultoni (42) spread at 2.5-3.5 kHz near types
1 and 5 but on neither; Gryllus texensis/rubens/assimilis near types 9-14 as chirpers measured by syllable rate; Anaxipha
vernalis 5 of 7 on type 18 (Walker & Funk: 94 %); Eunemobius/Neonemobius/Allonemobius shalontaki in the 6-9 kHz band near
30-35; katydids scattered, Pterophylla camellifolia's 526 points a horizontal band at 3-6 kHz across all rates (the
measurement reading whatever cricket or noise is in the band). Verdict as for InsectSet459 (19 AQ): a few common species
sit where the reference recordings put them; the set is too heterogeneous for temperature responses or as a reference.
Tables: runs/inat/measurements.csv, inat_vs_types.csv, inat_species_table.csv; points exploratory/inat_points.csv.
