# Data dictionary

Tables in data/. Positions are in Hz; log quantities are natural logs; temperatures are HRRR 2 m air temperature in C; times are local (America/Chicago) unless a column says UTC. Type numbers are the paper's (trill types 1-37, chirp types C1-C8).


## Table1_temperature_slopes.csv

Table 1 of the paper: per trill type the within-singer slopes of log rate and log carrier on temperature (pooled over tracks with >= 4 h and >= 2 C of range), the interquartile range over singers, the carrier/rate ratio, the pooled slopes over all singer-hours, and counts.

Columns: type, F25_Hz, R25_Hz, n_singers, n_hours, T_range_within_median_C, rate_within_pct_per_C, rate_within_median, rate_within_IQR, carrier_within_pct_per_C, carrier_within_median, carrier_within_IQR, carrier_over_rate_within, rate_pooled_pct_per_C, carrier_pooled_pct_per_C


## animal_days_streaks_v9.csv

The same for singer-days, assigned to the night lines afterwards.

Columns: an, recorder, site, night, period, F25, R25, n_hours, streak, across, along, form, F, R, T, harm2


## animal_nights_streaks_v9.csv

One row per singer-night (a night track of >= 24 windows with a fixed-window carrier). an = unit:track id; F, R = median carrier (Hz) and syllable rate (Hz); T = mean HRRR temperature (C); F25, R25 = position brought to 25 C; depth = median modulation depth; streak = cluster id on the trill plane (-1 = none); form = song form from the audio (trill / chirp / mixed); band, placeable, form_channel, contrast, harm2 = song-form test details.

Columns: an, recorder, site, night, F25, R25, streak, type_corrected, form, band, placeable, form_channel, F, R, T, depth, Rc, rc, bw_rel, R_scatter, contrast, harm2


## chirp/animal_nights_streaks_c1.csv

Columns: an, recorder, site, night, F, R, T, F25, R25, depth, streak, type_corrected


## chirp/animal_nights_streaks_c2.csv

Chirp plane: one row per typed chirping singer-night (syllables read in >= 50 % of hours); R is the CHIRP rate; streak = chirp cluster id (-1 after the QC exclusion of streak 13).

Columns: an, recorder, site, night, F, R, T, F25, R25, depth, streak, type_corrected


## chirp/animal_nights_streaks_c2_all.csv

Columns: an, recorder, site, night, F, R, T, F25, R25, depth, streak, type_corrected


## chirp/bout_only_hours.csv

Columns: track, hour, F, Rc, R, depth, n, tid, recorder, site, night, T


## chirp/bout_only_tracks.csv

Chirping tracks with a syllable rate in under 50 % of their hours: the untyped bout-only class (frogs, birds, raspers, unreadable chirpers).

Columns: tid, recorder, site, night, F, Rc, T, depth, n_hours, syl_share


## chirp/chirp_params.csv

Columns: Unnamed: 0, 0


## chirp/chirp_type_map_v1_to_v2.csv

How the 15 chirp types of the first pass (all bout rhythms) map onto the types of the syllable-bearing pass.

Columns: old_id, old_streak, n_old, n_kept, kept_share, new_id, new_streak, share_in_new, F25_old, F25_new


## chirp/excluded_streaks.csv

Chirp streaks dropped from the typed set by quality control, with the reason.

Columns: streak, n, F25, R25, reason


## chirp/paper_ids_chirp.csv

Chirp cluster -> C number (C1-C8), with the chirp-rate slope on temperature.

Columns: streak, paper_id, n, F25, F25_iqr, R25, R25_iqr, F_min, F_max, R_min, R_max, n_sites, n_nights, depth, harmonic_of, Rc_slope_pct_perC


## chirp/streak_hourly_points_all.csv

Chirp plane singer-hours: R = Rc = chirp rate, Rs = syllable rate where read.

Columns: hour, R, F, depth, T, Rc, Rs, rc_frac, an, site, recorder, night, n, streak


## chirp/streak_merges_c2.csv

Columns: 


## chirp/streaks_coarse_c1.csv

Columns: streak, n, F25, F25_iqr, R25, R25_iqr, F_min, F_max, R_min, R_max, n_sites, n_nights, depth, R_perC, F_perC, coupling


## chirp/streaks_coarse_c2.csv

Chirp plane clusters (R25 = chirp rate at 25 C); *_c2_all.csv are the tables before the QC exclusion.

Columns: streak, n, F25, F25_iqr, R25, R25_iqr, F_min, F_max, R_min, R_max, n_sites, n_nights, depth, harmonic_of


## chirp/streaks_coarse_c2_all.csv

Columns: streak, n, F25, F25_iqr, R25, R25_iqr, F_min, F_max, R_min, R_max, n_sites, n_nights, depth, harmonic_of


## chirp/temperature_animal_nights.csv

Columns: an, recorder, site, night, F, R, T, depth, n_hours, period, Rc, rc_frac, type_corrected


## chirp/temperature_season_by_type.csv

Columns: type, R_perC, F_perC


## chirp/weather_hourly_hrrr.csv

Columns: t, T_C, RH, precip_mm, wind_ms


## nov24_nov25_day_hours.csv

Columns: tid, hour, R, F, T, depth, Rc, rc_frac, recorder, site, period, month, ym


## paper_ids_v9.csv

Streak id -> the paper's type number (1-37 are the trill types with >= 40 singer-nights, by carrier at 25 C; higher numbers are the smaller streaks).

Columns: streak, paper_id, F25, R25, n, form


## pulse_shapes_streaks_v9.csv

Columns: streak, n, glide_med, glide_iqr, dur_med, duty_med, bw_med, pulse_iqr_med, dbic, mu_lo, mu_hi, w_min, sep_sd, split


## pulse_shapes_v9.csv

Per singer-night pulse measurements from the loudest 30 s of its deepest window: n_pulses, R, glide_hz, dur_ms, duty, bw_hz, f0, snr_db (clear if >= 6).

Columns: status, n_pulses, R, glide_hz, glide_iqr, dur_ms, dur_iqr, duty, bw_hz, f0, snr_db, F_fix, t, an


## recorded_day_hours_by_hour.csv

Columns: Unnamed: 0, 0


## recorded_night_hours_by_hour.csv

Columns: Unnamed: 0, 0


## recorded_windows.csv

Per recorder-night the number of 5-minute windows recorded (hours = windows x 300 s).

Columns: recorder, site, night, windows


## reference_tests/anaxipha_walker_funk_2014_vs_types.csv

Walker & Funk's (2014) Anaxipha recordings (from their supplementary tables; not redistributed) tested species by species against the trill lines at matched temperature (Figure 12): best type, share on the line, n.

Columns: species, type, position, n_in_range, n_total, z_across_med, z_along_med, share_on, share_across_on


## reference_tests/inaturalist_recordings_vs_types.csv

The same per recording (observation and sound ids, taxon, state, date, ERA5 temperature, measured carrier and rate, best type, z scores, license).

Columns: obs_id, sound_id, taxon, state, observed, T, F, R, depth, is_chirp, chirp_rate, license, best_type, type_position, z_across, z_along, on_line


## reference_tests/inaturalist_species_vs_types.csv

iNaturalist research-grade Orthoptera recordings from Arkansas and six neighboring states, measured as the SINA clips were and tested against the lines, summarized per species (exploratory; audio not redistributed).

Columns: taxon, n, n_T, on, best, best_share, F_med, R_med, T_med, states


## reference_tests/sina_vs_chirp_lines.csv

The same on the chirp plane for the chirping clips.

Columns: wav, name, species, T, R, F, group, streak, z, z_along, z_tot, R_expected, T_in_range, best, paper_id, in_range


## reference_tests/sina_vs_trill_lines.csv

Every SINA clip with a measured point against every trill line whose rate range covers it: z across, z_along, z_tot, R_expected, best, in_range (species recorded in Arkansas or a neighboring state), paper_id (Figure 13).

Columns: wav, name, species, T, R, F, group, streak, z, z_along, z_tot, R_expected, T_in_range, best, in_range, paper_id


## reference_tests/sina_vs_trill_lines_in_range.csv

The same restricted to in-range species.

Columns: wav, name, species, T, R, F, group, streak, z, z_along, z_tot, R_expected, T_in_range, best, in_range, paper_id


## sina/sina_recordings.csv

Columns: page, name, subfamily, wav, caption, T, kind, n_near, supplement


## sina/sina_reference_chirp.csv

The bout test on every SINA clip: chirp_rate, contrast, duty, is_chirp.

Columns: wav, species, T, F_c, f_lo, f_hi, R, status, chirp_rate, contrast, duty, is_chirp, dur_s


## sina/sina_reference_measurements.csv

Every SINA reference recording run through the window analysis: species, recording temperature, measured carrier (F_c, f_lo, f_hi), syllable rate R, depth, components.

Columns: page, name, subfamily, wav, T, dur_s, sr, n_blobs, n_parts, n_animals, sina_kHz, n_AR, F_c, f_lo, f_hi, R, depth, on_frac, n_components, n_am_harm, matches_page_kHz, components


## sina/sina_regressions.csv

Columns: species_no, page, file, sheet, rate_col, n, rate25, rate_slope_pctC, T_lo, T_hi, name, subfamily, kHz_prose, kHz25, kHz_slope_pctC, n_kHz


## sina/sina_songs.csv

Columns: song_text, song_at_25, rates_per_s, chirp_rates, rate_first, rate_is_at_25, temps_C, kHz, kHz_first, pattern, group, subfamily, page, name


## sina/site_species.csv

Columns: species, observed_by, note


## sina/species_geography.csv

Columns: SpeciesNo, page, Genus, SpecificName, CommonName, FamOrSubfamily, n_AR, n_near


## sina/supplement_species.csv

Columns: page, name, subfamily, group, source, rate25, rate_lo, rate_hi, kHz, pattern, note


## streak_coherence_v9.csv

Columns: streak, n, F25, R25, candidate, pairs, pair_frac, null_frac, incoh_frac, slope_pct_per_C, shared, shared_applies, n_shared_pairs, season_max_pct, n_split, splits, verdict, position


## streak_hourly_points_all.csv

One row per singer-hour of every night track >= 2 h: hour (local), R, F, depth, T, Rc (chirp rate), Rs (syllable rate inside chirps where read), rc_frac (share of windows with a chirp rate), an, site, recorder, night, n windows, streak.

Columns: hour, R, F, depth, T, Rc, rc_frac, an, site, recorder, night, n, streak


## streak_lines_v9.csv

Per streak the least-squares line of log F on log R over its singer-hours (slope, intercept) and its 25 C point: Figure 7C.

Columns: streak, paper_id, slope, intercept, R25, F25, lnR_lo, lnR_hi, n_hours, r


## streaks_coarse_v9.csv

One row per streak (cluster) on the trill plane: n singer-nights, F25 / R25 position at 25 C with interquartile ranges, extent, n_sites, n_nights, depth, harmonic_of (>= 0: a carrier harmonic of another streak), form, n_firm / n_mixed.

Columns: streak, n, F25, F25_iqr, R25, R25_iqr, F_min, F_max, R_min, R_max, n_sites, n_nights, depth, harmonic_of, form, n_firm, n_mixed


## temperature_animal_nights.csv

Columns: an, recorder, site, night, period, F, R, T, depth, n_hours, Rc, rc_frac, type_corrected


## temperature_fits_carrier.csv

The per-type fits behind Figure 19 (within-singer carrier response).

Columns: type, streak, F25, R25, var, value_at_25C, slope_pct_per_C, r, n_hours, T_lo, T_hi


## temperature_fits_rate.csv

The per-type fits behind Figure 18 (within-singer rate response).

Columns: type, streak, F25, R25, var, value_at_25C, slope_pct_per_C, r, n_hours, T_lo, T_hi


## temperature_season_by_type.csv

Per type the temperatures and dates at which it sang.

Columns: type, R_perC, F_perC


## track_form_audio.csv

Per singer-night the song-form test: contrasts of the bout rhythm, duty, rhythm_hz, harm2, band_share, corr_min, form_audio.

Columns: an, status, n_win_track, n_measured, contrast_max, contrast_min, duty_min, duty_max, rhythm_hz, harm2, band_share, corr_min, form_audio, broad


## track_form_audio_day.csv

Columns: an, status, n_win_track, n_measured, contrast_max, contrast_min, duty_min, duty_max, rhythm_hz, harm2, band_share, corr_min, form_audio, broad


## type_catalog_v9.csv

Per streak the year-scale description (season, thermal range, day share, coupling, pulse form, chirp share); candidate-name columns removed.

Columns: type, n, F25, R25, n_sites, n_nights, n_firm, n_mixed, form, n_day_firm, cooccur_excess_frac, shared_response_r, descriptor_splits, season_resid_max_pct, song_form, position, pulse_ms, duty, glide_hz, sweep_hz_per_ms, chirp_share, chirp_rate_hz, hours, first_week, peak_week, last_week, n_weeks, T_cold_limit_p05, T_median, T_min_heard, day_share, day_months, coupling_within, coupling_n_tracks


## type_mix_v9.csv

Columns: streak, week, hours, recorder_periods, hours_per_rp


## type_reference/sample.csv

Per type the sample behind the type reference (key, paper id, form, n sampled, temperature window, n folds); folds_<key>.npy / env_<key>.npy are the aligned fold images and envelopes; windows_<key>.csv the per-window descriptors.

Columns: key, paper_id, form, n_sampled, T_lo, T_hi, T_sample, n_ok, fails


## type_reference/windows_100004.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_100005.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_100007.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_100009.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_100010.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_100021.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_100091.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_100211.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_31011.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_31013.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_31014.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_3201.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_3301.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_34013.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_34014.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_3402.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_3601.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_360123.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_360124.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_360125.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_36015.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_3701.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_4001.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_4301.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_4501.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_45011.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_4601.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_4701.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_4801.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_4901.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5101.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5201.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5301.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5401.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_54012.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5501.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5801.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5802.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_5901.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_590116.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_590117.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_590118.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_6001.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_6102.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_reference/windows_6301.csv

Columns: duty, rise, fall, asym, n_sub, glide_hz, curve_hz, bw_hz, f_off, contrast, key, an, status, t, R, F, site, Rs, R_syl


## type_renumbering_v5.csv

How the type numbers of the final catalog relate to the numbers used before the last split pass (pass2_number -> v5_number), with each type's position and size.

Columns: streak, pass2_number, v5_number, F25, R25, n


## y2024_fixed_rate_lines.csv

Columns: rec, site, month, R, n, share, contrast, units, F_c_med, F_span


## y2024_weather_hourly_hrrr.csv

Columns: t, T_C, RH, precip_mm, wind_ms


## y2025_fixed_rate_lines.csv

Columns: rec, site, month, R, n, share, contrast, units, F_c_med, F_span


## y2025_weather_hourly_hrrr.csv

Columns: t, T_C, RH, precip_mm, wind_ms


## year_chirpers_v9.csv

Columns: streak, F25, R25, hours, chirp_hours, chirp_share, Rc_med, Rc_q25, Rc_q75, per_chirp_med, Rc_slope_pct_per_C, R_slope_pct_per_C, T_med, chirp_months


## year_coupling_tracks_v9.csv

Per track (>= 12 windows): n, R_range, slope of log F on log R within the track, r: Figure 8.

Columns: an, n, R_range, slope, r


## year_day_night_v9.csv

Columns: streak, F25, R25, n_night, n_day, day_share, T_night, T_day, day_months


## year_site_occupancy_v9.csv

Columns: streak, NorthSide, Back9, House, EastCreek, WestSideJunction, Pond, MidCreek, FarNorth, SecretGarden, Copse, EastMeander


## year_thermal_limits_v9.csv

Columns: streak, F25, R25, hours, nights, T_min, T_p05, T_p50, T_p95, T_max, Tnightmin_p05, Tnightmin_min, Tnightmin_med


## tracks_day_all.csv.gz

The same for the day periods.

Columns: unit, track, n_windows, t0, t1, span_h, fill, F_ref, F_fix, F_sd, R, R_sd, R_detr_sd_pct, Rc, depth, n_comp_med, F_fix_sd


## tracks_night_all.csv.gz

Every night track of the year (all lengths) with its unit: span_h, R, Rc, bands and per-track summaries from the linker.

Columns: unit, track, n_windows, t0, t1, span_h, fill, F_ref, F_fix, F_sd, R, R_sd, R_detr_sd_pct, Rc, depth, n_comp_med, F_fix_sd
