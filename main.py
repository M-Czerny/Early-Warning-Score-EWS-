"""
Main pipeline for wearable SpO2 estimation from multi-LED PPG signals.

Steps:
  1. Load raw PPG (4 LEDs + accelerometer) and reference SpO2 recordings.
  2. Time-synchronise the two devices.
  3. (Optional) Savitzky-Golay denoising of the optical channels.
  4. Remove samples where the reference SpO2 is unreliable.
  5. Slide a window over each recording and extract features per window.
  6. Filter out low-quality windows with the Signal Quality Index (SQI).
  7. Train and evaluate SpO2 regressors (Leave-One-Subject-Out cross-validation).
  8. Smooth predictions with a Kalman filter.
  9. Save diagnostic plots.
"""

from signal_acquisition import load_all_subjects
from preprocessing import synchronise, remove_artifacts, apply_savgol
from data_extraction import extract_windows, apply_sqi, apply_composite_sqi
from calibration import (summarise,
                         calibrate_all_combos, summarise_combos,
                         apply_kalman_to_all_combos)
from plotting import plot_all

DATA_DIR = r'C:\EWS\prototype_data'

# --- Artifact-removal thresholds --------------------------------------------
MIN_SPO2  = 90.0  # drop samples where SpO2 (%) is at or below this
SPO2_DROP = 3.0   # drop samples where SpO2 drops this many % in one step

# --- Window parameters ------------------------------------------------------
WINDOW_S  = 10.0   # window length in seconds
OVERLAP   = 0.75   # fraction of overlap between successive windows (0 – <1)

# --- Bandpass / lowpass cut-off frequencies (Hz) ----------------------------
# These split each PPG channel into its pulsatile AC part (cardiac band) and
# its slowly-varying DC baseline.
BP_LOW    = 0.5    # low cut-off  (also used as lowpass cut-off for DC)
BP_HIGH   = 6.0    # high cut-off

# --- SQI mode ---------------------------------------------------------------
# True  → use a single composite SQI score (weighted combination of metrics)
# False → use individual per-metric thresholds (see below)
USE_COMPOSITE_SQI    = False
#SQI_COMPOSITE_MIN    = 0.25   # normalised score in [0, 1]; retains top (1-min)*100 %
SQI_COMPOSITE_MIN    = 0.4500400589599263  # youden-threshold

# --- SQI thresholds (None = no bound on that side) --------------------------
# Note: SNR here is computed in the cardiac band vs. residual; because the PPG
#       DC component is very large, values are typically negative (-20 to +8 dB).
#       Set snr_min conservatively or leave as None until signal characteristics
#       are better understood.
SQI_PI_MIN           = 0.001   # minimum perfusion index (AC/DC of IR channel)
#SQI_PI_MIN           = 0.01499426068129607 # youden-threshold
SQI_PI_MAX           = None
#SQI_SKEWNESS_MIN     =  -2.0   # waveform shape bounds (bandpass signal)
#SQI_SKEWNESS_MAX     =  2.0
SQI_SKEWNESS_MIN     =  -1.8838607533911418 # youden-threshold
SQI_SKEWNESS_MAX     =  1.8838607533911418
SQI_KURTOSIS_MIN     = -2.0
SQI_KURTOSIS_MAX     = 10.0    # p99 ≈ 14; 10 retains ~95 % of windows
#SQI_KURTOSIS_MIN     = -1.7230888300092866
#SQI_KURTOSIS_MAX     = 6.929752895833283
SQI_SNR_MIN          = -20.0   # dB; typical range is −22 to +8 dB
#SQI_SNR_MIN          = -11.319350470854252 # youden-threshold
SQI_SNR_MAX          = None
SQI_ENTROPY_MIN      = None 
SQI_ENTROPY_MAX      =  4.5    # bits  (lower = more concentrated spectrum = better)
#SQI_ENTROPY_MAX      =  3.037512586042237 # youden-threshold
SQI_REL_POWER_MIN    =  0.05   # fraction of power in cardiac band; p5 IR ≈ 0.05
#SQI_REL_POWER_MIN    =  0.3842735801590486 # youden-threshold
SQI_REL_POWER_MAX    = None
#SQI_ACC_STD_MAX      = 100    # std-dev of acceleration L2-norm; quiet window ≈ 100–220
SQI_ACC_STD_MAX      = 87.35546614091636  # youden-threshold; rejects motion artefacts

# --- Baseline calibration (raw windows, no artifact removal / SQI) ----------
# Enable to compare the full pipeline against an unfiltered baseline.
RUN_BASELINE = False

# --- LED combination selection ----------------------------------------------
# None → run all 11 combinations defined in calibration.LED_COMBOS
# List → run only the named subsets, e.g. ['Red-IR', 'IR-Green', 'All-4LEDs']
# Valid names: 'Red-IR', 'Red-Green', 'Red-Blue', 'IR-Green', 'IR-Blue',
#              'Green-Blue', 'Red-IR-Green', 'Red-IR-Blue', 'Red-Green-Blue',
#              'IR-Green-Blue', 'All-4LEDs'
LED_COMBOS_INCLUDE = ['Red-IR-Green']

# --- Feature mode -----------------------------------------------------------
# False (default) → ratio-of-ratios R values as regressor features
# True            → raw AC (bandpass std) and DC (lowpass mean) per LED channel
USE_AC_DC = True

# --- Calibration parameters -------------------------------------------------
RANDOM_SEED = 42   # seed for training-set shuffle in LOSO cross-validation

# --- Savitzky-Golay denoising (applied to optical PPG channels) -------------
# Smooths high-frequency noise while preserving waveform shape and amplitude.
USE_SAVGOL       = True
SAVGOL_WINDOW    = 15    # samples; must be odd — at 50 Hz, 15 samples = 300 ms
SAVGOL_POLYORDER = 3     # polynomial order (cubic)

# --- Zero-order Kalman filter on SpO2 predictions ---------------------------
# Smooths the per-window SpO2 predictions over time and rejects sudden spikes.
USE_KALMAN      = True
KALMAN_Q        = 0.5   # process noise variance (expected SpO2 drift per window)
KALMAN_R        = 2.0   # measurement noise variance (regressor uncertainty)
KALMAN_SIGMA    = 3.0   # outlier gate: reject prediction if it deviates > this many sigma


def main():
    # Step 1 — Load raw sensor data from all subject/episode folders.
    print("Loading PPG and SpO2 data...")
    data = load_all_subjects(DATA_DIR)

    # Step 2 — Align PPG and SpO2 time axes (the two devices start independently).
    print("Synchronising signals...")
    data = synchronise(data)

    # --- Optional baseline: calibrate on unfiltered windows for comparison ----
    combo_results_raw = None
    results_raw = None
    if RUN_BASELINE:
        print("\nExtracting raw windows for baseline comparison...")
        raw_windows = extract_windows(
            data,
            window_s=WINDOW_S,
            overlap=OVERLAP,
            bp_low=BP_LOW,
            bp_high=BP_HIGH,
        )
        print(f"  Extracted {len(raw_windows)} raw windows")
        print("Running baseline calibration (all LED combinations, no preprocessing)...")
        combo_results_raw = calibrate_all_combos(raw_windows, random_seed=RANDOM_SEED, include_combos=LED_COMBOS_INCLUDE, use_ac_dc=USE_AC_DC)
        if USE_KALMAN:
            apply_kalman_to_all_combos(combo_results_raw,
                                       Q=KALMAN_Q, R=KALMAN_R, reject_sigma=KALMAN_SIGMA)
        _primary = next(iter(combo_results_raw))
        results_raw = combo_results_raw[_primary]
        summarise(results_raw)

    # Step 3 — Remove samples where the reference SpO2 is unreliable
    #          (e.g. sensor loss, patient movement causing a sudden drop).
    print("\nRemoving artifacts...")
    data = remove_artifacts(
        data,
        min_spo2=MIN_SPO2,
        spo2_drop=SPO2_DROP,
    )

    # Step 4 — Smooth optical channels to reduce high-frequency noise.
    if USE_SAVGOL:
        print("Applying Savitzky-Golay filter to PPG channels...")
        apply_savgol(data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

    # Print a per-recording summary of how many samples were removed.
    print(f"\n  {'Recording':<30}  {'PPG':>8}  {'SpO2':>8}  {'removed':>8}")
    for rec in data:
        label = rec['subject'] + (f" / {rec['episode']}" if rec['episode'] else "")
        pct = rec['n_ppg_removed'] / (len(rec['ppg']['ir']) + rec['n_ppg_removed']) * 100
        print(f"  {label:<30}  {len(rec['ppg']['ir']):>8}  "
              f"{len(rec['spo2']['spo2']):>8}  {pct:>7.1f}%")

    # Step 5 — Slide a window over each clean recording and compute features.
    #          Each window yields AC/DC amplitudes, ratio-of-ratios R values,
    #          and spectral quality metrics for all four LED channels.
    print("\nExtracting windows and features...")
    all_windows = extract_windows(
        data,
        window_s=WINDOW_S,
        overlap=OVERLAP,
        bp_low=BP_LOW,
        bp_high=BP_HIGH,
    )
    print(f"  Extracted {len(all_windows)} windows total")

    # Step 6 — Discard windows that are too noisy or motion-corrupted.
    print("\nApplying SQI filter...")
    if USE_COMPOSITE_SQI:
        # Single weighted score — simpler to tune but less transparent.
        kept_windows = apply_composite_sqi(all_windows, composite_min=SQI_COMPOSITE_MIN)
    else:
        # Per-metric thresholds — each criterion is checked independently.
        kept_windows = apply_sqi(
            all_windows,
            pi_min=SQI_PI_MIN,             pi_max=SQI_PI_MAX,
            skewness_min=SQI_SKEWNESS_MIN, skewness_max=SQI_SKEWNESS_MAX,
            kurtosis_min=SQI_KURTOSIS_MIN, kurtosis_max=SQI_KURTOSIS_MAX,
            snr_min=SQI_SNR_MIN,           snr_max=SQI_SNR_MAX,
            entropy_min=SQI_ENTROPY_MIN,   entropy_max=SQI_ENTROPY_MAX,
            rel_power_min=SQI_REL_POWER_MIN, rel_power_max=SQI_REL_POWER_MAX,
            acc_std_max=SQI_ACC_STD_MAX,
        )

    print(f"\nDone — {len(kept_windows)} windows ready for analysis.")

    # Step 7 — Train regressors (Linear, Quadratic, SVR, GPR, DecisionTree)
    #          using Leave-One-Subject-Out cross-validation, then smooth
    #          per-window predictions with a Kalman filter.
    print("\nRunning calibration (all LED combinations, Leave-One-Subject-Out)...")
    combo_results = calibrate_all_combos(kept_windows, random_seed=RANDOM_SEED, include_combos=LED_COMBOS_INCLUDE, use_ac_dc=USE_AC_DC)
    if USE_KALMAN:
        # Step 8 — Post-process: smooth SpO2 predictions over time per subject.
        apply_kalman_to_all_combos(combo_results,
                                   Q=KALMAN_Q, R=KALMAN_R, reject_sigma=KALMAN_SIGMA)
    _primary = next(iter(combo_results))
    results = combo_results[_primary]
    summarise(results)
    summarise_combos(combo_results)

    sqi_params = dict(
        pi_min=SQI_PI_MIN,               pi_max=SQI_PI_MAX,
        skewness_min=SQI_SKEWNESS_MIN,   skewness_max=SQI_SKEWNESS_MAX,
        kurtosis_min=SQI_KURTOSIS_MIN,   kurtosis_max=SQI_KURTOSIS_MAX,
        snr_min=SQI_SNR_MIN,             snr_max=SQI_SNR_MAX,
        entropy_min=SQI_ENTROPY_MIN,     entropy_max=SQI_ENTROPY_MAX,
        rel_power_min=SQI_REL_POWER_MIN, rel_power_max=SQI_REL_POWER_MAX,
        acc_std_max=SQI_ACC_STD_MAX,
        composite_min=SQI_COMPOSITE_MIN if USE_COMPOSITE_SQI else None,
    )

    plot_all(
        records=data,
        all_windows=all_windows,
        kept_windows=kept_windows,
        results=results,
        results_raw=results_raw,
        sqi_params=sqi_params,
        combo_results=combo_results,
        combo_results_raw=combo_results_raw,
        window_s=WINDOW_S,
        bp_low=BP_LOW,
        bp_high=BP_HIGH,
    )

    return kept_windows, results


if __name__ == '__main__':
    main()
