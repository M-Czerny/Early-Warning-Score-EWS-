from signal_acquisition import load_all_subjects
from preprocessing import synchronise, remove_artifacts, apply_savgol
from data_extraction import extract_windows, apply_sqi
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
BP_LOW    = 0.5    # low cut-off  (also used as lowpass cut-off for DC)
BP_HIGH   = 6.0    # high cut-off

# --- SQI thresholds (None = no bound on that side) --------------------------
# Note: SNR here is computed in the cardiac band vs. residual; because the PPG
#       DC component is very large, values are typically negative (-20 to +8 dB).
#       Set snr_min conservatively or leave as None until signal characteristics
#       are better understood.
#SQI_PI_MIN           = 0.001   # minimum perfusion index
SQI_PI_MIN           = 0.01499426068129607 # youden-threshold
SQI_PI_MAX           = None
SQI_SKEWNESS_MIN     =  -2.0
SQI_SKEWNESS_MAX     =  2.0
SQI_KURTOSIS_MIN     = -2.0
SQI_KURTOSIS_MAX     = 10.0    # p99 ≈ 14; 10 retains ~95 % of windows
#SQI_SNR_MIN          = -20.0   # dB; typical range is −22 to +8 dB
SQI_SNR_MIN          = -11.319350470854252 # youden-threshold
SQI_SNR_MAX          = None
SQI_ENTROPY_MIN      = None
SQI_ENTROPY_MAX      =  4.5    # bits  (lower = more concentrated spectrum)
SQI_REL_POWER_MIN    =  0.05   # fraction; loosened — p5 IR ≈ 0.05, p5 red ≈ 0.03
SQI_REL_POWER_MAX    = None
#SQI_ACC_STD_MAX      = 100    # std-dev of acceleration L2-norm; quiet window ≈ 100–220
SQI_ACC_STD_MAX      = 87.35546614091636 # youden-threshold

# --- Calibration parameters -------------------------------------------------
RANDOM_SEED = 42   # seed for training-set shuffle in LOSO cross-validation

# --- Savitzky-Golay denoising (applied to optical PPG channels) -------------
USE_SAVGOL       = True
SAVGOL_WINDOW    = 15    # samples; must be odd — at 50 Hz, 9 samples ≈ 180 ms
SAVGOL_POLYORDER = 3    # polynomial order (quadratic, as in Renesas OB1203 note)

# --- Zero-order Kalman filter on SpO2 predictions ---------------------------
USE_KALMAN      = True
KALMAN_Q        = 0.5   # process noise variance (expected SpO2 drift per window)
KALMAN_R        = 2.0   # measurement noise variance (regressor uncertainty)
KALMAN_SIGMA    = 3.0   # outlier gate: reject if deviation > this many sigma


def main():
    print("Loading PPG and SpO2 data...")
    data = load_all_subjects(DATA_DIR)

    print("Synchronising signals...")
    data = synchronise(data)

    # --- Baseline: calibrate on raw (no artifact removal, no SQI) windows ----
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
    combo_results_raw = calibrate_all_combos(raw_windows, random_seed=RANDOM_SEED)
    if USE_KALMAN:
        apply_kalman_to_all_combos(combo_results_raw,
                                   Q=KALMAN_Q, R=KALMAN_R, reject_sigma=KALMAN_SIGMA)
    results_raw = combo_results_raw['Red-IR']
    summarise(results_raw)

    # --- Full pipeline -------------------------------------------------------
    print("\nRemoving artifacts...")
    data = remove_artifacts(
        data,
        min_spo2=MIN_SPO2,
        spo2_drop=SPO2_DROP,
    )

    if USE_SAVGOL:
        print("Applying Savitzky-Golay filter to PPG channels...")
        apply_savgol(data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

    print(f"\n  {'Recording':<30}  {'PPG':>8}  {'SpO2':>8}  {'removed':>8}")
    for rec in data:
        label = rec['subject'] + (f" / {rec['episode']}" if rec['episode'] else "")
        pct = rec['n_ppg_removed'] / (len(rec['ppg']['ir']) + rec['n_ppg_removed']) * 100
        print(f"  {label:<30}  {len(rec['ppg']['ir']):>8}  "
              f"{len(rec['spo2']['spo2']):>8}  {pct:>7.1f}%")

    print("\nExtracting windows and features...")
    all_windows = extract_windows(
        data,
        window_s=WINDOW_S,
        overlap=OVERLAP,
        bp_low=BP_LOW,
        bp_high=BP_HIGH,
    )
    print(f"  Extracted {len(all_windows)} windows total")

    print("\nApplying SQI filter...")
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

    print("\nRunning calibration (all LED combinations, Leave-One-Subject-Out)...")
    combo_results = calibrate_all_combos(kept_windows, random_seed=RANDOM_SEED)
    if USE_KALMAN:
        apply_kalman_to_all_combos(combo_results,
                                   Q=KALMAN_Q, R=KALMAN_R, reject_sigma=KALMAN_SIGMA)
    results = combo_results['Red-IR']
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
