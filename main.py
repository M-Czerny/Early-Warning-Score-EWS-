"""
SpO2 estimation from a multi-LED PPG wristband.  Run:  python main.py

Steps:
  1. Load the wristband (PPG) and reference oximeter (SpO2) recordings.
  2. Line up the two devices in time.
  3. Remove samples where the reference SpO2 is unreliable.
  4. Smooth the PPG signals (Savitzky-Golay filter).
  5. Cut the recordings into windows and compute features per window.
  6. Drop low-quality windows (Signal Quality Index, SQI).
  7. Train and test SpO2 models (Leave-One-Subject-Out cross-validation).
  8. Smooth the predictions over time (Kalman filter).
  9. Save plots and a CSV of all window features to plots/<timestamp>/.

All settings are in the block below.
"""

import copy
import os
from typing import Any, Dict

from calibration import (apply_kalman_to_all_combos, best_rmse, calibrate_all_combos,
                         grid_search_model_params, summarise, summarise_combos,
                         sweep_kalman_params)
from data_extraction import extract_windows
from export import export_windows_csv
from plotting import plot_all
from preprocessing import apply_savgol, remove_artifacts, synchronise
from signal_acquisition import load_all_subjects
from sqi import (apply_baseline_sqi, apply_composite_sqi, apply_sqi,
                 composite_percentile_threshold, fit_composite_sqi_scaler,
                 percentile_thresholds, report_hr_agreement, report_percentile_overlap)

# =============================================================================
# Settings
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prototype_data')

# --- Artifact removal (based on the reference SpO2) --------------------------
MIN_SPO2  = 90.0   # drop samples where SpO2 (%) is at or below this
SPO2_DROP = 3.0    # drop samples where SpO2 falls by this many % in one step

# --- Windows -----------------------------------------------------------------
WINDOW_S = 10.0    # window length (s)
OVERLAP  = 0.75    # overlap between consecutive windows (0 = none)

# --- Filters -----------------------------------------------------------------
BP_LOW  = 0.5      # Hz. Pulse (AC) band is BP_LOW..BP_HIGH, baseline (DC) is below BP_LOW
BP_HIGH = 6.0

USE_SAVGOL       = True   # smooth the PPG channels before feature extraction
SAVGOL_WINDOW    = 15     # samples, must be odd (15 samples = 300 ms at 50 Hz)
SAVGOL_POLYORDER = 2

# --- Signal quality (SQI) filter ---------------------------------------------
# How the quality limits are set (details in sqi.py):
#   'baseline'   compare each window with the first SQI_BASELINE_S seconds of
#                its own recording (default)
#   'fixed'      use the fixed limits from the literature below
#   'percentile' drop the worst SQI_REJECT_PCT % of windows on each metric
SQI_THRESHOLD_MODE       = 'baseline'
SQI_BASELINE_S           = 60.0   # length of the calibration phase (s)
SQI_BASELINE_K           = 3.0    # how many robust SDs a value may move (3 = lenient)
SQI_BASELINE_INDEPENDENT = False  # True: calibrate on non-overlapping windows only
SQI_REJECT_PCT           = 6      # for 'percentile' mode
SQI_IR_ONLY_METRICS      = False  # True: check per-LED metrics on the IR channel only

# Use one weighted quality score instead of the per-metric limits.
USE_COMPOSITE_SQI = False
SQI_COMPOSITE_MIN = 0.25          # keep windows with a score (0..1) of at least this

# Fixed limits for 'fixed' mode (None = no limit).  They come from the papers
# and were found on other sensors, so they are a reference point only.
SQI_FIXED_LIMITS = dict(
    skewness_min = 0.0,     # Elgendi 2016
    kurtosis_max = 3.5,     # Selvaraj 2011
    snr_min      = 0.0,     # de Haan 2013: below 0 dB the pulse rate is unreliable
    entropy_min  = 0.8,     # Selvaraj 2011
    acc_std_max  = 100,     # movement; a still window is about 50-80
)

# --- Calibration -------------------------------------------------------------
# LED combinations to test.  None = all 11:
#   'Red-IR', 'Red-Green', 'Red-Blue', 'IR-Green', 'IR-Blue', 'Green-Blue',
#   'Red-IR-Green', 'Red-IR-Blue', 'Red-Green-Blue', 'IR-Green-Blue', 'All-4LEDs'
LED_COMBOS_INCLUDE = None

# Model inputs:
#   'R'           ratio of ratios
#   'AC_DC'       AC and DC of each LED
#   'AC_DC_RATIO' AC/DC of each LED
FEATURE_MODE = 'R'

RANDOM_SEED = 42

# SVR and GPR settings per feature mode, found with TUNE_MODEL_PARAMS.
#   svr_C       higher = follows the training data more closely
#   svr_epsilon errors smaller than this (% SpO2) are ignored
#   svr_gamma   width of the SVR kernel
#   gpr_alpha   higher = smoother fit
MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    'R':           dict(svr_C=0.1,   svr_epsilon=1.0,  svr_gamma='scale', gpr_alpha=2.0),
    'AC_DC':       dict(svr_C=100.0, svr_epsilon=0.01, svr_gamma='scale', gpr_alpha=0.01),
    'AC_DC_RATIO': dict(svr_C=100.0, svr_epsilon=0.01, svr_gamma='scale', gpr_alpha=2.0),
}
GPR_N_RESTARTS = 5   # more = better GPR fit, but slower

# --- Kalman smoothing of the predictions -------------------------------------
USE_KALMAN   = True
KALMAN_Q     = 0.1   # how much SpO2 may change from one window to the next
KALMAN_R     = 0.5   # how noisy a single prediction is
KALMAN_SIGMA = 2.0   # ignore predictions further than this many SDs away

# --- Extra runs --------------------------------------------------------------
RUN_BASELINE = False   # also calibrate on raw windows (no artifact removal, no SQI)
EXPORT_CSV   = True    # write window_features.csv next to the plots

# --- Tuning (print a table of results, then continue as normal) --------------
# Turn one on, run once, copy the best values into the settings above.
TUNE_MODEL_PARAMS = False   # SVR (C, epsilon) and GPR alpha
TUNE_KALMAN       = False   # Kalman Q, R and sigma
TUNE_SAVGOL       = False   # Savitzky-Golay window and polynomial order
TUNE_COMPOSITE    = False   # composite SQI cut-off; prints and stops

TUNE_SVR_C        = [0.1, 1.0, 10.0, 100.0, 500.0]
TUNE_SVR_EPSILON  = [0.01, 0.1, 0.5, 1.0]
TUNE_GPR_ALPHA    = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
TUNE_KALMAN_Q     = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
TUNE_KALMAN_R     = [0.5, 1.0, 2.0, 5.0, 10.0]
TUNE_KALMAN_SIGMA = [2.0, 3.0, 4.0, 5.0]
TUNE_SAVGOL_WIN   = [7, 11, 15, 21, 31]
TUNE_SAVGOL_POLY  = [2, 3]
# Picking the cut-off with the lowest RMSE makes that RMSE look better than it
# really is, so treat this table as a rough guide only.
TUNE_COMPOSITE_MIN = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]


# =============================================================================
# Pipeline
# =============================================================================

def calibrate(windows):
    """LOSO calibration with the settings above."""
    return calibrate_all_combos(
        windows, include_combos=LED_COMBOS_INCLUDE, feature_mode=FEATURE_MODE,
        random_seed=RANDOM_SEED, gpr_n_restarts=GPR_N_RESTARTS,
        **MODEL_PARAMS[FEATURE_MODE],
    )


def extract(records):
    return extract_windows(records, window_s=WINDOW_S, overlap=OVERLAP,
                           bp_low=BP_LOW, bp_high=BP_HIGH)


def sqi_limits(windows) -> dict:
    """The per-metric limits for this run ({} in baseline mode: limits are per recording)."""
    if SQI_THRESHOLD_MODE == 'baseline':
        return {}
    if SQI_THRESHOLD_MODE == 'fixed':
        return dict(SQI_FIXED_LIMITS)
    if SQI_THRESHOLD_MODE == 'percentile':
        return percentile_thresholds(windows, SQI_REJECT_PCT, SQI_IR_ONLY_METRICS)
    raise ValueError("SQI_THRESHOLD_MODE must be 'baseline', 'fixed' or "
                     f"'percentile', got {SQI_THRESHOLD_MODE!r}")


def composite_min(windows, scaler) -> float:
    if SQI_THRESHOLD_MODE == 'percentile':
        return composite_percentile_threshold(windows, SQI_REJECT_PCT, scaler)
    return SQI_COMPOSITE_MIN


def sqi_filter(windows):
    """Apply the SQI filter chosen in the settings.  Returns (kept, limits used)."""
    if USE_COMPOSITE_SQI:
        scaler = fit_composite_sqi_scaler(windows)
        cut = composite_min(windows, scaler)
        return apply_composite_sqi(windows, cut, scaler), {'composite_min': cut}
    if SQI_THRESHOLD_MODE == 'baseline':
        kept = apply_baseline_sqi(windows, baseline_s=SQI_BASELINE_S, window_s=WINDOW_S,
                                  k=SQI_BASELINE_K, ir_only=SQI_IR_ONLY_METRICS,
                                  independent=SQI_BASELINE_INDEPENDENT)
        return kept, {}
    limits = sqi_limits(windows)
    return apply_sqi(windows, limits, ir_only=SQI_IR_ONLY_METRICS), limits


def first_combo_rmse(windows) -> float:
    """Best-model RMSE of the first LED combination (NaN if too few windows)."""
    if len(windows) < 5:
        return float('nan')
    return best_rmse(next(iter(calibrate(windows).values())))


def tune_savgol(data_before_savgol) -> None:
    """Rerun extraction, SQI and calibration for every Savitzky-Golay setting."""
    print("\n[SavGol tuning]  window × polyorder")
    print(f"  {'window':>8}  {'polyorder':>10}  {'best_RMSE':>10}")
    for win in TUNE_SAVGOL_WIN:
        for poly in TUNE_SAVGOL_POLY:
            if poly >= win:
                continue
            d = copy.deepcopy(data_before_savgol)
            apply_savgol(d, window_length=win, polyorder=poly)
            wins, _ = sqi_filter(extract(d))
            print(f"  {win:>8}  {poly:>10}  {first_combo_rmse(wins):>10.4f}")


def tune_composite(windows) -> None:
    """Print the best RMSE for every composite SQI cut-off."""
    scaler = fit_composite_sqi_scaler(windows)
    print(f"\n[Composite tuning]\n  {'cut-off':>10}  {'n_kept':>7}  {'best_RMSE':>10}")
    for cut in TUNE_COMPOSITE_MIN:
        kept = apply_composite_sqi(windows, cut, scaler)
        print(f"  {cut:>10.3f}  {len(kept):>7}  {first_combo_rmse(kept):>10.4f}")


def main():
    print("Loading PPG and SpO2 data...")
    data = load_all_subjects(DATA_DIR)

    print("Synchronising signals...")
    data = synchronise(data)

    # Optional: calibrate on raw windows, to compare with the full pipeline.
    combo_results_raw = results_raw = None
    if RUN_BASELINE:
        print("\nCalibrating on raw windows (no artifact removal, no SQI)...")
        combo_results_raw = calibrate(extract(data))
        if USE_KALMAN:
            apply_kalman_to_all_combos(combo_results_raw, Q=KALMAN_Q, R=KALMAN_R,
                                       reject_sigma=KALMAN_SIGMA)
        results_raw = next(iter(combo_results_raw.values()))
        summarise(results_raw)

    print("\nRemoving artifacts...")
    data = remove_artifacts(data, min_spo2=MIN_SPO2, spo2_drop=SPO2_DROP)
    data_before_savgol = copy.deepcopy(data) if TUNE_SAVGOL else None

    if USE_SAVGOL:
        print("Applying Savitzky-Golay filter to PPG channels...")
        apply_savgol(data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

    print(f"\n  {'Recording':<30}  {'PPG':>8}  {'SpO2':>8}  {'removed':>8}")
    for rec in data:
        label = rec['subject'] + (f" / {rec['episode']}" if rec['episode'] else "")
        n_ppg = len(rec['ppg']['ir'])
        pct = rec['n_ppg_removed'] / (n_ppg + rec['n_ppg_removed']) * 100
        print(f"  {label:<30}  {n_ppg:>8}  {len(rec['spo2']['spo2']):>8}  {pct:>7.1f}%")

    print("\nExtracting windows and features...")
    all_windows = extract(data)
    print(f"  Extracted {len(all_windows)} windows total")

    if TUNE_COMPOSITE:
        tune_composite(all_windows)
        return None, None

    print("\nApplying SQI filter...")
    kept_windows, limits = sqi_filter(all_windows)
    if SQI_THRESHOLD_MODE == 'percentile' and not USE_COMPOSITE_SQI:
        report_percentile_overlap(all_windows, limits, SQI_IR_ONLY_METRICS)
    report_hr_agreement(all_windows, kept_windows)
    print(f"\nDone — {len(kept_windows)} windows ready for analysis.")

    if TUNE_MODEL_PARAMS:
        grid_search_model_params(
            kept_windows, TUNE_SVR_C, TUNE_SVR_EPSILON, TUNE_GPR_ALPHA,
            include_combos=LED_COMBOS_INCLUDE, feature_mode=FEATURE_MODE,
            random_seed=RANDOM_SEED, gpr_n_restarts=GPR_N_RESTARTS,
            svr_gamma=MODEL_PARAMS[FEATURE_MODE]['svr_gamma'],
        )
    if TUNE_SAVGOL:
        tune_savgol(data_before_savgol)

    print("\nRunning calibration (Leave-One-Subject-Out)...")
    combo_results = calibrate(kept_windows)
    if TUNE_KALMAN:
        sweep_kalman_params(combo_results, TUNE_KALMAN_Q, TUNE_KALMAN_R, TUNE_KALMAN_SIGMA)
    if USE_KALMAN:
        apply_kalman_to_all_combos(combo_results, Q=KALMAN_Q, R=KALMAN_R,
                                   reject_sigma=KALMAN_SIGMA)
    results = next(iter(combo_results.values()))
    summarise(results)
    summarise_combos(combo_results)

    out_dir = plot_all(
        records=data, all_windows=all_windows, kept_windows=kept_windows,
        results=results, results_raw=results_raw, sqi_limits=limits,
        combo_results=combo_results, combo_results_raw=combo_results_raw,
        window_s=WINDOW_S, bp_low=BP_LOW, bp_high=BP_HIGH,
    )

    if EXPORT_CSV:
        print('\nExporting window features...')
        export_windows_csv(all_windows, kept_windows, out_dir)

    return kept_windows, results


if __name__ == '__main__':
    main()
