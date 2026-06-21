# EWS — Wearable SpO2 Estimation Pipeline

This project estimates blood oxygen saturation (SpO2) from a multi-LED PPG
(photoplethysmography) prototype sensor. A reference pulse oximeter provides
the ground-truth SpO2 values used for training and evaluation.

---

## How the pipeline works

```
Raw data  →  Synchronise  →  Denoise  →  Remove artifacts
          →  Extract windows  →  SQI filter  →  Calibrate  →  Kalman filter  →  Plots
```

### 1. Load data (`signal_acquisition.py`)

The pipeline reads two files per recording session from `prototype_data/`:

| File | Device | Contents |
|------|--------|----------|
| `<timestamp>.txt` | Prototype wristband | IR, Red, Green, Blue PPG + 3-axis accelerometer at **50 Hz** |
| `opensignals_*.txt` | Reference pulse oximeter | SpO2 (%), Red, IR at **100 Hz** |

Both flat (`subject1/<files>`) and nested (`subject1/episode 1/<files>`) folder layouts are supported.

### 2. Synchronise (`preprocessing.py → synchronise`)

The two devices start independently, so their recordings need to be aligned.
The prototype encodes its start time in the filename (Unix milliseconds).
The reference device stores its start time in a JSON header.  The earlier
signal is trimmed to match the start of the later one.

### 3. Denoise (`preprocessing.py → apply_savgol`)

A **Savitzky-Golay filter** smooths the four optical channels.  It fits a
polynomial through a short sliding window, removing high-frequency noise while
preserving the waveform shape better than a plain moving average.

### 4. Remove artifacts (`preprocessing.py → remove_artifacts`)

Reference SpO2 samples are flagged as bad when:
- SpO2 ≤ `MIN_SPO2` (90%) — probe loss or very low saturation
- SpO2 drops ≥ `SPO2_DROP` (3%) in a single step — sudden probe-off event

The same time windows are removed from the PPG data.  Motion artefacts are
handled later at the window level by the SQI filter.

### 5. Extract windows and features (`data_extraction.py → extract_windows`)

A sliding window (default: 10 s, 75% overlap) passes over each clean recording.
For every window the following features are computed for all four LED channels:

- **AC amplitude**: standard deviation of the bandpass-filtered signal (0.5–6 Hz).
  This captures the pulsatile component driven by the cardiac cycle.
- **DC amplitude**: mean of the lowpass-filtered signal (< 0.5 Hz).
  This is the slow-varying tissue absorption baseline.
- **Ratio-of-ratios R** = (AC_A / DC_A) / (AC_B / DC_B) for every LED pair.
  R is the key SpO2 feature: it cancels tissue and path-length differences
  between subjects and relates monotonically to SpO2.
- **Signal quality metrics**: perfusion index (PI = AC/DC of IR), SNR,
  skewness, kurtosis, spectral entropy, relative cardiac-band power.
- **Motion metric**: standard deviation of the accelerometer L2-norm.

### 6. SQI filter (`data_extraction.py → apply_sqi / apply_composite_sqi`)

The **Signal Quality Index** discards windows that are too noisy or corrupted
by motion. Two modes are available:

- **Per-metric** (`USE_COMPOSITE_SQI = False`): each quality metric is checked
  against an independent threshold. A window is discarded if any metric fails.
- **Composite** (`USE_COMPOSITE_SQI = True`): a single weighted score combines
  all metrics; windows below `SQI_COMPOSITE_MIN` are discarded.

### 7. Calibrate (`calibration.py → calibrate_all_combos`)

Five regression models are trained and evaluated using
**Leave-One-Subject-Out (LOSO) cross-validation**:

| Model | Notes |
|-------|-------|
| Linear | Straight-line R → SpO2 mapping |
| Quadratic | Second-order polynomial |
| SVR | Support Vector Regression with RBF kernel |
| GPR | Gaussian Process Regression (also gives uncertainty) |
| DecisionTree | Decision tree, max depth 5 |

In each fold, one subject is held out for testing and all other subjects are
used for training.  This gives a realistic estimate of how well the model
generalises to a new individual.

For LED pair combinations, both the forward R (e.g. Red/IR) and its inverse
(IR/Red) are evaluated and the better-performing direction is kept.

### 8. Kalman filter (`calibration.py → apply_kalman_to_all_combos`)

A **zero-order Kalman filter** is applied per subject to the sequence of
per-window SpO2 predictions. It assumes SpO2 changes slowly between windows
and rejects sudden prediction spikes (outliers) while allowing genuine
sustained changes to pass through after a short lag.

### 9. Plot results (`plotting.py → plot_all`)

Diagnostic plots are saved to a timestamped folder under `plots/`, including:
- Raw and cleaned PPG traces with artifact regions highlighted
- Scatter and time-series plots of predicted vs. reference SpO2
- SQI metric distributions and Bland-Altman agreement plots
- Per-LED-combination RMSE comparison

---

## Configuration

All parameters are set at the top of `main.py`. Key ones:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `WINDOW_S` | 10.0 s | Window length |
| `OVERLAP` | 0.75 | Fraction of overlap between windows |
| `USE_SAVGOL` | True | Apply Savitzky-Golay denoising |
| `USE_COMPOSITE_SQI` | False | Use composite vs. per-metric SQI |
| `LED_COMBOS_INCLUDE` | `['Red-IR']` | Which LED pairs to calibrate |
| `USE_AC_DC` | False | Use raw AC/DC features instead of R values |
| `USE_KALMAN` | True | Apply Kalman smoothing to predictions |
| `RUN_BASELINE` | False | Also run without artifact removal / SQI |

---

## Running

```bash
python main.py
```

Output: RMSE/MAE/R² per model printed to the console, plots saved to `plots/<timestamp>/`.

---

## File overview

| File | Role |
|------|------|
| `main.py` | Entry point; configuration and pipeline orchestration |
| `signal_acquisition.py` | Discover and load raw data files |
| `preprocessing.py` | Synchronise clocks, denoise, remove bad samples |
| `data_extraction.py` | Sliding-window feature extraction and SQI filter |
| `calibration.py` | LOSO cross-validation, Kalman filter, summary tables |
| `plotting.py` | Diagnostic and results plots |
