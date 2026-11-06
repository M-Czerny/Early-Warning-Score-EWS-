# EWS: SpO2 estimation from a wearable PPG sensor

This project estimates blood oxygen saturation (SpO2) from a wristband
prototype with four LEDs (red, infrared, green, blue). A reference pulse
oximeter is worn at the same time and provides the true SpO2 values used to
train and test the models.

## Quick start

```bash
python -m venv EWSvenv
EWSvenv\Scripts\activate          # on macOS/Linux: source EWSvenv/bin/activate
pip install -r requirements.txt
python main.py                    # takes about 5 minutes
python test_sqi.py                # quick self-checks
python test_calibration.py
```

The console shows the error (RMSE, MAE, R², bias) of every model. Plots and a
CSV of all window features are saved in `plots/<date_time>/`.

## How it works

```
load → synchronise → remove artifacts → smooth → windows + features
     → quality filter (SQI) → train/test models (LOSO) → Kalman smoothing → plots + CSV
```

| Step | File | What happens |
|------|------|--------------|
| 1. Load | `signal_acquisition.py` | Reads the two files of each recording from `prototype_data/`. |
| 2. Synchronise | `preprocessing.py` | The two devices start at different times. The earlier recording is cut so both start together. |
| 3. Remove artifacts | `preprocessing.py` | Drops moments where the reference SpO2 is at or below 90 %, or falls by 3 % or more in one step (probe slipped). |
| 4. Smooth | `preprocessing.py` | Savitzky-Golay filter on the four LED channels to remove fast noise. |
| 5. Features | `data_extraction.py` | Cuts each recording into 10 s windows (75 % overlap) and computes AC, DC, ratio of ratios R and quality metrics for every LED. |
| 6. Quality filter | `sqi.py` | Drops windows with a poor signal (see below). |
| 7. Calibrate | `calibration.py` | Trains five models (Linear, Quadratic, SVR, GPR, Decision Tree) with Leave-One-Subject-Out cross-validation. |
| 8. Kalman | `calibration.py` | Smooths each subject's predictions over time and ignores sudden jumps. |
| 9. Output | `plotting.py`, `export.py` | Saves the plots and `window_features.csv`. |

### Features

For each window and LED:

- **AC**: size of the pulse (standard deviation of the 0.5 to 6 Hz bandpassed signal)
- **DC**: baseline light level (mean of the signal below 0.5 Hz)
- **R** = (AC_a / DC_a) / (AC_b / DC_b) for every LED pair. This is the classic
  pulse oximetry feature. Each pair is tried both ways round (for example
  Red/IR and IR/Red) and the better direction is kept.

`FEATURE_MODE` in `main.py` chooses what the models get: `'R'`, `'AC_DC'`
(AC and DC of each LED) or `'AC_DC_RATIO'` (AC/DC of each LED).

### Quality filter (SQI)

Every window gets seven quality metrics, each taken from the literature:

| Metric | Source | Poor quality when |
|--------|--------|-------------------|
| Perfusion | Elgendi 2016 | too low or too high |
| Skewness | Elgendi 2016 | too low |
| Kurtosis | Selvaraj 2011 | too high |
| SNR around the heart rate | de Haan 2013 | too low |
| Shannon entropy | Selvaraj 2011 | too low |
| Relative power in the pulse band | Elgendi 2016 | too low |
| Accelerometer movement | | too high |

`SQI_THRESHOLD_MODE` in `main.py` chooses how the limits are set:

- `'baseline'` (default): the first 60 s of each recording act as a
  calibration phase. A window is dropped when a metric moves more than 3 robust
  standard deviations (median and MAD, Leys 2013) away from that person's own
  calibration values. This works better than fixed limits because the metrics
  differ a lot between people even when the signal is clean.
- `'fixed'`: fixed limits from the papers (`SQI_FIXED_LIMITS`).
- `'percentile'`: drop the worst `SQI_REJECT_PCT` % of windows on each metric.
- `USE_COMPOSITE_SQI = True`: one weighted score instead of seven limits.

As a check, each run prints how often the heart rate of kept and of dropped
windows agrees with the reference oximeter. The filter never sees the
reference, so this is an independent test.

### Leave-One-Subject-Out (LOSO)

Each subject in turn is held out. The models are trained on all the other
subjects and tested on the held-out one. This shows how well a model works on a
person it has never seen.

## Settings

All settings are at the top of `main.py`. The most useful ones:

| Setting | Default | Meaning |
|---------|---------|---------|
| `WINDOW_S`, `OVERLAP` | 10 s, 0.75 | window length and overlap |
| `SQI_THRESHOLD_MODE` | `'baseline'` | how the quality limits are set |
| `LED_COMBOS_INCLUDE` | `None` (all 11) | which LED combinations to test, e.g. `['Red-IR']` |
| `FEATURE_MODE` | `'R'` | model inputs |
| `USE_SAVGOL`, `USE_KALMAN` | `True` | turn the smoothing steps on or off |
| `RUN_BASELINE` | `False` | also train on raw windows, for comparison |
| `TUNE_*` | `False` | print tuning tables for the model, Kalman, Savitzky-Golay or composite settings |

## Data layout

```
prototype_data/
  subject5/
    1775057212809.txt               wristband: IR, Red, Green, Blue, ACCx/y/z at 50 Hz
                                    (file name = start time in Unix ms)
    opensignals_<...>.txt           reference oximeter: Red, IR, SpO2 at 100 Hz
  subject9/
    episode 2/                      a subject can also have one folder per recording
      ...
```

## Output

Each run creates `plots/<date_time>/` with:

| File | Content |
|------|---------|
| `01_signals_<subject>.png` | raw signals, removed artifacts and dropped windows |
| `02_predictions_<subject>.png` | predicted vs reference SpO2 |
| `03_bland_altman.png` | agreement of the best model with the reference |
| `04_bias_per_subject.png` | mean error per subject and model |
| `05_sqi_overview.png` | every quality metric for every window |
| `06_rmse_<combo>.png` | RMSE of each model for one LED combination |
| `07_residuals.png` | distribution of prediction errors |
| `window_features.csv` | one row per window with all features and `passed_sqi` |

## Files

| File | Role |
|------|------|
| `main.py` | settings and the pipeline, start here |
| `signal_acquisition.py` | find and read the data files |
| `preprocessing.py` | synchronise, remove artifacts, smooth |
| `data_extraction.py` | windows, features and quality metrics |
| `sqi.py` | the quality filters |
| `calibration.py` | models, LOSO cross-validation, Kalman filter, tuning |
| `plotting.py` | plots |
| `export.py` | CSV export |
| `test_sqi.py`, `test_calibration.py` | self-checks, run with `python <file>` |

## References

- Elgendi M. (2016). Optimal signal quality index for photoplethysmogram signals. *Bioengineering* 3(4):21.
- Selvaraj N. et al. (2011). Statistical approach for the detection of motion/noise artifacts in photoplethysmogram. *IEEE EMBC*.
- de Haan G., Jeanne V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Trans. Biomed. Eng.* 60(10).
- Leys C. et al. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *J. Exp. Soc. Psychol.* 49(4):764-766.
