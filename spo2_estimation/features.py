"""
features.py
~~~~~~~~~~~
Window-based feature extraction for SpO2 estimation.
 
Features produced per window
-----------------------------
Ratio-of-ratios R_{a}_{b}  for every LED combination (a, b).
AC_{a}, DC_{a}, AC_{b}, DC_{b} for each channel in the combinations.
Signal quality features:
    sqi_{a}        – spectral SNR in the cardiac band (0.5–4 Hz)
    skewness_{a}   – AC-signal skewness
    kurtosis_{a}   – AC-signal kurtosis
Motion features (when ACC data present):
    motion_rms     – RMS of band-passed ACC magnitude in the window
    motion_frac    – fraction of samples flagged high-motion
    acc_mag_mean   – mean raw ACC magnitude
"""
 
from typing import Dict, List, Tuple
 
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from scipy.signal import welch
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Windowing helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def sliding_windows(n: int, window_size: int, step_size: int) -> List[Tuple[int, int]]:
    bounds, start = [], 0
    while start + window_size <= n:
        bounds.append((start, start + window_size))
        start += step_size
    return bounds
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Ratio of ratios
# ─────────────────────────────────────────────────────────────────────────────
 
def align_channel_pair(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    max_shift: int = 8,
) -> np.ndarray:
    """
    Align sig_b to sig_a using cross-correlation, constrained to
    +/- max_shift samples.  Returns a shifted copy of sig_b with
    the same length, padded at the edges with edge values.
 
    This corrects for:
      - LED time-multiplexing offsets (deterministic sub-sample shift)
      - Wavelength-dependent tissue scattering (varies by subject)
      - Any residual phase mismatch after bandpass filtering
    """
    n = len(sig_a)
    corr = np.correlate(sig_a - sig_a.mean(), sig_b - sig_b.mean(), mode="full")
    lags  = np.arange(-(n - 1), n)
    valid = np.abs(lags) <= max_shift
    best_lag = int(lags[valid][np.argmax(corr[valid])])
 
    if best_lag == 0:
        return sig_b.copy()
 
    shifted = np.empty_like(sig_b)
    if best_lag > 0:
        shifted[best_lag:] = sig_b[:-best_lag]
        shifted[:best_lag] = sig_b[0]
    else:
        shifted[:best_lag] = sig_b[-best_lag:]
        shifted[best_lag:] = sig_b[-1]
    return shifted
 
 
def compute_ac_dc_ratio(
    window_num_raw:  np.ndarray,
    window_den_raw:  np.ndarray,
    window_num_filt: np.ndarray,
    window_den_filt: np.ndarray,
    align: bool = True,
    max_shift: int = 8,
) -> Tuple[float, float, float, float, float, int]:
    """
    R = (AC_num / DC_num) / (AC_den / DC_den)
 
    DC is the mean of the spike-cleaned (raw) signal.
    AC is the std of the bandpass-filtered signal.
 
    If align=True, sig_b (denominator) is cross-correlation shifted
    to match sig_a (numerator) before AC is computed, removing the
    inter-channel timing offset.  The lag in samples is returned as
    the 6th element of the tuple so callers can log it.
    """
    eps = 1e-12
 
    if align:
        window_den_filt = align_channel_pair(window_num_filt, window_den_filt, max_shift)
        lag = int(np.argmax(
            np.correlate(window_num_filt - window_num_filt.mean(),
                         window_den_filt - window_den_filt.mean(), mode="full")
        ) - (len(window_num_filt) - 1))
    else:
        lag = 0
 
    # DC: use the midpoint of the rolling DC curve so local drift within
    # the window does not bias the ratio.  Falls back to mean if no rolling
    # DC was supplied (window_num_raw == window_den_raw path).
    mid = len(window_num_raw) // 2
    dc_num = float(window_num_raw[mid]) if window_num_raw is not None else 1.0
    dc_den = float(window_den_raw[mid]) if window_den_raw is not None else 1.0
    # Guard against zero / negative DC (e.g. post-baseline-removal residual)
    dc_num = max(dc_num, 1.0)
    dc_den = max(dc_den, 1.0)
    ac_num = np.std(window_num_filt, ddof=1)
    ac_den = np.std(window_den_filt, ddof=1)
    r = (ac_num / (dc_num + eps)) / ((ac_den / (dc_den + eps)) + eps)
    return r, ac_num, dc_num, ac_den, dc_den, lag
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Signal Quality Index
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_sqi(
    sig: np.ndarray,
    fs: float,
    cardiac_low: float = 0.5,
    cardiac_high: float = 4.0,
) -> float:
    """
    Spectral SNR: power in the cardiac band / total power.
    Returns 0–1 (higher = cleaner).
    """
    if len(sig) < 32 or np.all(sig == sig[0]):
        return 0.0
 
    nperseg = min(len(sig), 256)
    f, pxx  = welch(sig, fs=fs, nperseg=nperseg)
    total   = np.trapezoid(pxx, f)
    if total < 1e-20:
        return 0.0
 
    mask    = (f >= cardiac_low) & (f <= cardiac_high)
    cardiac = np.trapezoid(pxx[mask], f[mask])
    return float(np.clip(cardiac / total, 0.0, 1.0))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main feature builder
# ─────────────────────────────────────────────────────────────────────────────
 
def build_window_features(
    df: pd.DataFrame,
    combinations: List[Tuple[str, str]],
    fs: float,
    window_sec: float,
    overlap_sec: float,
    motion_threshold: float = 0.05,
    drop_motion_windows: bool = True,
    align_channels: bool = True,
    max_align_shift: int = 8,
    min_channel_sqi: float = 0.15,
) -> pd.DataFrame:
    """
    Slide a window over *df* and extract features for every LED combination.
 
    Expected columns in df
    ----------------------
    time, spo2
    {c}_raw          – spike-cleaned signal (for DC source)
    {c}_rolling_dc   – rolling-mean DC estimate (preferred for R computation)
    {c}              – baseline-removed + bandpass-filtered signal (for AC)
    motion_energy, high_motion  (optional but recommended)
    acc_mag                     (optional)
 
    Parameters
    ----------
    align_channels  : cross-correlation align channel pairs per window.
    max_align_shift : max shift in samples searched (+/-).
    min_channel_sqi : minimum spectral SNR (0-1) required for both channels
                      in a combination to be included in a window.  Combos
                      where either channel falls below this threshold are
                      skipped for that window — their R column is left NaN.
                      This prevents weak green/blue channels (SQI ~ 0.10)
                      from contributing biased R values to the regression.
    drop_motion_windows : exclude high-motion windows from output.
    """
    w    = int(window_sec * fs)
    step = int((window_sec - overlap_sec) * fs)
    if step <= 0:
        raise ValueError("overlap_sec must be strictly less than window_sec.")
 
    bounds      = sliding_windows(len(df), w, step)
    has_motion  = "high_motion"   in df.columns
    has_acc_mag = "acc_mag"       in df.columns
    has_me      = "motion_energy" in df.columns
 
    rows = []
 
    for i0, i1 in bounds:
        row: Dict[str, float] = {
            "t0":          float(df["time"].iloc[i0]),
            "t1":          float(df["time"].iloc[i1 - 1]),
            "time_center": float(df["time"].iloc[(i0 + i1) // 2]),
            "spo2":        float(np.mean(df["spo2"].iloc[i0:i1])),
        }
 
        # ── Motion features ──────────────────────────────────────────────────
        if has_motion:
            hm_win = float(np.mean(df["high_motion"].iloc[i0:i1]))
            row["motion_frac"] = hm_win
        else:
            hm_win = 0.0
            row["motion_frac"] = 0.0
 
        if has_me:
            me_slice = df["motion_energy"].iloc[i0:i1].to_numpy(dtype=float)
            row["motion_rms"] = float(np.sqrt(np.mean(me_slice ** 2)))
        else:
            row["motion_rms"] = 0.0
 
        if has_acc_mag:
            row["acc_mag_mean"] = float(np.mean(df["acc_mag"].iloc[i0:i1]))
        else:
            row["acc_mag_mean"] = 0.0
 
        row["high_motion_window"] = int(hm_win > 0.5)
 
        # ── Optical features per combination ─────────────────────────────────
        valid_window = True
 
        # Collect per-channel SQI once (not per combo)
        channels_seen = set()
        for a, b in combinations:
            channels_seen.update((a, b))
 
        for ch in channels_seen:
            filt_col = ch
            if filt_col not in df.columns:
                continue
            sig_filt = df[filt_col].iloc[i0:i1].to_numpy(dtype=float)
            if not np.all(np.isfinite(sig_filt)):
                continue
            row[f"sqi_{ch}"]      = compute_sqi(sig_filt, fs)
            row[f"skewness_{ch}"] = float(sp_skew(sig_filt))
            row[f"kurtosis_{ch}"] = float(sp_kurtosis(sig_filt, fisher=True))
 
        for a, b in combinations:
            a_raw        = f"{a}_raw"
            b_raw        = f"{b}_raw"
            a_rolling_dc = f"{a}_rolling_dc"
            b_rolling_dc = f"{b}_rolling_dc"
            a_filt       = a
            b_filt       = b
 
            # Require at minimum the raw and filtered columns
            for col in (a_raw, b_raw, a_filt, b_filt):
                if col not in df.columns:
                    raise KeyError(
                        f"Required column '{col}' not found. "
                        f"Available: {df.columns.tolist()}"
                    )
 
            wa_filt = df[a_filt].iloc[i0:i1].to_numpy(dtype=float)
            wb_filt = df[b_filt].iloc[i0:i1].to_numpy(dtype=float)
            wa_raw  = df[a_raw ].iloc[i0:i1].to_numpy(dtype=float)
            wb_raw  = df[b_raw ].iloc[i0:i1].to_numpy(dtype=float)
 
            if not all(
                np.all(np.isfinite(arr))
                for arr in (wa_raw, wb_raw, wa_filt, wb_filt)
            ):
                valid_window = False
                break
 
            # ── SQI gate ─────────────────────────────────────────────────────
            # Skip this combination for this window if either channel has
            # insufficient spectral quality. The R value would be dominated
            # by noise rather than the cardiac signal.
            sqi_a = row.get(f"sqi_{a}", 1.0)
            sqi_b = row.get(f"sqi_{b}", 1.0)
            if sqi_a < min_channel_sqi or sqi_b < min_channel_sqi:
                row[f"R_{a}_{b}"]    = np.nan
                row[f"AC_{a}"]       = np.nan
                row[f"DC_{a}"]       = np.nan
                row[f"AC_{b}"]       = np.nan
                row[f"DC_{b}"]       = np.nan
                row[f"lag_{a}_{b}"]  = 0
                continue
 
            # ── Rolling DC (preferred) or fallback to raw ─────────────────────
            # The rolling DC array tracks slow drift within the window so the
            # midpoint value gives a local DC estimate rather than a mean that
            # averages over any thermal/pressure ramp in the window.
            if a_rolling_dc in df.columns:
                wa_dc = df[a_rolling_dc].iloc[i0:i1].to_numpy(dtype=float)
            else:
                wa_dc = wa_raw
 
            if b_rolling_dc in df.columns:
                wb_dc = df[b_rolling_dc].iloc[i0:i1].to_numpy(dtype=float)
            else:
                wb_dc = wb_raw
 
            r, ac_a, dc_a, ac_b, dc_b, lag = compute_ac_dc_ratio(
                wa_dc, wb_dc, wa_filt, wb_filt,
                align=align_channels,
                max_shift=max_align_shift,
            )
 
            row[f"R_{a}_{b}"]    = r
            row[f"AC_{a}"]       = ac_a
            row[f"DC_{a}"]       = dc_a
            row[f"AC_{b}"]       = ac_b
            row[f"DC_{b}"]       = dc_b
            row[f"lag_{a}_{b}"]  = lag
 
        if valid_window and np.isfinite(row["spo2"]):
            rows.append(row)
 
    feat = pd.DataFrame(rows)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[c for c in feat.columns if c.startswith("R_")]
    ).reset_index(drop=True)
 
    n_total  = len(feat)
    n_motion = int(feat["high_motion_window"].sum()) if "high_motion_window" in feat.columns else 0
    print(f"  [features] {n_total} windows | {n_motion} high-motion")
 
    if drop_motion_windows and "high_motion_window" in feat.columns:
        feat_clean = feat[feat["high_motion_window"] == 0].reset_index(drop=True)
        print(f"  [features] {len(feat_clean)} windows retained after motion gating.")
        return feat_clean
 
    return feat