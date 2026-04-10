"""
preprocessing.py
~~~~~~~~~~~~~~~~
Signal preprocessing pipeline:
 
1.  Reference SpO2 sanitisation (clamp, jump-filter, smooth).
2.  Prototype spike / saturation removal.
3.  Morphological baseline wander removal (breathing / vasomotion).
4.  Optional Adaptive Noise Cancellation (LMS) using ACC channels.
5.  Bandpass filtering for AC components; low-pass for DC.
6.  Common-grid resampling.
7.  Cross-correlation–based temporal alignment.
8.  Motion energy computation and per-sample motion flag.
9.  Warm-up discard (startup glitch + sensor drift).
"""
 
from typing import List
 
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
 
from .config import SignalConfig
from .utils import make_relative_time_seconds, normalize_signal
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Reference SpO2
# ─────────────────────────────────────────────────────────────────────────────
 
def sanitize_reference_spo2(
    spo2: np.ndarray,
    min_valid: float = 90.0,
    max_valid: float = 100.0,
    max_jump: float = 5.0,
) -> np.ndarray:
    x = np.asarray(spo2, dtype=float).copy()
    x[(x < min_valid) | (x > max_valid)] = np.nan
 
    for i in range(1, len(x)):
        if np.isnan(x[i - 1]) or np.isnan(x[i]):
            continue
        if abs(x[i] - x[i - 1]) > max_jump:
            x[i] = np.nan
 
    s = pd.Series(x).interpolate(limit_direction="both")
    s = s.rolling(window=5, center=True, min_periods=1).median()
    return s.to_numpy()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Prototype spike removal
# ─────────────────────────────────────────────────────────────────────────────
 
def remove_prototype_adc_spikes(
    x: np.ndarray,
    kernel_size: int = 11,
    z_thresh: float = 6.0,
    saturation_quantile: float = 0.999,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
 
    baseline  = signal.medfilt(x, kernel_size=kernel_size)
    resid     = x - baseline
    mad       = np.median(np.abs(resid - np.median(resid))) + 1e-12
    robust_z  = 0.6745 * resid / mad
 
    dx     = np.diff(x, prepend=x[0])
    dx_mad = np.median(np.abs(dx - np.median(dx))) + 1e-12
    dx_z   = 0.6745 * dx / dx_mad
 
    hi_sat   = np.quantile(x, saturation_quantile)
    lo_sat   = np.quantile(x, 1.0 - saturation_quantile)
    sat_mask = (x >= hi_sat) | (x <= lo_sat)
 
    spike_mask = (np.abs(robust_z) > z_thresh) | (np.abs(dx_z) > z_thresh) | sat_mask
    spike_mask = spike_mask | np.roll(spike_mask, 1) | np.roll(spike_mask, -1)
    spike_mask[0]  = False
    spike_mask[-1] = False
 
    idx  = np.arange(len(x))
    good = ~spike_mask
    if good.sum() < max(3, len(x) // 10):
        return baseline
 
    x[spike_mask] = np.interp(idx[spike_mask], idx[good], x[good])
    return x
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Filtering utilities
# ─────────────────────────────────────────────────────────────────────────────
 
def butter_bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyq  = 0.5 * fs
    b, a = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)
 
 
def butter_lowpass_filter(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyq  = 0.5 * fs
    b, a = signal.butter(order, cutoff_hz / nyq, btype="low")
    return signal.filtfilt(b, a, x)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Baseline wander removal
# ─────────────────────────────────────────────────────────────────────────────
 
def remove_baseline_wander(
    x: np.ndarray,
    fs: float,
    window_sec: float = 0.6,
) -> np.ndarray:
    """
    Morphological baseline estimator using a uniform (box) filter.
 
    A box filter of width >> one cardiac cycle acts as a low-pass that tracks
    only the slow baseline drift (breathing, vasomotion, pressure changes).
    Subtracting this from the raw signal removes wander while preserving
    the pulse shape far better than a high-pass filter alone, because no
    ringing artefacts are introduced at the start/end of the signal.
 
    Parameters
    ----------
    x          : spike-cleaned raw signal.
    fs         : sampling rate (Hz).
    window_sec : filter width in seconds.  Must be longer than one cardiac
                 cycle (> ~1.0 s at 40 bpm).  Default 0.6 s gives a 30-sample
                 kernel at 50 Hz — use 1.2–2.0 s for noisier signals.
 
    Returns
    -------
    x_detrended : baseline-subtracted signal, same length as x.
    """
    kernel = max(3, int(window_sec * fs))
    if kernel % 2 == 0:
        kernel += 1
    # Three-pass smoothing approximates a Gaussian and gives a cleaner
    # baseline estimate than a single wide box filter.
    baseline = uniform_filter1d(x, size=kernel)
    baseline = uniform_filter1d(baseline, size=kernel)
    baseline = uniform_filter1d(baseline, size=kernel)
    return x - baseline
 
 
def compute_rolling_dc(
    x: np.ndarray,
    fs: float,
    window_sec: float = 2.0,
) -> np.ndarray:
    """
    Rolling-mean DC estimate.
 
    Using the window mean as DC is sensitive to slow within-window drift:
    if DC rises 5% across a 10-s window the mean underestimates the true
    DC at the end and overestimates it at the start, biasing R.  A rolling
    mean with a shorter window tracks the drift and gives a local DC
    estimate at each sample instead.
 
    Parameters
    ----------
    x          : spike-cleaned raw signal (before baseline removal).
    fs         : sampling rate (Hz).
    window_sec : width of the rolling mean in seconds.
 
    Returns
    -------
    dc : array of local DC values, same length as x.
    """
    kernel = max(3, int(window_sec * fs))
    if kernel % 2 == 0:
        kernel += 1
    return uniform_filter1d(x, size=kernel)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Noise Cancellation  (LMS)
# ─────────────────────────────────────────────────────────────────────────────
 
def lms_filter(
    primary: np.ndarray,
    reference: np.ndarray,
    order: int = 32,
    mu: float = 0.005,
) -> np.ndarray:
    """
    Least-Mean-Squares adaptive filter.
 
    Treats *reference* (motion signal) as the noise reference and
    returns the residual (PPG with motion suppressed).
 
    Parameters
    ----------
    primary   : PPG signal contaminated with motion (length N).
    reference : Motion reference — normalised ACC magnitude (length N).
    order     : Number of LMS filter taps.
    mu        : Step-size (learning rate). Smaller → more stable, slower.
 
    Returns
    -------
    residual  : Motion-cancelled PPG, same length as *primary*.
    """
    n   = len(primary)
    w   = np.zeros(order, dtype=float)
    out = np.zeros(n, dtype=float)
 
    for i in range(order, n):
        x_vec   = reference[i - order: i][::-1]   # most-recent first
        y_hat   = np.dot(w, x_vec)
        e       = primary[i] - y_hat
        w      += 2.0 * mu * e * x_vec
        out[i]  = e
 
    # fill the initial transient with the un-filtered signal
    out[:order] = primary[:order]
    return out
 
 
def compute_acc_magnitude(
    df: pd.DataFrame,
    acc_cols: List[str] = ("accx", "accy", "accz"),
) -> np.ndarray:
    """
    Euclidean magnitude of the ACC vector.
    Returns zeros if no ACC columns are present.
    """
    present = [c for c in acc_cols if c in df.columns]
    if not present:
        return np.zeros(len(df))
 
    mats = [df[c].to_numpy(dtype=float) for c in present]
    mag  = np.sqrt(sum(m ** 2 for m in mats))
 
    # remove gravity (1 g DC) with a high-pass at 0.5 Hz if we have ≥ 3 axes
    return mag
 
 
def apply_anc_to_ppg(
    proto: pd.DataFrame,
    sig_cfg: SignalConfig,
    optical_cols: List[str] = ("red", "ir", "green", "blue"),
) -> pd.DataFrame:
    """
    Apply LMS-based adaptive noise cancellation to every optical channel
    using the band-passed ACC magnitude as a motion reference.
 
    The raw (spike-cleaned) signals are preserved in *_raw columns;
    the ANC output replaces the bandpass-filtered columns used for AC.
    """
    proto = proto.copy()
 
    acc_cols = [c for c in ("accx", "accy", "accz") if c in proto.columns]
    if not acc_cols:
        print("  [ANC] No ACC channels found — skipping adaptive filtering.")
        return proto
 
    # Band-pass ACC magnitude to isolate motion frequencies
    acc_mag = compute_acc_magnitude(proto, acc_cols)
    acc_bp  = butter_bandpass_filter(
        acc_mag,
        fs=sig_cfg.fs_prototype,
        low_hz=sig_cfg.acc_motion_low_hz,
        high_hz=sig_cfg.acc_motion_high_hz,
        order=4,
    )
    # Normalise reference for stable LMS convergence
    acc_ref = acc_bp / (np.std(acc_bp) + 1e-12)
 
    for c in optical_cols:
        if c not in proto.columns:
            continue
        ppg_bp = proto[c].to_numpy(dtype=float)
        ppg_clean = lms_filter(
            ppg_bp,
            acc_ref,
            order=sig_cfg.anc_filter_order,
            mu=sig_cfg.anc_mu,
        )
        proto[c] = ppg_clean
        print(f"  [ANC] Applied to '{c}' channel.")
 
    return proto
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Motion energy per sample
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_motion_energy(
    df: pd.DataFrame,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 10.0,
) -> np.ndarray:
    """
    Band-passed ACC magnitude (root-mean-power proxy).
    Returns zeros when ACC is unavailable.
    """
    acc_cols = [c for c in ("accx", "accy", "accz") if c in df.columns]
    if not acc_cols:
        return np.zeros(len(df))
 
    mag    = compute_acc_magnitude(df, acc_cols)
    mag_bp = butter_bandpass_filter(mag, fs=fs, low_hz=low_hz, high_hz=high_hz, order=4)
    return np.abs(mag_bp)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────
 
def resample_dataframe(
    df: pd.DataFrame,
    time_col: str,
    target_time: np.ndarray,
    cols: List[str],
) -> pd.DataFrame:
    """
    Resample selected columns onto *target_time* using linear interpolation.
    Appropriate for slowly-varying signals (SpO2, ACC, motion energy) and for
    prototype optical channels whose source rate already matches fs_common.
 
    For reference optical channels being downsampled from a higher rate,
    use downsample_reference instead to avoid aliasing.
    """
    out = {"time": target_time}
    t   = df[time_col].to_numpy(dtype=float)
 
    for c in cols:
        if c not in df.columns:
            continue
        y     = df[c].to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(y)
        if valid.sum() < 2:
            out[c] = np.full_like(target_time, np.nan, dtype=float)
            continue
        f      = interp1d(t[valid], y[valid], kind="linear",
                          fill_value="extrapolate", bounds_error=False)
        out[c] = f(target_time)
 
    return pd.DataFrame(out)
 
 
def downsample_reference(
    df: pd.DataFrame,
    fs_in: float,
    fs_out: float,
    optical_cols: List[str],
    slow_cols: List[str],
    target_time: np.ndarray,
) -> pd.DataFrame:
    """
    Correctly downsample reference optical signals using a polyphase
    anti-aliasing filter (scipy.signal.resample_poly), then align onto
    *target_time* with linear interpolation for any residual timing offset.
 
    Parameters
    ----------
    optical_cols : bandpass-filtered channels that need proper anti-aliasing
                   (red, ir, red_raw, ir_raw).
    slow_cols    : slowly-varying channels where linear interpolation is fine
                   (spo2).
    """
    from math import gcd
    from scipy.signal import resample_poly
 
    out  = {"time": target_time}
    t_in = df["time"].to_numpy(dtype=float)
 
    # Integer up/down ratio — e.g. 100 Hz -> 50 Hz: up=1, down=2
    g    = gcd(int(fs_out), int(fs_in))
    up   = int(fs_out) // g
    down = int(fs_in)  // g
 
    # Optical: polyphase resample with built-in anti-aliasing
    for c in optical_cols:
        if c not in df.columns:
            continue
        y = df[c].to_numpy(dtype=float)
 
        # Fill NaNs before polyphase filter (it cannot handle them)
        nan_mask = ~np.isfinite(y)
        if nan_mask.any():
            idx  = np.arange(len(y))
            good = ~nan_mask
            if good.sum() < 2:
                out[c] = np.full_like(target_time, np.nan)
                continue
            y = np.interp(idx, idx[good], y[good])
 
        y_ds = resample_poly(y, up, down)
 
        # Time axis of the downsampled signal
        t_ds = np.arange(len(y_ds)) / fs_out + t_in[0]
 
        # Fine alignment onto the common target_time grid
        f      = interp1d(t_ds, y_ds, kind="linear",
                          fill_value="extrapolate", bounds_error=False)
        out[c] = f(target_time)
 
    # Slow signals: linear interpolation is sufficient
    for c in slow_cols:
        if c not in df.columns:
            continue
        y     = df[c].to_numpy(dtype=float)
        valid = np.isfinite(t_in) & np.isfinite(y)
        if valid.sum() < 2:
            out[c] = np.full_like(target_time, np.nan)
            continue
        f      = interp1d(t_in[valid], y[valid], kind="linear",
                          fill_value="extrapolate", bounds_error=False)
        out[c] = f(target_time)
 
    return pd.DataFrame(out)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Alignment
# ─────────────────────────────────────────────────────────────────────────────
 
def estimate_lag_seconds(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: float,
    max_lag_sec: float = 10.0,
) -> float:
    """
    Cross-correlation lag estimate.  Returns the shift (in seconds) that
    should be *added* to sig_b's time axis to align it with sig_a.
    """
    a       = normalize_signal(sig_a)
    b       = normalize_signal(sig_b)
    corr    = signal.correlate(a, b, mode="full")
    lags    = signal.correlation_lags(len(a), len(b), mode="full")
    max_lag = int(max_lag_sec * fs)
    valid   = np.abs(lags) <= max_lag
    best    = lags[valid][np.argmax(corr[valid])]
    return float(best) / fs
 
 
def apply_time_shift(df: pd.DataFrame, shift_sec: float) -> pd.DataFrame:
    out        = df.copy()
    out["time"] = out["time"] + shift_sec
    return out
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Master preprocessing + merge
# ─────────────────────────────────────────────────────────────────────────────
 
def preprocess_and_merge(
    reference_df: pd.DataFrame,
    prototype_df: pd.DataFrame,
    sig_cfg: SignalConfig,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
 
    Returns a merged DataFrame on a common time grid containing:
 
    Columns produced
    ----------------
    time          – common time axis (s)
    spo2          – sanitised reference SpO2 (%)
    ref_red/ir    – bandpass-filtered reference optical channels
    ref_red_raw/ir_raw – raw reference optical channels
    red/ir/green/blue          – ANC + bandpass-filtered prototype channels (AC)
    red_raw/ir_raw/green_raw/blue_raw – spike-cleaned prototype channels (DC source)
    motion_energy – per-sample ACC-derived motion energy
    high_motion   – boolean flag (1 = high motion)
    acc_mag       – raw ACC magnitude (for diagnostics)
    accx/accy/accz – resampled raw ACC axes
    """
    ref   = reference_df.copy()
    proto = prototype_df.copy()
 
    # ── Time axis ─────────────────────────────────────────────────────────────
    if "time" not in ref.columns:
        raise KeyError("Reference dataframe must contain 'time'.")
    ref["time"] = make_relative_time_seconds(ref["time"])
 
    if "timestamp" in proto.columns:
        proto["time"] = make_relative_time_seconds(proto["timestamp"])
    elif "time" in proto.columns:
        proto["time"] = make_relative_time_seconds(proto["time"])
    else:
        raise KeyError("Prototype dataframe must contain 'timestamp' or 'time'.")
 
    # ── Reference SpO2 ────────────────────────────────────────────────────────
    ref["spo2"] = sanitize_reference_spo2(
        ref["spo2"].to_numpy(dtype=float),
        min_valid=sig_cfg.min_valid_spo2,
        max_valid=sig_cfg.max_valid_spo2,
        max_jump=sig_cfg.max_spo2_jump_pct,
    )
 
    # ── Reference optical: raw + bandpass ────────────────────────────────────
    for c in ("red", "ir"):
        if c in ref.columns:
            ref[f"{c}_raw"] = ref[c].to_numpy(dtype=float)
            ref[c] = butter_bandpass_filter(
                ref[c].to_numpy(dtype=float),
                fs=sig_cfg.fs_reference,
                low_hz=sig_cfg.bandpass_low_hz,
                high_hz=sig_cfg.bandpass_high_hz,
                order=sig_cfg.bandpass_order,
            )
 
    # ── Prototype optical: spike removal → baseline removal → bandpass ─────────
    optical_present = [c for c in ("red", "ir", "green", "blue") if c in proto.columns]
    for c in optical_present:
        # Step 1: remove ADC spikes / saturation events
        cleaned = remove_prototype_adc_spikes(
            proto[c].to_numpy(dtype=float),
            kernel_size=sig_cfg.prototype_spike_kernel,
            z_thresh=sig_cfg.prototype_spike_zscore,
            saturation_quantile=sig_cfg.prototype_saturation_quantile,
        )
 
        # Step 2: store spike-cleaned signal as the raw DC source
        proto[f"{c}_raw"] = cleaned
 
        # Step 3: rolling DC — tracks slow drift within each window
        # (stored as {c}_rolling_dc and used in features.py instead of mean)
        proto[f"{c}_rolling_dc"] = compute_rolling_dc(
            cleaned,
            fs=sig_cfg.fs_prototype,
            window_sec=sig_cfg.rolling_dc_window_sec,
        )
 
        # Step 4: morphological baseline wander removal
        # Do this on the raw signal before bandpass so the filter operates
        # on a signal whose slow drift has already been removed, reducing
        # edge transients and leaving the cardiac AC component intact.
        if sig_cfg.remove_baseline_wander:
            detrended = remove_baseline_wander(
                cleaned,
                fs=sig_cfg.fs_prototype,
                window_sec=sig_cfg.baseline_wander_window_sec,
            )
        else:
            detrended = cleaned
 
        # Step 5: bandpass filter the detrended signal → AC component
        proto[c] = butter_bandpass_filter(
            detrended,
            fs=sig_cfg.fs_prototype,
            low_hz=sig_cfg.bandpass_low_hz,
            high_hz=sig_cfg.bandpass_high_hz,
            order=sig_cfg.bandpass_order,
        )
 
        # Step 6: keep a lowpass DC for legacy compatibility (1 Hz cutoff)
        proto[f"{c}_dc"] = butter_lowpass_filter(
            cleaned, fs=sig_cfg.fs_prototype, cutoff_hz=1.0,
            order=sig_cfg.bandpass_order,
        )
 
    # ── Adaptive Noise Cancellation (LMS) ────────────────────────────────────
    if sig_cfg.enable_anc:
        proto = apply_anc_to_ppg(proto, sig_cfg, optical_cols=optical_present)
 
    # ── Pre-alignment motion energy (on prototype grid) ──────────────────────
    proto_motion = compute_motion_energy(
        proto, fs=sig_cfg.fs_prototype,
        low_hz=sig_cfg.acc_motion_low_hz,
        high_hz=sig_cfg.acc_motion_high_hz,
    )
    proto["motion_energy"] = proto_motion
 
    # Raw ACC magnitude for diagnostics
    acc_mag = compute_acc_magnitude(proto)
    proto["acc_mag"] = acc_mag
 
    # ── Common time grid ─────────────────────────────────────────────────────
    t_start     = max(ref["time"].min(),   proto["time"].min())
    t_end       = min(ref["time"].max(),   proto["time"].max())
    target_time = np.arange(t_start, t_end, 1.0 / sig_cfg.fs_common)
 
    # ── Resample reference ───────────────────────────────────────────────────
    # red/ir/red_raw/ir_raw are bandpass-filtered signals being downsampled
    # from fs_reference (100 Hz) to fs_common (50 Hz), so they need a proper
    # polyphase anti-aliasing filter.  SpO2 is a slow signal so linear
    # interpolation is sufficient.
    if sig_cfg.fs_reference != sig_cfg.fs_common:
        ref_rs = downsample_reference(
            ref,
            fs_in=sig_cfg.fs_reference,
            fs_out=sig_cfg.fs_common,
            optical_cols=["red", "ir", "red_raw", "ir_raw"],
            slow_cols=["spo2"],
            target_time=target_time,
        )
        print(f"  [resample] Reference downsampled {sig_cfg.fs_reference:.0f} Hz "
              f"-> {sig_cfg.fs_common:.0f} Hz via polyphase filter.")
    else:
        ref_rs = resample_dataframe(
            ref, "time", target_time,
            ["red", "ir", "red_raw", "ir_raw", "spo2"],
        )
 
    # ── Resample prototype (optical + ACC + motion) ──────────────────────────
    proto_optical_raw = [f"{c}_raw" for c in optical_present]
    proto_rolling_dc  = [f"{c}_rolling_dc" for c in optical_present]
    proto_acc = [c for c in ("accx", "accy", "accz", "acc_mag", "motion_energy") if c in proto.columns]
    proto_rs = resample_dataframe(
        proto, "time", target_time,
        optical_present + proto_optical_raw + proto_rolling_dc + proto_acc,
    )
 
    # ── Temporal alignment ────────────────────────────────────────────────────
    if sig_cfg.enable_alignment and "red" in ref_rs.columns and "red" in proto_rs.columns:
        lag = estimate_lag_seconds(
            ref_rs["red"].to_numpy(),
            proto_rs["red"].to_numpy(),
            fs=sig_cfg.fs_common,
            max_lag_sec=sig_cfg.max_alignment_lag_sec,
        )
        print(f"  [align] Estimated prototype shift: {lag:+.3f} s")
 
        proto_shifted = apply_time_shift(proto_rs, lag)
        proto_rs = resample_dataframe(
            proto_shifted, "time", target_time,
            optical_present + proto_optical_raw + proto_rolling_dc + proto_acc,
        )
 
    # ── Build merged DataFrame ────────────────────────────────────────────────
    merged = pd.DataFrame({"time": target_time})
 
    # SpO2 from reference
    if "spo2" in ref_rs.columns:
        merged["spo2"] = ref_rs["spo2"].to_numpy()
 
    # Reference optical (debugging / plotting)
    for c, col_rs in (("red", "ref_red"), ("ir", "ref_ir")):
        if c in ref_rs.columns:
            merged[col_rs] = ref_rs[c].to_numpy()
    for c, col_rs in (("red_raw", "ref_red_raw"), ("ir_raw", "ref_ir_raw")):
        if c in ref_rs.columns:
            merged[col_rs] = ref_rs[c].to_numpy()
 
    # Prototype optical — filtered (AC), raw (DC), and rolling DC
    for c in optical_present:
        if c in proto_rs.columns:
            merged[c] = proto_rs[c].to_numpy()
    for c in proto_optical_raw:
        if c in proto_rs.columns:
            merged[c] = proto_rs[c].to_numpy()
    for c in proto_rolling_dc:
        if c in proto_rs.columns:
            merged[c] = proto_rs[c].to_numpy()
 
    # ACC / motion columns
    for c in ("accx", "accy", "accz", "acc_mag", "motion_energy"):
        if c in proto_rs.columns:
            merged[c] = proto_rs[c].to_numpy()
 
    # Motion flag — threshold is derived from the data itself.
    # sig_cfg.motion_threshold is treated as a percentile (0–1) of the
    # motion energy distribution rather than an absolute ADC value.
    # e.g. 0.5 means the top 50% of motion energy is flagged as high motion,
    # which naturally adapts to any sensor scaling.
    if "motion_energy" in merged.columns:
        me = merged["motion_energy"].to_numpy()
 
        # motion_threshold is a 0-1 fraction; convert to percentile in [0, 100]
        pct = float(np.clip(sig_cfg.motion_threshold, 0.0, 1.0)) * 100.0
        adaptive_threshold = np.nanpercentile(me, pct)
        merged["high_motion"] = (abs(me) > adaptive_threshold).astype(int)
        print(f"  [motion] Adaptive threshold (p{pct:.0f}): "
              f"{adaptive_threshold:.4f}  |  min={me.min():.4f}  max={me.max():.4f}")
    else:
        merged["high_motion"] = 0
 
    # ── Drop rows with invalid SpO2 or missing essential channels ────────────
    required = {"time", "spo2"}
    for c in ("red", "ir", "red_raw", "ir_raw"):
        if c in merged.columns:
            required.add(c)
 
    merged = merged.dropna(subset=list(required)).reset_index(drop=True)
 
    # ── Warm-up discard ───────────────────────────────────────────────────────
    # Remove the first N seconds to eliminate the ADC startup glitch
    # (dropout + overshoot in first ~30 samples) and the rapid DC drift
    # caused by LED/sensor thermal settling.
    if sig_cfg.warmup_discard_sec > 0 and len(merged) > 0:
        cutoff_time = merged["time"].min() + sig_cfg.warmup_discard_sec
        n_before = len(merged)
        merged = merged[merged["time"] >= cutoff_time].reset_index(drop=True)
        n_discarded = n_before - len(merged)
        print(f"  [warmup] Discarded first {sig_cfg.warmup_discard_sec:.0f} s "
              f"({n_discarded} samples removed)")
 
    n_total  = len(merged)
    n_motion = int(merged["high_motion"].sum()) if "high_motion" in merged.columns else 0
    print(f"  [merge] {n_total} samples retained | "
          f"{n_motion} high-motion ({100*n_motion/max(n_total,1):.1f}%)")
 
    return merged