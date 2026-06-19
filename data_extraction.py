from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import signal as dsp
from scipy import stats


# ---------------------------------------------------------------------------
# Window object
# ---------------------------------------------------------------------------

@dataclass
class PPGWindow:
    """Features extracted from one windowed segment of a PPG recording."""
    subject:       str
    episode:       Optional[str]
    window_idx:    int
    t_start_s:     float        # seconds from the (artifact-cleaned) signal start

    # Core SpO2 features
    R:             float        # ratio of ratios (AC_red/DC_red) / (AC_ir/DC_ir)
    spo2_ref:      float        # mean reference SpO2 over the window (%)
    pi:            float        # perfusion index AC_ir / DC_ir

    # Statistical / signal quality features — all four optical channels
    skewness_red:     float
    skewness_ir:      float
    skewness_green:   float
    skewness_blue:    float
    kurtosis_red:     float
    kurtosis_ir:      float
    kurtosis_green:   float
    kurtosis_blue:    float
    snr_red:          float       # dB  signal-band power / residual-noise power
    snr_ir:           float
    snr_green:        float
    snr_blue:         float
    entropy_red:      float       # spectral entropy (bits)
    entropy_ir:       float
    entropy_green:    float
    entropy_blue:     float
    rel_power_red:    float       # fraction of total power inside the cardiac band
    rel_power_ir:     float
    rel_power_green:  float
    rel_power_blue:   float
    acc_energy_std:   float       # std-dev of acceleration L2-norm within the window

    # All pairwise ratio-of-ratios (R already = red/IR above)
    R_red_green:  float           # (AC_red/DC_red) / (AC_green/DC_green)
    R_red_blue:   float
    R_ir_green:   float
    R_ir_blue:    float
    R_green_blue: float


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _r_ratio(ac_num: float, dc_num: float, ac_den: float, dc_den: float) -> float:
    """(AC_num/DC_num) / (AC_den/DC_den) — returns NaN when any component is zero."""
    if dc_num == 0.0 or dc_den == 0.0 or ac_den == 0.0:
        return np.nan
    return (ac_num / dc_num) / (ac_den / dc_den)


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = dsp.butter(4, [low / nyq, high / nyq], btype='band', output='ba') # type: ignore
    return dsp.filtfilt(b, a, x)


def _lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = dsp.butter(4, cutoff / nyq, btype='low', output='ba') # type: ignore
    return dsp.filtfilt(b, a, x)


# ---------------------------------------------------------------------------
# Per-window feature helpers
# ---------------------------------------------------------------------------

def _snr_db(raw: np.ndarray, bandpassed: np.ndarray) -> float:
    """SNR in dB: signal-band power vs. residual noise power."""
    p_signal = float(np.var(bandpassed))
    p_noise  = float(np.var(raw - bandpassed))
    if p_noise == 0.0:
        return np.inf
    return 10.0 * np.log10(p_signal / p_noise)


def _spectral_entropy(x: np.ndarray, fs: float) -> float:
    """Shannon entropy (bits) of the normalised power spectral density."""
    _, psd = dsp.welch(x, fs=fs, nperseg=min(len(x), 256))
    psd = psd[psd > 0]
    psd = psd / psd.sum()
    return float(-np.sum(psd * np.log2(psd)))


def _relative_power(x: np.ndarray, fs: float, low: float, high: float) -> float:
    """Fraction of total signal power contained in [low, high] Hz."""
    f, psd  = dsp.welch(x, fs=fs, nperseg=min(len(x), 256))
    total   = float(np.trapz(psd, f))
    if total == 0.0:
        return 0.0
    band    = (f >= low) & (f <= high)
    return float(np.trapezoid(psd[band], f[band]) / total)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_windows(
    records:  list,
    window_s: float = 10.0,
    overlap:  float = 0.75,
    bp_low:   float = 0.5,
    bp_high:  float = 6.0,
) -> List[PPGWindow]:
    """
    Slide a window over every clean record and compute features per window.

    Parameters
    ----------
    records  : preprocessed records from preprocessing.remove_artifacts
    window_s : window length in seconds
    overlap  : fraction of overlap between successive windows (0 – <1)
    bp_low   : bandpass / lowpass cut-off frequency for AC/DC split (Hz)
    bp_high  : bandpass upper cut-off frequency (Hz)

    Returns
    -------
    List of PPGWindow objects ordered by subject → episode → time.

    Notes
    -----
    After artifact removal the remaining samples are treated as contiguous.
    t_start_s therefore reflects position in the cleaned signal, not the
    original recording clock.
    """
    windows: List[PPGWindow] = []

    for rec in records:
        ppg_fs  = rec['ppg_fs']
        spo2_fs = rec['spo2_fs']
        ratio   = spo2_fs / ppg_fs       # SpO2 samples per PPG sample

        red   = rec['ppg']['red'].astype(float)
        ir    = rec['ppg']['ir'].astype(float)
        green = rec['ppg']['green'].astype(float)
        blue  = rec['ppg']['blue'].astype(float)
        spo2  = rec['spo2']['spo2'].astype(float)
        acc_energy = np.sqrt(
            rec['ppg']['acc_x'].astype(float) ** 2 +
            rec['ppg']['acc_y'].astype(float) ** 2 +
            rec['ppg']['acc_z'].astype(float) ** 2
        )

        win_ppg  = int(window_s * ppg_fs)
        step_ppg = max(1, int(win_ppg * (1.0 - overlap)))
        n_ppg    = len(red)
        n_spo2   = len(spo2)

        win_idx = 0
        start   = 0
        while start + win_ppg <= n_ppg:
            end_ppg = start + win_ppg
            red_w   = red[start:end_ppg]
            ir_w    = ir[start:end_ppg]
            green_w = green[start:end_ppg]
            blue_w  = blue[start:end_ppg]

            # Corresponding SpO2 window
            s_s = min(int(round(start   * ratio)), n_spo2)
            e_s = min(int(round(end_ppg * ratio)), n_spo2)
            spo2_w = spo2[s_s:e_s]

            # --- AC (bandpass) and DC (lowpass) — all 4 channels ---
            red_bp   = _bandpass(red_w,   ppg_fs, bp_low, bp_high)
            ir_bp    = _bandpass(ir_w,    ppg_fs, bp_low, bp_high)
            green_bp = _bandpass(green_w, ppg_fs, bp_low, bp_high)
            blue_bp  = _bandpass(blue_w,  ppg_fs, bp_low, bp_high)
            red_lp   = _lowpass(red_w,   ppg_fs, bp_low)
            ir_lp    = _lowpass(ir_w,    ppg_fs, bp_low)
            green_lp = _lowpass(green_w, ppg_fs, bp_low)
            blue_lp  = _lowpass(blue_w,  ppg_fs, bp_low)

            ac_red   = float(red_bp.std());   dc_red   = float(red_lp.mean())
            ac_ir    = float(ir_bp.std());    dc_ir    = float(ir_lp.mean())
            ac_green = float(green_bp.std()); dc_green = float(green_lp.mean())
            ac_blue  = float(blue_bp.std());  dc_blue  = float(blue_lp.mean())

            # All 6 pairwise ratios of ratios
            R            = _r_ratio(ac_red,   dc_red,   ac_ir,    dc_ir)
            R_red_green  = _r_ratio(ac_red,   dc_red,   ac_green, dc_green)
            R_red_blue   = _r_ratio(ac_red,   dc_red,   ac_blue,  dc_blue)
            R_ir_green   = _r_ratio(ac_ir,    dc_ir,    ac_green, dc_green)
            R_ir_blue    = _r_ratio(ac_ir,    dc_ir,    ac_blue,  dc_blue)
            R_green_blue = _r_ratio(ac_green, dc_green, ac_blue,  dc_blue)

            # Perfusion index (IR channel, AC/DC)
            pi = (ac_ir / dc_ir) if dc_ir != 0.0 else np.nan

            # Reference SpO2 (mean over window; exclude zero/missing values)
            valid_spo2 = spo2_w[spo2_w > 0]
            spo2_ref   = float(valid_spo2.mean()) if len(valid_spo2) > 0 else np.nan

            # Acceleration energy std-dev within the window
            acc_energy_std = float(acc_energy[start:end_ppg].std())

            windows.append(PPGWindow(
                subject       = rec['subject'],
                episode       = rec['episode'],
                window_idx    = win_idx,
                t_start_s     = start / ppg_fs,
                R             = float(R),
                spo2_ref      = spo2_ref,
                pi            = float(pi),
                skewness_red   = float(stats.skew(red_bp)),
                skewness_ir    = float(stats.skew(ir_bp)),
                skewness_green = float(stats.skew(green_bp)),
                skewness_blue  = float(stats.skew(blue_bp)),
                kurtosis_red   = float(stats.kurtosis(red_bp)),
                kurtosis_ir    = float(stats.kurtosis(ir_bp)),
                kurtosis_green = float(stats.kurtosis(green_bp)),
                kurtosis_blue  = float(stats.kurtosis(blue_bp)),
                snr_red        = _snr_db(red_w,   red_bp),
                snr_ir         = _snr_db(ir_w,    ir_bp),
                snr_green      = _snr_db(green_w, green_bp),
                snr_blue       = _snr_db(blue_w,  blue_bp),
                entropy_red    = _spectral_entropy(red_w,   ppg_fs),
                entropy_ir     = _spectral_entropy(ir_w,    ppg_fs),
                entropy_green  = _spectral_entropy(green_w, ppg_fs),
                entropy_blue   = _spectral_entropy(blue_w,  ppg_fs),
                rel_power_red   = _relative_power(red_w,   ppg_fs, bp_low, bp_high),
                rel_power_ir    = _relative_power(ir_w,    ppg_fs, bp_low, bp_high),
                rel_power_green = _relative_power(green_w, ppg_fs, bp_low, bp_high),
                rel_power_blue  = _relative_power(blue_w,  ppg_fs, bp_low, bp_high),
                acc_energy_std = acc_energy_std,
                R_red_green    = float(R_red_green),
                R_red_blue     = float(R_red_blue),
                R_ir_green     = float(R_ir_green),
                R_ir_blue      = float(R_ir_blue),
                R_green_blue   = float(R_green_blue),
            ))

            start   += step_ppg
            win_idx += 1

    return windows


# ---------------------------------------------------------------------------
# Signal Quality Index filter
# ---------------------------------------------------------------------------

def _in_range(val: float, lo: Optional[float], hi: Optional[float]) -> bool:
    if lo is not None and val < lo:
        return False
    if hi is not None and val > hi:
        return False
    return True


def apply_sqi(
    windows:          List[PPGWindow],
    pi_min:           Optional[float] = None,
    pi_max:           Optional[float] = None,
    skewness_min:     Optional[float] = None,
    skewness_max:     Optional[float] = None,
    kurtosis_min:     Optional[float] = None,
    kurtosis_max:     Optional[float] = None,
    snr_min:          Optional[float] = None,
    snr_max:          Optional[float] = None,
    entropy_min:      Optional[float] = None,
    entropy_max:      Optional[float] = None,
    rel_power_min:    Optional[float] = None,
    rel_power_max:    Optional[float] = None,
    acc_std_max:      Optional[float] = None,
) -> List[PPGWindow]:
    """
    Drop windows that fall outside quality thresholds.

    Spectral metrics (skewness, kurtosis, SNR, entropy, relative power) are
    checked on *all four* optical channels — a window is rejected if any
    channel is out of range.
    acc_std_max rejects windows where acceleration energy std-dev exceeds the
    limit (indicates motion within the window).
    Any threshold left as None imposes no bound on that side.

    Returns the filtered list and prints a rejection summary.
    """
    kept = []
    for w in windows:
        if not _in_range(w.pi,             pi_min,        pi_max):        continue
        if not _in_range(w.skewness_red,    skewness_min,  skewness_max):  continue
        if not _in_range(w.skewness_ir,     skewness_min,  skewness_max):  continue
        if not _in_range(w.skewness_green,  skewness_min,  skewness_max):  continue
        if not _in_range(w.skewness_blue,   skewness_min,  skewness_max):  continue
        if not _in_range(w.kurtosis_red,    kurtosis_min,  kurtosis_max):  continue
        if not _in_range(w.kurtosis_ir,     kurtosis_min,  kurtosis_max):  continue
        if not _in_range(w.kurtosis_green,  kurtosis_min,  kurtosis_max):  continue
        if not _in_range(w.kurtosis_blue,   kurtosis_min,  kurtosis_max):  continue
        if not _in_range(w.snr_red,         snr_min,       snr_max):       continue
        if not _in_range(w.snr_ir,          snr_min,       snr_max):       continue
        if not _in_range(w.snr_green,       snr_min,       snr_max):       continue
        if not _in_range(w.snr_blue,        snr_min,       snr_max):       continue
        if not _in_range(w.entropy_red,     entropy_min,   entropy_max):   continue
        if not _in_range(w.entropy_ir,      entropy_min,   entropy_max):   continue
        if not _in_range(w.entropy_green,   entropy_min,   entropy_max):   continue
        if not _in_range(w.entropy_blue,    entropy_min,   entropy_max):   continue
        if not _in_range(w.rel_power_red,   rel_power_min, rel_power_max): continue
        if not _in_range(w.rel_power_ir,    rel_power_min, rel_power_max): continue
        if not _in_range(w.rel_power_green, rel_power_min, rel_power_max): continue
        if not _in_range(w.rel_power_blue,  rel_power_min, rel_power_max): continue
        if not _in_range(w.acc_energy_std, None,          acc_std_max):   continue
        kept.append(w)

    n_total    = len(windows)
    n_rejected = n_total - len(kept)
    print(f"  SQI: kept {len(kept)} / {n_total} windows ({n_rejected} rejected, "
          f"{100 * n_rejected / n_total:.1f}%)")
    return kept
