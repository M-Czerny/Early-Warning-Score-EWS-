"""
Feature extraction: cut each recording into overlapping windows and describe
each window with numbers.

For every window and every LED (red, IR, green, blue) we compute:
  - AC: size of the pulse (std of the 0.5-6 Hz bandpassed signal)
  - DC: baseline light level (mean of the < 0.5 Hz lowpassed signal)
  - R:  ratio of ratios (AC_a/DC_a) / (AC_b/DC_b) for every LED pair,
        the classic pulse-oximetry feature
  - quality metrics (SQIs) used by sqi.py to drop noisy windows
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import List, Optional

import numpy as np
from scipy import signal as dsp
from scipy import stats

LEDS = ('red', 'ir', 'green', 'blue')


# ---------------------------------------------------------------------------
# Window object
# ---------------------------------------------------------------------------

@dataclass
class PPGWindow:
    """All features of one window.  Every field also becomes a CSV column."""
    subject:       str
    episode:       Optional[str]
    window_idx:    int
    t_start_s:     float        # start time in seconds, counted on the cleaned signal

    R_red_ir:      float        # classic pulse-oximetry R = (AC/DC red) / (AC/DC IR)
    spo2_ref:      float        # mean reference SpO2 in the window (%): the target
    pi:            float        # perfusion index of the IR channel (%)

    # AC (pulse size) and DC (baseline) per LED
    ac_red:   float
    dc_red:   float
    ac_ir:    float
    dc_ir:    float
    ac_green: float
    dc_green: float
    ac_blue:  float
    dc_blue:  float

    # AC/DC per LED (NaN when DC is 0)
    acdc_red:   float
    acdc_ir:    float
    acdc_green: float
    acdc_blue:  float

    # Quality metrics per LED (see "Quality metrics" below)
    skewness_red:     float
    skewness_ir:      float
    skewness_green:   float
    skewness_blue:    float
    kurtosis_red:     float
    kurtosis_ir:      float
    kurtosis_green:   float
    kurtosis_blue:    float
    snr_red:          float
    snr_ir:           float
    snr_green:        float
    snr_blue:         float
    entropy_red:      float
    entropy_ir:       float
    entropy_green:    float
    entropy_blue:     float
    rel_power_red:    float
    rel_power_ir:     float
    rel_power_green:  float
    rel_power_blue:   float
    acc_energy_std:   float     # std of the accelerometer magnitude: high = movement

    # R for the other LED pairs (first LED / second LED)
    R_red_green:  float
    R_red_blue:   float
    R_ir_green:   float
    R_ir_blue:    float
    R_green_blue: float

    # The same pairs the other way round
    R_ir_red:     float
    R_green_red:  float
    R_blue_red:   float
    R_green_ir:   float
    R_blue_ir:    float
    R_blue_green: float

    # Heart rate (bpm) from the prototype IR channel and from the reference
    # device.  Only used to check the SQI filter, never as a model feature.
    hr_ppg: float = np.nan
    hr_ref: float = np.nan


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """4th-order Butterworth bandpass, run forwards and backwards (no time shift)."""
    b, a = dsp.butter(4, [low, high], btype='band', fs=fs)
    return dsp.filtfilt(b, a, x)


def lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """4th-order Butterworth lowpass, run forwards and backwards (no time shift)."""
    b, a = dsp.butter(4, cutoff, btype='low', fs=fs)
    return dsp.filtfilt(b, a, x)


def r_ratio(ac_a: float, dc_a: float, ac_b: float, dc_b: float) -> float:
    """Ratio of ratios (AC_a/DC_a) / (AC_b/DC_b).  NaN if it would divide by zero."""
    if dc_a == 0.0 or dc_b == 0.0 or ac_b == 0.0:
        return np.nan
    return float((ac_a / dc_a) / (ac_b / dc_b))


def dominant_hr_bpm(x: np.ndarray, fs: float,
                    low: float = 0.67, high: float = 3.5) -> float:
    """Heart rate (bpm) at the strongest frequency between 40 and 210 bpm."""
    if len(x) < 2 * fs:
        return np.nan
    # Zero-padding to 8192 points gives a finer frequency grid.
    f, psd = dsp.periodogram(x, fs=fs, window='hann', nfft=max(8192, len(x)),
                             detrend='constant')
    band = (f >= low) & (f <= high)
    return float(f[band][np.argmax(psd[band])] * 60.0)


# ---------------------------------------------------------------------------
# Quality metrics (SQIs)
# ---------------------------------------------------------------------------
# Each follows a published definition (references in sqi.py).
# `raw` is the window as recorded, `filt` is the same window bandpassed 0.5-6 Hz.

# The sensor measures light, which drops when blood volume rises, so the raw
# waveform is upside down compared with the literature.  Only skewness cares
# about the sign, so it is computed on the flipped signal.
PPG_INVERTED = True


def perfusion_sqi(raw: np.ndarray, filt: np.ndarray) -> float:
    """Perfusion (Elgendi): peak-to-peak pulse size / baseline * 100 [%]."""
    dc = abs(float(np.mean(raw)))
    return float((filt.max() - filt.min()) / dc * 100.0) if dc > 0 else np.nan


def skewness_sqi(filt: np.ndarray) -> float:
    """Skewness (Elgendi).  Higher = better."""
    return float(stats.skew(-filt if PPG_INVERTED else filt))


def kurtosis_sqi(filt: np.ndarray) -> float:
    """Kurtosis (Selvaraj), Pearson version: clean ~2.3, artefact above 3.5.  Higher = worse."""
    return float(stats.kurtosis(filt, fisher=False))


def shannon_entropy_sqi(filt: np.ndarray, bins: int = 16) -> float:
    """
    Shannon entropy of the amplitude histogram (Selvaraj), scaled to [0, 1].
    Clean ~0.9, artefact below 0.8.  Higher = better.
    """
    counts, _ = np.histogram(filt, bins=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(bins))


def snr_hr_db(filt: np.ndarray, fs: float,
              f_lo: float = 0.5, f_hi: float = 4.0) -> float:
    """
    SNR in dB (de Haan): power near the heart rate and its first harmonic,
    divided by the rest of the power in 30-240 bpm.  Higher = better.

    The heart rate is the window's own strongest peak, because the real device
    has no reference.  Below 0 dB there is more noise than pulse.
    """
    hr = dominant_hr_bpm(filt, fs) / 60.0
    if np.isnan(hr):
        return np.nan
    f, psd = dsp.periodogram(filt, fs=fs, window='hann', nfft=max(8192, len(filt)),
                             detrend='constant')
    in_range = (f >= f_lo) & (f <= f_hi)
    is_pulse = in_range & ((np.abs(f - hr) <= 0.1) | (np.abs(f - 2 * hr) <= 0.2))
    p_sig, p_noise = psd[is_pulse].sum(), psd[in_range & ~is_pulse].sum()
    return float(10 * np.log10(p_sig / p_noise)) if p_noise > 0 else np.inf


def relative_power_sqi(raw: np.ndarray, fs: float) -> float:
    """Relative power (Elgendi): power in 1-2.25 Hz / power in 0-8 Hz.  Higher = better."""
    f, psd = dsp.welch(raw, fs=fs, nperseg=min(len(raw), 256))
    total  = psd[(f >= 0) & (f <= 8)].sum()
    return float(psd[(f >= 1) & (f <= 2.25)].sum() / total) if total > 0 else np.nan


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
    Slide a window over every record and compute the features of each window.

    records  : output of preprocessing.remove_artifacts
    window_s : window length in seconds
    overlap  : how much consecutive windows overlap (0 = none, 0.75 = 75 %)
    bp_low   : low edge of the pulse band (Hz), also the DC lowpass cut-off
    bp_high  : high edge of the pulse band (Hz)

    Samples removed as artifacts are simply skipped, so the remaining samples
    are treated as one continuous signal.
    """
    windows: List[PPGWindow] = []

    for rec in records:
        ppg_fs, spo2_fs = rec['ppg_fs'], rec['spo2_fs']
        to_spo2_idx = spo2_fs / ppg_fs        # converts a PPG index to a SpO2 index

        ppg    = {led: rec['ppg'][led].astype(float) for led in LEDS}
        spo2   = rec['spo2']['spo2'].astype(float)
        ref_ir = rec['spo2']['ir'].astype(float)
        acc    = np.sqrt(sum(rec['ppg'][a].astype(float) ** 2
                             for a in ('acc_x', 'acc_y', 'acc_z')))

        win  = int(window_s * ppg_fs)                  # window length in samples
        step = max(1, int(win * (1.0 - overlap)))      # step between windows

        for win_idx, start in enumerate(range(0, len(ppg['red']) - win + 1, step)):
            end = start + win
            raw = {led: ppg[led][start:end] for led in LEDS}
            ac_part = {led: bandpass(raw[led], ppg_fs, bp_low, bp_high) for led in LEDS}
            ac = {led: float(ac_part[led].std()) for led in LEDS}
            dc = {led: float(lowpass(raw[led], ppg_fs, bp_low).mean()) for led in LEDS}

            # The matching slice of the reference device.
            s0 = min(int(round(start * to_spo2_idx)), len(spo2))
            s1 = min(int(round(end * to_spo2_idx)), len(spo2))
            spo2_w   = spo2[s0:s1]
            ref_ir_w = ref_ir[s0:s1]
            valid    = spo2_w[spo2_w > 0]
            hr_ref   = (dominant_hr_bpm(bandpass(ref_ir_w, spo2_fs, bp_low, bp_high), spo2_fs)
                        if len(ref_ir_w) >= 2 * spo2_fs else np.nan)

            f = dict(
                subject=rec['subject'], episode=rec['episode'],
                window_idx=win_idx, t_start_s=start / ppg_fs,
                spo2_ref=float(valid.mean()) if len(valid) else np.nan,
                pi=perfusion_sqi(raw['ir'], ac_part['ir']),
                acc_energy_std=float(acc[start:end].std()),
                hr_ppg=dominant_hr_bpm(ac_part['ir'], ppg_fs),
                hr_ref=hr_ref,
            )
            for led in LEDS:
                f[f'ac_{led}'] = ac[led]
                f[f'dc_{led}'] = dc[led]
                f[f'acdc_{led}'] = ac[led] / dc[led] if dc[led] else np.nan
                f[f'skewness_{led}']  = skewness_sqi(ac_part[led])
                f[f'kurtosis_{led}']  = kurtosis_sqi(ac_part[led])
                f[f'snr_{led}']       = snr_hr_db(ac_part[led], ppg_fs)
                f[f'entropy_{led}']   = shannon_entropy_sqi(ac_part[led])
                f[f'rel_power_{led}'] = relative_power_sqi(raw[led], ppg_fs)
            for a, b in permutations(LEDS, 2):         # all 12 ordered LED pairs
                f[f'R_{a}_{b}'] = r_ratio(ac[a], dc[a], ac[b], dc[b])

            windows.append(PPGWindow(**f))

    return windows
