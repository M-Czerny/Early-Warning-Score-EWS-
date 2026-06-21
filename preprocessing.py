"""
Preprocessing: synchronise device clocks, denoise, and remove bad samples.

The two recording devices (prototype PPG sensor and reference pulse oximeter)
start independently and use different clocks, so their signals must be
time-aligned before any analysis.  Bad reference SpO2 values (probe loss,
sudden drops) are then removed along with the corresponding PPG samples.
"""

import json
import os
import datetime

import numpy as np
from scipy.signal import savgol_filter

# PPG sampling rate — not encoded in the PPG file header.
PPG_FS = 50  # Hz

# Only optical channels are denoised; the accelerometer is left unchanged.
_PPG_OPTICAL = ('red', 'ir', 'green', 'blue')


def apply_savgol(
    records:       list,
    window_length: int = 9,
    polyorder:     int = 2,
) -> list:
    """
    Apply a Savitzky-Golay filter to the optical PPG channels of every record.

    Savitzky-Golay fits a polynomial through a sliding window and replaces each
    sample with the polynomial value — this removes high-frequency noise while
    preserving the waveform amplitude better than a plain moving average.

    Only the optical channels (red, IR, green, blue) are filtered;
    accelerometer channels are left unchanged.

    window_length is silently incremented to the next odd value if even.
    """
    if window_length % 2 == 0:
        window_length += 1

    for rec in records:
        for ch in _PPG_OPTICAL:
            if ch in rec['ppg']:
                rec['ppg'][ch] = savgol_filter(
                    rec['ppg'][ch].astype(float),
                    window_length=window_length,
                    polyorder=polyorder,
                )
    return records


def _parse_ppg_start_utc(ppg_path):
    """Return PPG recording start as a UTC datetime from the filename timestamp (Unix ms)."""
    ts_ms = int(os.path.basename(ppg_path).replace('.txt', ''))
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000.0)


def _parse_spo2_header(spo2_path):
    """
    Return (start_naive_dt, spo2_fs) from the opensignals header.

    The datetime is naive (no tzinfo) — it reflects the device's local clock.
    """
    with open(spo2_path) as f:
        f.readline()                          # "# OpenSignals Text File Format…"
        meta = json.loads(f.readline().lstrip('# '))

    device = next(iter(meta))
    info   = meta[device]

    # date: "2026-4-1", time: "16:26:59.60"
    # %f in strptime pads right to 6 digits, so ".60" → 600 000 µs = 0.6 s ✓
    start = datetime.datetime.strptime(
        f"{info['date']} {info['time']}", '%Y-%m-%d %H:%M:%S.%f'
    )
    return start, int(info['sampling rate'])


def _correct_timezone(spo2_start_naive, ppg_start_utc):
    """
    Convert spo2_start_naive to UTC by detecting whole-hour offsets.

    If the raw difference is within ±30 min of a whole-hour multiple the device
    clock is in a fixed UTC offset — subtract that many hours.
    """
    raw_diff_s   = (spo2_start_naive - ppg_start_utc).total_seconds()
    offset_hours = round(raw_diff_s / 3600)

    if abs(offset_hours) >= 1:
        return spo2_start_naive - datetime.timedelta(hours=offset_hours)
    return spo2_start_naive   # already UTC (or sub-minute difference)


def _trim(sig: dict, n: int) -> dict:
    """Drop the first n samples from every array in sig."""
    return {k: v[n:] for k, v in sig.items()}


def synchronise(records: list) -> list:
    """
    Time-synchronise PPG and SpO2 for every record by trimming the earlier signal.

    The PPG start time is embedded in its filename (Unix ms).
    The SpO2 start time is in the opensignals JSON header (local clock).
    After timezone correction, the positive offset means PPG started first, so
    PPG samples before the SpO2 start are discarded, and vice versa.

    Each record gains three new keys after this call:
        ppg_fs        (int)   – PPG sampling rate in Hz
        spo2_fs       (int)   – SpO2 sampling rate in Hz
        sync_offset_s (float) – seconds SpO2 started after PPG (negative = SpO2 started first)
    """
    for rec in records:
        ppg_start_utc             = _parse_ppg_start_utc(rec['ppg_path'])
        spo2_start_naive, spo2_fs = _parse_spo2_header(rec['spo2_path'])
        spo2_start_utc            = _correct_timezone(spo2_start_naive, ppg_start_utc)

        # Positive offset means SpO2 started later → trim the PPG leading samples.
        offset_s = (spo2_start_utc - ppg_start_utc).total_seconds()

        if offset_s > 0:
            n_cut = round(offset_s * PPG_FS)
            rec['ppg'] = _trim(rec['ppg'], n_cut)
        elif offset_s < 0:
            n_cut = round(-offset_s * spo2_fs)
            rec['spo2'] = _trim(rec['spo2'], n_cut)

        rec['ppg_fs']        = PPG_FS
        rec['spo2_fs']       = spo2_fs
        rec['sync_offset_s'] = offset_s

    return records


def remove_artifacts(
    records:   list,
    min_spo2:  float = 90.0,
    spo2_drop: float = 3.0,
) -> list:
    """
    Remove bad samples from every record in-place.

    A SpO2 sample is flagged as bad if any of the following hold:
      • SpO2 ≤ min_spo2 — value is too low or probe has lost contact
      • SpO2 drops ≥ spo2_drop % in a single step — sudden probe-off event

    The bad mask is built at the SpO2 rate and then mapped onto the higher PPG
    rate so both signals lose exactly the same time windows.

    Motion artefact rejection is handled later at the window level via
    acc_std_max in data_extraction.apply_sqi.

    Each record gains these new keys:
        n_ppg_removed    (int)        – number of PPG samples removed
        n_spo2_removed   (int)        – number of SpO2 samples removed
        ppg_original     (dict)       – unmasked PPG signals (for plotting)
        spo2_original    (dict)       – unmasked SpO2 signals (for plotting)
        ppg_bad_mask     (bool array) – True where PPG sample was removed
        spo2_bad_mask    (bool array) – True where SpO2 sample was removed
        ppg_clean_indices(int array)  – original indices of surviving PPG samples
    """
    for rec in records:
        ppg_fs  = rec['ppg_fs']
        spo2_fs = rec['spo2_fs']
        # SpO2 is sampled at twice the PPG rate (typically 100 Hz vs 50 Hz).
        ratio   = spo2_fs / ppg_fs

        spo2_vals = rec['spo2']['spo2'].astype(float)
        n_spo2    = len(spo2_vals)
        n_ppg     = len(rec['ppg']['ir'])

        # Build bad-sample mask at the SpO2 rate.
        low_mask  = spo2_vals <= min_spo2
        drop_mask = np.diff(spo2_vals, prepend=spo2_vals[0]) <= -spo2_drop
        spo2_bad  = low_mask | drop_mask   # shape: (n_spo2,)

        # Map SpO2 bad mask to PPG rate by nearest-neighbour indexing.
        spo2_idx      = np.clip(np.round(np.arange(n_ppg) * ratio).astype(int), 0, n_spo2 - 1)
        ppg_bad       = spo2_bad[spo2_idx]   # shape: (n_ppg,)
        spo2_bad_full = spo2_bad

        # Keep copies of the original (pre-removal) signals for diagnostic plots.
        rec['ppg_original']      = {k: v.copy() for k, v in rec['ppg'].items()}
        rec['spo2_original']     = {k: v.copy() for k, v in rec['spo2'].items()}
        rec['ppg_bad_mask']      = ppg_bad.copy()
        rec['spo2_bad_mask']     = spo2_bad_full.copy()
        rec['ppg_clean_indices'] = np.where(~ppg_bad)[0]

        # Remove flagged samples from both signals.
        rec['n_ppg_removed']  = int(ppg_bad.sum())
        rec['n_spo2_removed'] = int(spo2_bad_full.sum())

        rec['ppg']  = {k: v[~ppg_bad]       for k, v in rec['ppg'].items()}
        rec['spo2'] = {k: v[~spo2_bad_full] for k, v in rec['spo2'].items()}

    return records
