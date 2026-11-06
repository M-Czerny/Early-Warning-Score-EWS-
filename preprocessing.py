"""
Preprocessing: line up the two devices in time, smooth the PPG, and remove
samples where the reference SpO2 cannot be trusted.

The wristband and the reference oximeter are started separately and have
their own clocks, so their recordings must be aligned first.
"""

import datetime
import json
import os

import numpy as np
from scipy.signal import savgol_filter

PPG_FS = 50  # Hz. Wristband sampling rate (it is not stored in the file)

# Only the light channels are smoothed; the accelerometer is left as it is.
_PPG_OPTICAL = ('red', 'ir', 'green', 'blue')


def apply_savgol(records: list, window_length: int = 9, polyorder: int = 2) -> list:
    """
    Smooth the red, IR, green and blue channels of every record (in place).

    A Savitzky-Golay filter fits a small polynomial around each sample.  It
    removes fast noise but keeps the pulse shape better than a moving average.
    An even window_length is increased by one, because it must be odd.
    """
    if window_length % 2 == 0:
        window_length += 1

    for rec in records:
        for ch in _PPG_OPTICAL:
            if ch in rec['ppg']:
                rec['ppg'][ch] = savgol_filter(rec['ppg'][ch].astype(float),
                                               window_length=window_length,
                                               polyorder=polyorder)
    return records


def _parse_ppg_start_utc(ppg_path):
    """Wristband start time (UTC).  The file name is the start time in Unix milliseconds."""
    ts_ms = int(os.path.basename(ppg_path).replace('.txt', ''))
    utc = datetime.datetime.fromtimestamp(ts_ms / 1000.0, datetime.timezone.utc)
    return utc.replace(tzinfo=None)


def _parse_spo2_header(spo2_path):
    """(start time, sampling rate) from the OpenSignals header.  The time is the device's local clock."""
    with open(spo2_path) as f:
        f.readline()                          # "# OpenSignals Text File Format..."
        meta = json.loads(f.readline().lstrip('# '))

    info = meta[next(iter(meta))]             # the header is keyed by device id
    # e.g. date "2026-4-1", time "16:26:59.60" (".60" is read as 0.6 s)
    start = datetime.datetime.strptime(f"{info['date']} {info['time']}",
                                       '%Y-%m-%d %H:%M:%S.%f')
    return start, int(info['sampling rate'])


def _correct_timezone(spo2_start_local, ppg_start_utc):
    """
    Turn the oximeter's local time into UTC.

    The time zone is not stored, so we round the difference between the two
    start times to whole hours and remove that offset.
    """
    offset_hours = round((spo2_start_local - ppg_start_utc).total_seconds() / 3600)
    return spo2_start_local - datetime.timedelta(hours=offset_hours)


def _trim(sig: dict, n: int) -> dict:
    """Drop the first n samples of every channel."""
    return {k: v[n:] for k, v in sig.items()}


def synchronise(records: list) -> list:
    """
    Align the two devices: cut the start of whichever recording began first.

    Adds to each record:
        ppg_fs, spo2_fs  sampling rates (Hz)
        sync_offset_s    how many seconds after the wristband the oximeter started
                         (negative = the oximeter started first)
    """
    for rec in records:
        ppg_start           = _parse_ppg_start_utc(rec['ppg_path'])
        spo2_local, spo2_fs = _parse_spo2_header(rec['spo2_path'])
        spo2_start          = _correct_timezone(spo2_local, ppg_start)

        offset_s = (spo2_start - ppg_start).total_seconds()
        if offset_s > 0:
            rec['ppg'] = _trim(rec['ppg'], round(offset_s * PPG_FS))
        elif offset_s < 0:
            rec['spo2'] = _trim(rec['spo2'], round(-offset_s * spo2_fs))

        rec['ppg_fs']        = PPG_FS
        rec['spo2_fs']       = spo2_fs
        rec['sync_offset_s'] = offset_s
    return records


def remove_artifacts(records: list, min_spo2: float = 90.0, spo2_drop: float = 3.0) -> list:
    """
    Remove samples where the reference SpO2 cannot be trusted (in place).

    A SpO2 sample is bad when:
      - it is at or below min_spo2 (probe lost or reading too low), or
      - it falls by spo2_drop % or more in one step (probe slipped off).
    The same moments are removed from the PPG, so both stay aligned.
    Movement is handled later, per window, by the SQI filter.

    Adds to each record:
        n_ppg_removed, n_spo2_removed   number of samples removed
        ppg_original, spo2_original     signals before removal (for the plots)
        ppg_bad_mask, spo2_bad_mask     True where a sample was removed
        ppg_clean_indices               original position of each kept PPG sample
    """
    for rec in records:
        ratio     = rec['spo2_fs'] / rec['ppg_fs']   # e.g. 100 Hz / 50 Hz = 2
        spo2_vals = rec['spo2']['spo2'].astype(float)
        n_spo2    = len(spo2_vals)
        n_ppg     = len(rec['ppg']['ir'])

        # Find bad samples at the SpO2 rate.
        too_low  = spo2_vals <= min_spo2
        big_drop = np.diff(spo2_vals, prepend=spo2_vals[0]) <= -spo2_drop
        spo2_bad = too_low | big_drop

        # Mark the same moments at the PPG rate (nearest SpO2 sample).
        spo2_idx = np.clip(np.round(np.arange(n_ppg) * ratio).astype(int), 0, n_spo2 - 1)
        ppg_bad  = spo2_bad[spo2_idx]

        rec['ppg_original']      = {k: v.copy() for k, v in rec['ppg'].items()}
        rec['spo2_original']     = {k: v.copy() for k, v in rec['spo2'].items()}
        rec['ppg_bad_mask']      = ppg_bad
        rec['spo2_bad_mask']     = spo2_bad
        rec['ppg_clean_indices'] = np.where(~ppg_bad)[0]
        rec['n_ppg_removed']     = int(ppg_bad.sum())
        rec['n_spo2_removed']    = int(spo2_bad.sum())

        rec['ppg']  = {k: v[~ppg_bad]  for k, v in rec['ppg'].items()}
        rec['spo2'] = {k: v[~spo2_bad] for k, v in rec['spo2'].items()}
    return records
