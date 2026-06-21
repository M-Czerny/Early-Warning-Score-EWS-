"""
Signal acquisition: discover and load raw PPG and SpO2 files.

Two separate devices produce two files per session:
  - PPG file   : named as a Unix-millisecond timestamp (e.g. 1776952668764.txt).
                 Contains 4 optical LED channels (IR, Red, Green, Blue) and a
                 3-axis accelerometer, all sampled at 50 Hz.
  - SpO2 file  : named opensignals_*.txt (BITalino / OpenSignals format).
                 Contains reference pulse oximetry (SpO2 %) alongside its own
                 Red and IR channels, sampled at 100 Hz.
"""

import os
import re
import pandas as pd


def _find_ppg_file(directory):
    """Find the UTC-timestamp-named PPG .txt file in a directory."""
    for fname in os.listdir(directory):
        if fname.endswith('.txt') and re.match(r'^\d+\.txt$', fname):
            return os.path.join(directory, fname)
    return None


def _find_spo2_file(directory):
    """Find the opensignals .txt file (reference SpO2 device) in a directory."""
    for fname in os.listdir(directory):
        if fname.startswith('opensignals') and fname.endswith('.txt'):
            return os.path.join(directory, fname)
    return None


def _load_ppg(filepath):
    """Load 4 optical PPG channels and 3-axis accelerometer from the prototype device file."""
    df = pd.read_csv(
        filepath,
        sep='\t',
        usecols=['IR', 'Green', 'Red', 'Blue', 'ACCx', 'ACCy', 'ACCz'],
    )
    return {
        'ir':    df['IR'].to_numpy(),
        'green': df['Green'].to_numpy(),
        'red':   df['Red'].to_numpy(),
        'blue':  df['Blue'].to_numpy(),
        'acc_x': df['ACCx'].to_numpy(),
        'acc_y': df['ACCy'].to_numpy(),
        'acc_z': df['ACCz'].to_numpy(),
    }


def _load_spo2(filepath):
    """Load reference SpO2 (%) plus its Red and IR channels from an opensignals file."""
    # Rows have a trailing tab, so pandas sees an extra empty column — use
    # positional indexing (cols 2=red, 3=ir, 4=spo2) to stay robust.
    df = pd.read_csv(
        filepath,
        sep='\t',
        skiprows=3,   # skip two comment lines and the JSON header
        header=None,
        usecols=[2, 3, 4],
    )
    df.columns = ['red', 'ir', 'spo2']
    return {
        'red':  df['red'].to_numpy(),
        'ir':   df['ir'].to_numpy(),
        'spo2': df['spo2'].to_numpy(),
    }


def _load_session(directory, subject, episode=None):
    """Load one recording session: find both device files and return a record dict."""
    ppg_path  = _find_ppg_file(directory)
    spo2_path = _find_spo2_file(directory)

    if ppg_path is None:
        raise FileNotFoundError(f"No PPG file found in {directory}")
    if spo2_path is None:
        raise FileNotFoundError(f"No SpO2 file found in {directory}")

    return {
        'subject':   subject,
        'episode':   episode,
        'ppg_path':  ppg_path,
        'spo2_path': spo2_path,
        'ppg':       _load_ppg(ppg_path),
        'spo2':      _load_spo2(spo2_path),
    }


def load_all_subjects(data_dir):
    """
    Walk data_dir and load every subject/episode.

    Supports two folder layouts:
      - Flat   : prototype_data/subject1/<files>
      - Nested : prototype_data/subject1/episode 1/<files>

    Returns a list of dicts, each with keys:
        subject (str), episode (str | None), ppg (dict), spo2 (dict)

    ppg keys  : ir, green, red, blue, acc_x, acc_y, acc_z  (numpy arrays)
    spo2 keys : red, ir, spo2                               (numpy arrays)
    """
    records = []

    subject_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('subject')
    )

    for subject in subject_dirs:
        subject_path = os.path.join(data_dir, subject)

        if _find_ppg_file(subject_path) is not None:
            # Flat subject — data lives directly in the subject folder.
            records.append(_load_session(subject_path, subject, episode=None))
        else:
            # Subject has episode sub-folders.
            episode_dirs = sorted(
                d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d))
            )
            for episode in episode_dirs:
                episode_path = os.path.join(subject_path, episode)
                records.append(_load_session(episode_path, subject, episode=episode))

    return records
