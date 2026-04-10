"""
utils.py
~~~~~~~~
Small utility functions shared across the package.
"""
 
import json
from datetime import datetime
from pathlib import Path
from typing import Any
 
import numpy as np
import pandas as pd
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def get_datetime(session_path) -> datetime:
    """
    Parse the JSON header comment from the reference H5/TXT session file
    and return the recording start as a Python datetime.
    """
    with open(str(session_path), "r") as f:
        for line in f:
            if line.startswith("# {"):
                header = json.loads(line[2:].strip())
                break
        else:
            raise ValueError(f"No JSON header found in {session_path}")
 
    device_key = next(iter(header))
    date_str   = header[device_key]["date"]
    time_str   = header[device_key]["time"]
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
 
 
def timediff(
    reference_file,
    prototype_file,
    filename: str,
    window_sec: float,
    ref_timezone_offset_hours: float,
) -> float:
    """
    Compute the wall-clock difference (in seconds) between the start of
    the prototype recording (encoded in its filename as a Unix-ms timestamp)
    and the reference device's start timestamp embedded in its header.
 
    ``window_sec`` is added so that feature-R values can start being
    computed from the first full window.
    """
    proto_df = pd.read_csv(str(prototype_file), delimiter="\t")
    proto_ts_ms = int(Path(filename).stem)          # filename without extension
 
    proto_dt = datetime.fromtimestamp(proto_ts_ms / 1000.0)
    ref_dt   = get_datetime(reference_file)
 
    diff = (proto_dt - ref_dt).total_seconds() + window_sec
    return diff
 
 
# ─────────────────────────────────────────────────────────────────────────────
# File format inference
# ─────────────────────────────────────────────────────────────────────────────
 
def infer_file_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    if suffix == ".csv":
        return "csv"
    if suffix in {".txt", ".tsv", ".json"}:
        return "txt"
    raise ValueError(f"Unsupported file extension: {suffix}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Time-axis parsing
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_time_to_seconds(x: Any) -> float:
    """
    Flexible parser: handles raw floats (Unix s or ms), HH:MM:SS.mmm strings,
    and ISO datetime strings.
    """
    if pd.isna(x):
        return np.nan
 
    if isinstance(x, (int, float, np.integer, np.floating)):
        t = float(x)
        # Unix milliseconds heuristic
        if t > 1e10:
            return t / 1000.0
        return t
 
    s = str(x).strip()
    if not s:
        return np.nan
 
    parts = s.split(":")
    try:
        if len(parts) == 3:
            a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
            if a >= 1 and b < 60 and c < 60:
                return a * 3600 + b * 60 + c
            return a * 60 + b + c / 1000.0
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except (ValueError, IndexError):
        pass
 
    try:
        return pd.to_datetime(s).timestamp()
    except Exception:
        pass
 
    try:
        return float(s)
    except Exception as exc:
        raise ValueError(f"Cannot parse time value: {x!r}") from exc
 
 
def make_relative_time_seconds(series: pd.Series) -> np.ndarray:
    """
    Convert a raw time series (any supported format) to a relative float
    array starting at 0.0 seconds.
    """
    t = series.apply(parse_time_to_seconds).astype(float).to_numpy()
    if np.nanmedian(t) > 1e8:
        t = t - t[0]
    else:
        t = t - np.nanmin(t)
    return t
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Signal utilities
# ─────────────────────────────────────────────────────────────────────────────
 
def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-score normalise, returning a float array."""
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)