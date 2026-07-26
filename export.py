"""
CSV export of the per-window feature table.

One row per extracted window, one column per PPGWindow field, plus the SQI
outcome.  This is a superset of the feature matrix handed to the regressors in
calibration.calibrate_multi: selecting a combination's fields (see
calibration.LED_COMBOS / LED_COMBOS_ACDC) from the rows where passed_sqi is
True reproduces exactly what that regressor was trained and tested on.

Columns are derived from the PPGWindow dataclass and bucketed by name, so a
field added there appears in the CSV automatically.
"""

from __future__ import annotations

import os
from dataclasses import fields as dataclass_fields
from typing import List

import numpy as np
import pandas as pd

from data_extraction import (PPGWindow, composite_sqi_score,
                             fit_composite_sqi_scaler)


# Fields that identify a window rather than describe it.
_IDENTITY = ('subject', 'episode', 'window_idx', 't_start_s')
# The regression target.
_TARGET   = ('spo2_ref',)


def _column_order() -> List[str]:
    """
    PPGWindow field names grouped for readability: identity, target, AC/DC
    features, R features, then quality metrics.

    Grouping is by name so new fields land in the right block without this
    module needing to be updated; anything unrecognised falls through to the
    quality group rather than being dropped.
    """
    names  = [f.name for f in dataclass_fields(PPGWindow)]
    ac_dc  = [n for n in names if n.startswith(('ac_', 'dc_'))]
    ratios = [n for n in names if n.startswith('R_')]
    rest   = [n for n in names
              if n not in _IDENTITY and n not in _TARGET
              and n not in ac_dc and n not in ratios]
    return list(_IDENTITY) + list(_TARGET) + ac_dc + ratios + rest


def export_windows_csv(
    all_windows:  List[PPGWindow],
    kept_windows: List[PPGWindow],
    out_dir:      str,
    filename:     str = 'window_features.csv',
) -> str:
    """
    Write every extracted window to a CSV inside out_dir.

    Parameters
    ----------
    all_windows  : every PPGWindow before SQI filtering
    kept_windows : the subset that passed the SQI filter (identity-compared,
                   so either SQI mode works)
    out_dir      : destination directory, normally the timestamped run folder
                   returned by plotting.plot_all
    filename     : output file name

    Added columns
    -------------
    passed_sqi    : True if the window survived the SQI filter
    composite_sqi : composite quality score normalised to [0, 1] over the full
                    pre-SQI population (the same scale as SQI_COMPOSITE_MIN)

    Empty cells are NaN features — calibrate_multi silently drops those rows,
    so they never reach a regressor even when passed_sqi is True.

    Returns
    -------
    Path to the written file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    cols = _column_order()

    if not all_windows:
        pd.DataFrame(columns=cols + ['passed_sqi', 'composite_sqi']).to_csv(
            path, index=False)
        print(f'  Saved {filename} (no windows to export)')
        return path

    kept_ids = {id(w) for w in kept_windows}

    # Normalise the composite score over the full pre-SQI population so the
    # column is on the same scale as main.SQI_COMPOSITE_MIN.
    lo, hi = fit_composite_sqi_scaler(all_windows)
    span   = (hi - lo) if hi > lo else None

    rows = []
    for w in all_windows:
        row   = {c: getattr(w, c) for c in cols}
        score = composite_sqi_score(w)
        row['passed_sqi']    = id(w) in kept_ids
        row['composite_sqi'] = (score - lo) / span if span is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols + ['passed_sqi', 'composite_sqi'])
    df.to_csv(path, index=False)
    print(f'  Saved {filename}  ({len(df)} windows, {len(df.columns)} columns, '
          f'{int(df["passed_sqi"].sum())} passed SQI)')
    return path
