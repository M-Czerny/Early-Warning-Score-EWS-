"""
Save every window and its features to a CSV file.

One row per window, one column per PPGWindow field, plus:
  passed_sqi     True if the window passed the SQI filter
  composite_sqi  composite quality score, scaled to [0, 1] over all windows

To get exactly what a model was trained on, keep the rows where passed_sqi is
True and pick that combination's columns (calibration.LED_COMBOS and friends).
Rows with an empty (NaN) feature are skipped by the models.
"""

from __future__ import annotations

import os
from dataclasses import fields as dataclass_fields
from typing import List

import pandas as pd

from data_extraction import PPGWindow
from sqi import fit_composite_sqi_scaler, normalised_composite

_IDENTITY = ('subject', 'episode', 'window_idx', 't_start_s')
_TARGET   = ('spo2_ref',)


def _column_order() -> List[str]:
    """Columns grouped as: identity, target, AC/DC, R values, everything else."""
    names  = [f.name for f in dataclass_fields(PPGWindow)]
    ac_dc  = [n for n in names if n.startswith(('ac_', 'dc_', 'acdc_'))]
    ratios = [n for n in names if n.startswith('R_')]
    rest   = [n for n in names if n not in (*_IDENTITY, *_TARGET, *ac_dc, *ratios)]
    return [*_IDENTITY, *_TARGET, *ac_dc, *ratios, *rest]


def export_windows_csv(
    all_windows:  List[PPGWindow],
    kept_windows: List[PPGWindow],
    out_dir:      str,
    filename:     str = 'window_features.csv',
) -> str:
    """Write all windows to out_dir/filename and return the file path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    cols = _column_order() + ['passed_sqi', 'composite_sqi']

    kept_ids = {id(w) for w in kept_windows}
    df = pd.DataFrame([{c: getattr(w, c) for c in _column_order()} for w in all_windows],
                      columns=cols)
    if all_windows:
        df['passed_sqi']    = [id(w) in kept_ids for w in all_windows]
        df['composite_sqi'] = normalised_composite(all_windows,
                                                   fit_composite_sqi_scaler(all_windows))
    df.to_csv(path, index=False)
    print(f'  Saved {filename}  ({len(df)} windows, {len(df.columns)} columns, '
          f'{int(df["passed_sqi"].sum())} passed SQI)')
    return path
