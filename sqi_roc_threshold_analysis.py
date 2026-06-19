"""
sqi_roc_threshold_analysis.py
------------------------------
ROC-based threshold optimisation for PPG Signal Quality Index (SQI) metrics.

Generates per-metric ROC curves, a composite SQI score, sensitivity analysis,
and four publication-quality figures for use in a bachelor's thesis.

METHODS_NOTE
------------
Signal quality index (SQI) thresholds were optimised using receiver operating
characteristic (ROC) analysis. Windows were labelled as ACCEPT or REJECT based
on whether the absolute difference between the predicted and reference SpO2
exceeded 3.5 percentage points (ARMS threshold). For each SQI metric—perfusion
index, skewness, kurtosis, signal-to-noise ratio, spectral entropy, relative
power, and acceleration standard deviation—100 candidate thresholds were swept
between the 1st and 99th sample percentiles. At each threshold, sensitivity
(true-positive rate) and specificity (true-negative rate) were computed and the
area under the ROC curve (AUC) quantified discriminative ability. The optimal
threshold was selected via the Youden index (sensitivity + specificity − 1). A
composite SQI score was formed as a weighted, min-max normalised linear
combination of individual metrics (weights: PI 0.30, skewness 0.25, SNR 0.20,
relative power 0.10, kurtosis 0.10, entropy 0.05), with directionality adjusted
so that higher scores always indicate better quality. Robustness was assessed by
applying ±10% and ±20% perturbations to the Youden-optimal thresholds and
recording the resulting change in sensitivity and specificity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc as sklearn_auc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARMS_REJECT_THRESHOLD_DEFAULT = 3.5   # % SpO2 error above which window is REJECT

# SQI metrics analysed (column names in windows_df)
METRICS_SINGLE = [
    'pi',
    'skewness_ir',
    'kurtosis_ir',
    'snr_ir',
    'entropy_ir',
    'rel_power_ir',
    'acc_energy_std',
]

METRIC_LABELS = {
    'pi':             'Perfusion Index',
    'skewness_ir':    'Skewness (IR)',
    'kurtosis_ir':    'Kurtosis (IR)',
    'snr_ir':         'SNR (IR) [dB]',
    'entropy_ir':     'Spectral Entropy (IR)',
    'rel_power_ir':   'Relative Power (IR)',
    'acc_energy_std': 'Acceleration Std-dev',
    'composite_sqi':  'Composite SQI',
}

# Composite SQI weights; metrics marked True are inverted (low raw = good quality)
COMPOSITE_WEIGHTS = {
    'pi':           (0.30, False),   # high PI = good
    'skewness_ir':  (0.25, True),    # low |skewness| = good; we use abs → invert
    'snr_ir':       (0.20, False),   # high SNR = good
    'rel_power_ir': (0.10, False),   # high rel_power = good
    'kurtosis_ir':  (0.10, True),    # low excess kurtosis = good → invert
    'entropy_ir':   (0.05, True),    # low entropy = more concentrated → good → invert
}

N_THRESHOLDS = 100


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _make_labels(
    windows_df: pd.DataFrame,
    arms_reject_threshold: float,
) -> np.ndarray:
    """
    Return binary array: 1 = ACCEPT (good quality), 0 = REJECT (poor quality).

    A window is REJECT when |spo2_pred_mean - spo2_ref_mean| > threshold.
    Rows missing spo2_pred_mean or spo2_ref_mean are assigned REJECT.
    """
    if 'spo2_pred_mean' not in windows_df.columns:
        raise KeyError(
            "windows_df must contain 'spo2_pred_mean'. "
            "Run LOSO calibration first and merge mean predictions onto the window table."
        )
    err = (windows_df['spo2_pred_mean'] - windows_df['spo2_ref']).abs()
    labels = np.where(err <= arms_reject_threshold, 1, 0)
    nan_mask = windows_df['spo2_pred_mean'].isna() | windows_df['spo2_ref'].isna()
    labels[nan_mask.values] = 0
    return labels


# ---------------------------------------------------------------------------
# Composite SQI
# ---------------------------------------------------------------------------

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def compute_composite_sqi(windows_df: pd.DataFrame) -> np.ndarray:
    """
    Compute composite SQI score (0–1, higher = better quality) for every row.

    Uses COMPOSITE_WEIGHTS; missing columns are skipped with a warning.
    """
    score = np.zeros(len(windows_df), dtype=float)
    total_weight = 0.0

    for col, (weight, invert) in COMPOSITE_WEIGHTS.items():
        if col not in windows_df.columns:
            print(f"  [composite SQI] column '{col}' missing — skipped")
            continue
        vals = windows_df[col].fillna(windows_df[col].median()).values.astype(float)
        normed = _minmax(vals)
        if invert:
            normed = 1.0 - normed
        score += weight * normed
        total_weight += weight

    if total_weight > 0:
        score /= total_weight
    return score


# ---------------------------------------------------------------------------
# ROC analysis for one metric
# ---------------------------------------------------------------------------

def _roc_one_metric(
    values: np.ndarray,
    labels: np.ndarray,
    high_is_good: bool,
) -> dict:
    """
    Sweep 100 thresholds and compute ROC metrics.

    high_is_good : True  → threshold is a lower bound (accept if value >= thr)
                   False → threshold is an upper bound (accept if value <= thr)

    Returns dict with keys: thresholds, sens, spec, ppv, npv, f1, auc, youden_thr,
    youden_idx, opt_sens, opt_spec, opt_ppv, opt_npv, opt_f1.
    """
    valid = ~np.isnan(values)
    v = values[valid]
    y = labels[valid]

    thr_grid = np.percentile(v, np.linspace(1, 99, N_THRESHOLDS))
    thr_grid = np.unique(thr_grid)  # remove duplicates from flat distributions

    sens_arr = np.zeros(len(thr_grid))
    spec_arr = np.zeros(len(thr_grid))
    ppv_arr  = np.zeros(len(thr_grid))
    npv_arr  = np.zeros(len(thr_grid))
    f1_arr   = np.zeros(len(thr_grid))

    for i, thr in enumerate(thr_grid):
        pred = (v >= thr).astype(int) if high_is_good else (v <= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        sens_arr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_arr[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv_arr[i]  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv_arr[i]  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1_arr[i]   = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    # AUC via trapezoidal integration over (1-spec, sens)
    fpr = 1.0 - spec_arr
    sort_idx = np.argsort(fpr)
    roc_auc = float(sklearn_auc(fpr[sort_idx], sens_arr[sort_idx]))

    # Youden index
    youden = sens_arr + spec_arr - 1.0
    best_i = int(np.argmax(youden))

    return {
        'thresholds': thr_grid,
        'sens':       sens_arr,
        'spec':       spec_arr,
        'ppv':        ppv_arr,
        'npv':        npv_arr,
        'f1':         f1_arr,
        'fpr':        fpr,
        'auc':        roc_auc,
        'youden_thr': float(thr_grid[best_i]),
        'youden_idx': float(youden[best_i]),
        'opt_sens':   float(sens_arr[best_i]),
        'opt_spec':   float(spec_arr[best_i]),
        'opt_ppv':    float(ppv_arr[best_i]),
        'opt_npv':    float(npv_arr[best_i]),
        'opt_f1':     float(f1_arr[best_i]),
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def _sensitivity_analysis(
    roc_results: dict,
    windows_df:  pd.DataFrame,
    labels:      np.ndarray,
) -> pd.DataFrame:
    """±10% / ±20% perturbation of optimal thresholds."""
    rows = []
    perturbations = [-0.20, -0.10, 0.00, +0.10, +0.20]

    all_metrics = list(roc_results.keys())

    for metric in all_metrics:
        res = roc_results[metric]
        opt = res['youden_thr']
        high = metric in ('pi', 'snr_ir', 'rel_power_ir', 'composite_sqi')

        col = metric
        if metric not in windows_df.columns:
            continue
        vals = windows_df[col].values.astype(float)
        valid = ~np.isnan(vals)
        v = vals[valid]
        y = labels[valid]

        for p in perturbations:
            thr = opt * (1.0 + p)
            pred = (v >= thr).astype(int) if high else (v <= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            rows.append({
                'metric':       metric,
                'perturbation': f'{p:+.0%}',
                'threshold':    round(thr, 4),
                'sensitivity':  round(sens, 4),
                'specificity':  round(spec, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _figure_roc_curves(roc_results: dict, save_path: str) -> None:
    """3×3 grid of ROC curves, one per metric (last cell = composite)."""
    metrics = list(roc_results.keys())
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 11), dpi=300)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metrics[:9]):
        ax = axes_flat[idx]
        res = roc_results[metric]
        fpr_sorted = np.sort(res['fpr'])
        sort_idx   = np.argsort(res['fpr'])
        sens_sorted = res['sens'][sort_idx]
        ax.plot(fpr_sorted, sens_sorted, lw=2, color='steelblue',
                label=f"AUC = {res['auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        opt_fpr = 1.0 - res['opt_spec']
        ax.scatter([opt_fpr], [res['opt_sens']], zorder=5, color='crimson', s=60,
                   label=f"Youden thr = {res['youden_thr']:.3g}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('1 − Specificity', fontsize=8)
        ax.set_ylabel('Sensitivity', fontsize=8)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_aspect('equal')

    for idx in range(len(metrics), nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle('ROC Curves — SQI Metrics', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _figure_threshold_sweep(roc_results: dict, save_path: str) -> None:
    """3×3 grid: sensitivity, specificity, F1 vs threshold for each metric."""
    metrics = list(roc_results.keys())
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 11), dpi=300)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metrics[:9]):
        ax = axes_flat[idx]
        res = roc_results[metric]
        t   = res['thresholds']
        ax.plot(t, res['sens'], label='Sensitivity', color='steelblue', lw=1.5)
        ax.plot(t, res['spec'], label='Specificity', color='darkorange', lw=1.5)
        ax.plot(t, res['f1'],   label='F1',          color='forestgreen', lw=1.5, ls='--')
        ax.axvline(res['youden_thr'], color='crimson', lw=1.2, ls=':', label='Youden opt')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=8)
        ax.set_ylabel('Rate', fontsize=8)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='center right')

    for idx in range(len(metrics), nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Threshold Sweep — Sensitivity / Specificity / F1', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _figure_violin(
    windows_df: pd.DataFrame,
    labels:     np.ndarray,
    roc_results: dict,
    save_path:  str,
) -> None:
    """Violin plots: ACCEPT vs REJECT distributions for each SQI metric."""
    metrics = [m for m in roc_results.keys() if m != 'composite_sqi']
    nrows = int(np.ceil(len(metrics) / 3))
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), dpi=300)
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    accept_mask = labels == 1
    reject_mask = labels == 0

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        col = metric
        if col not in windows_df.columns:
            ax.set_visible(False)
            continue
        vals = windows_df[col].values.astype(float)
        data_accept = vals[accept_mask & ~np.isnan(vals)]
        data_reject = vals[reject_mask & ~np.isnan(vals)]

        parts = ax.violinplot([data_accept, data_reject], positions=[0, 1],
                              showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        parts['bodies'][0].set_facecolor('steelblue')
        parts['bodies'][1].set_facecolor('crimson')

        opt_thr = roc_results[metric]['youden_thr']
        ax.axhline(opt_thr, color='black', lw=1.2, ls='--', label=f'opt = {opt_thr:.3g}')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ACCEPT', 'REJECT'])
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9, fontweight='bold')
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=7)

    for idx in range(len(metrics), nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle('SQI Distributions: ACCEPT vs REJECT Windows', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _figure_composite_roc(roc_results: dict, save_path: str) -> None:
    """Single ROC panel comparing composite SQI with individual metrics."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cmap = plt.cm.tab10
    metrics = list(roc_results.keys())

    for i, metric in enumerate(metrics):
        res = roc_results[metric]
        sort_idx = np.argsort(res['fpr'])
        fpr_s    = res['fpr'][sort_idx]
        sens_s   = res['sens'][sort_idx]
        lw  = 2.5 if metric == 'composite_sqi' else 1.0
        ls  = '-'
        col = cmap(i % 10) if metric != 'composite_sqi' else 'black'
        ax.plot(fpr_s, sens_s, lw=lw, ls=ls, color=col,
                label=f"{METRIC_LABELS.get(metric, metric)} (AUC={res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('1 − Specificity', fontsize=11)
    ax.set_ylabel('Sensitivity', fontsize=11)
    ax.set_title('Composite SQI vs Individual Metrics — ROC', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.8)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_roc_analysis(
    windows_df:           pd.DataFrame,
    arms_reject_threshold: float = ARMS_REJECT_THRESHOLD_DEFAULT,
    save_dir:             str   = 'results/sqi_roc',
) -> pd.DataFrame:
    """
    Run full ROC threshold analysis on SQI metrics.

    Parameters
    ----------
    windows_df : DataFrame with columns for SQI metrics plus 'spo2_ref' and
                 'spo2_pred_mean' (mean predicted SpO2 across LOSO folds).
    arms_reject_threshold : ARMS error (%) above which a window is labelled REJECT.
    save_dir : output directory for plots and CSV/JSON artefacts.

    Returns
    -------
    DataFrame: ROC summary table (one row per metric).
    """
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[ROC analysis]  ARMS reject threshold = {arms_reject_threshold} %")
    print(f"[ROC analysis]  windows = {len(windows_df)}")

    # --- Labels ---------------------------------------------------------
    labels = _make_labels(windows_df, arms_reject_threshold)
    n_accept = int(labels.sum())
    n_reject = len(labels) - n_accept
    print(f"[ROC analysis]  ACCEPT = {n_accept}, REJECT = {n_reject}")

    # --- Composite SQI --------------------------------------------------
    windows_df = windows_df.copy()
    windows_df['composite_sqi'] = compute_composite_sqi(windows_df)

    # --- Per-metric ROC -------------------------------------------------
    # Metrics where high value = good quality (use lower-bound threshold)
    high_is_good = {
        'pi':             True,
        'skewness_ir':    False,   # absolute deviation from 0 — lower is better
        'kurtosis_ir':    False,
        'snr_ir':         True,
        'entropy_ir':     False,
        'rel_power_ir':   True,
        'acc_energy_std': False,
        'composite_sqi':  True,
    }

    analyse_metrics = METRICS_SINGLE + ['composite_sqi']
    roc_results: dict = {}

    for metric in analyse_metrics:
        if metric not in windows_df.columns:
            print(f"  [ROC] column '{metric}' not found — skipped")
            continue
        vals = windows_df[metric].values.astype(float)
        roc_results[metric] = _roc_one_metric(vals, labels, high_is_good.get(metric, True))
        res = roc_results[metric]
        print(f"  {METRIC_LABELS.get(metric, metric):<28}  "
              f"AUC={res['auc']:.3f}  Youden_thr={res['youden_thr']:.4g}  "
              f"Sens={res['opt_sens']:.3f}  Spec={res['opt_spec']:.3f}")

    # --- Summary table --------------------------------------------------
    rows = []
    for metric, res in roc_results.items():
        rows.append({
            'metric':            metric,
            'label':             METRIC_LABELS.get(metric, metric),
            'auc':               round(res['auc'], 4),
            'youden_threshold':  round(res['youden_thr'], 4),
            'youden_index':      round(res['youden_idx'], 4),
            'sensitivity':       round(res['opt_sens'], 4),
            'specificity':       round(res['opt_spec'], 4),
            'ppv':               round(res['opt_ppv'], 4),
            'npv':               round(res['opt_npv'], 4),
            'f1':                round(res['opt_f1'], 4),
        })
    summary_df = pd.DataFrame(rows)
    summary_path = out / 'roc_summary_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"[ROC analysis]  saved {summary_path}")

    # --- Sensitivity analysis -------------------------------------------
    sens_df = _sensitivity_analysis(roc_results, windows_df, labels)
    sens_path = out / 'threshold_sensitivity_table.csv'
    sens_df.to_csv(sens_path, index=False)
    print(f"[ROC analysis]  saved {sens_path}")

    # --- Optimal thresholds JSON ----------------------------------------
    opt_json = {
        m: {
            'youden_threshold': res['youden_thr'],
            'auc':              res['auc'],
            'sensitivity':      res['opt_sens'],
            'specificity':      res['opt_spec'],
            'f1':               res['opt_f1'],
        }
        for m, res in roc_results.items()
    }
    json_path = out / 'optimal_thresholds.json'
    with open(json_path, 'w') as fh:
        json.dump(opt_json, fh, indent=2)
    print(f"[ROC analysis]  saved {json_path}")

    # --- Plots ----------------------------------------------------------
    _figure_roc_curves(roc_results,       str(out / 'roc_curves.png'))
    _figure_threshold_sweep(roc_results,  str(out / 'threshold_sweep.png'))
    _figure_violin(windows_df, labels, roc_results, str(out / 'violin_accept_reject.png'))
    _figure_composite_roc(roc_results,    str(out / 'composite_roc.png'))
    print(f"[ROC analysis]  plots saved to {out}/")

    return summary_df


# ---------------------------------------------------------------------------
# Synthetic __main__ demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running ROC threshold analysis with synthetic data...")
    rng = np.random.default_rng(0)
    N = 500

    # Synthetic SQI features
    spo2_ref  = rng.normal(96.0, 2.0, N).clip(88, 100)
    pred_err  = rng.normal(0.0, 2.5, N)
    spo2_pred = spo2_ref + pred_err

    df = pd.DataFrame({
        'spo2_ref':      spo2_ref,
        'spo2_pred_mean': spo2_pred,
        'pi':            rng.uniform(0.001, 0.05, N),
        'skewness_ir':   rng.normal(0.5, 1.0, N),
        'kurtosis_ir':   rng.normal(2.0, 3.0, N),
        'snr_ir':        rng.normal(-5.0, 8.0, N),
        'entropy_ir':    rng.uniform(2.0, 5.0, N),
        'rel_power_ir':  rng.uniform(0.01, 0.5, N),
        'acc_energy_std': rng.exponential(50.0, N),
    })

    # Corrupt high-error windows: lower PI, higher acc_std
    high_err = np.abs(pred_err) > 3.5
    df.loc[high_err, 'pi']            *= 0.3
    df.loc[high_err, 'acc_energy_std'] *= 2.5
    df.loc[high_err, 'snr_ir']        -= 6.0

    summary = run_roc_analysis(df, arms_reject_threshold=3.5, save_dir='results/sqi_roc')
    print("\nROC Summary:")
    print(summary.to_string(index=False))
