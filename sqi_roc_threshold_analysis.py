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
exceeded 3.5 percentage points (ARMS threshold). Each SQI metric was analysed
with a gating mode matched to its physical interpretation:

  - One-sided lower bound (PI, SNR, relative power, composite SQI):
      accept if value >= threshold; 100 candidates swept from p1 to p99.
  - One-sided upper bound (spectral entropy, acceleration std-dev):
      accept if value <= threshold; same sweep.
  - Absolute-value upper bound (skewness):
      accept if |skewness| <= threshold; a symmetric gate around zero.
      100 candidates swept on |skewness| from its p1 to p99.
  - Two-sided band (kurtosis):
      accept if lo <= kurtosis <= hi; a 40×40 grid of (lo, hi) pairs is
      searched jointly to maximise the Youden index, capturing that both
      excessively flat (low) and excessively peaky (high) distributions
      indicate poor signal quality.

For each metric, the optimal threshold was selected via the Youden index
(sensitivity + specificity − 1). The composite SQI score was formed as a
weighted, min-max normalised linear combination of individual metrics (weights:
PI 0.30, skewness 0.25, SNR 0.20, relative power 0.10, kurtosis 0.10,
entropy 0.05), with directionality adjusted so that higher scores always
indicate better quality and skewness entered as its absolute value.
Robustness was assessed by applying ±10% and ±20% perturbations to the
Youden-optimal thresholds and recording the resulting sensitivity and specificity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc as sklearn_auc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARMS_REJECT_THRESHOLD_DEFAULT = 0.5   # % SpO2 error above which window is REJECT

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

# Gating mode for each metric:
#   'lo_bound'     → accept if value >= threshold       (high value = good quality)
#   'hi_bound'     → accept if value <= threshold       (low value = good quality)
#   'abs_hi_bound' → accept if |value| <= threshold     (close to zero = good)
#   'two_sided'    → accept if lo <= value <= hi        (central range = good)
METRIC_MODE: dict[str, str] = {
    'pi':             'lo_bound',
    'skewness_ir':    'abs_hi_bound',  # symmetric around 0; |skewness| must be small
    'kurtosis_ir':    'two_sided',     # neither too flat (low) nor too peaky (high)
    'snr_ir':         'lo_bound',
    'entropy_ir':     'hi_bound',
    'rel_power_ir':   'lo_bound',
    'acc_energy_std': 'hi_bound',
    'composite_sqi':  'lo_bound',
}

# Composite SQI weights; True = invert so that higher score always = better quality.
# Skewness uses |skewness| internally so both tails are penalised equally.
COMPOSITE_WEIGHTS: dict[str, tuple[float, bool]] = {
    'pi':           (0.30, False),   # high PI = good
    'skewness_ir':  (0.25, True),    # low |skewness| = good (abs applied in code)
    'snr_ir':       (0.20, False),   # high SNR = good
    'rel_power_ir': (0.10, False),   # high rel_power = good
    'kurtosis_ir':  (0.10, True),    # low excess kurtosis = good
    'entropy_ir':   (0.05, True),    # low entropy = more periodic = good
}

N_THRESHOLDS    = 100   # candidates for one-sided sweeps
N_THRESHOLDS_2D = 40    # candidates per axis for the two-sided grid search


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _make_labels(
    windows_df: pd.DataFrame,
    arms_reject_threshold: float,
) -> np.ndarray:
    """
    Return binary array: 1 = ACCEPT (good quality), 0 = REJECT (poor quality).

    A window is REJECT when |spo2_pred_mean - spo2_ref| > threshold or values
    are missing.
    """
    if 'spo2_pred_mean' not in windows_df.columns:
        raise KeyError(
            "windows_df must contain 'spo2_pred_mean'. "
            "Run LOSO calibration first and merge mean predictions onto the window table."
        )
    err      = (windows_df['spo2_pred_mean'] - windows_df['spo2_ref']).abs()
    nan_mask = windows_df['spo2_pred_mean'].isna() | windows_df['spo2_ref'].isna()
    labels   = np.where((err <= arms_reject_threshold) & ~nan_mask, 1, 0)
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

    Skewness enters as its absolute value so that both left-skewed and
    right-skewed windows are penalised equally.
    """
    score = np.zeros(len(windows_df), dtype=float)
    total_weight = 0.0

    for col, (weight, invert) in COMPOSITE_WEIGHTS.items():
        if col not in windows_df.columns:
            print(f"  [composite SQI] column '{col}' missing — skipped")
            continue
        vals = windows_df[col].fillna(windows_df[col].median()).to_numpy(dtype=float)
        # Use |skewness| so the composite penalises deviation from zero in both directions.
        if col == 'skewness_ir':
            vals = np.abs(vals)
        normed = _minmax(vals)
        if invert:
            normed = 1.0 - normed
        score        += weight * normed
        total_weight += weight

    if total_weight > 0:
        score /= total_weight
    return score


# ---------------------------------------------------------------------------
# ROC analysis helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    pred: np.ndarray,
    y:    np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Return (sens, spec, ppv, npv, f1) for binary predictions vs labels."""
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn)           if (tp + fn) > 0          else 0.0
    spec = tn / (tn + fp)           if (tn + fp) > 0          else 0.0
    ppv  = tp / (tp + fp)           if (tp + fp) > 0          else 0.0
    npv  = tn / (tn + fn)           if (tn + fn) > 0          else 0.0
    f1   = 2*tp / (2*tp + fp + fn)  if (2*tp + fp + fn) > 0  else 0.0
    return sens, spec, ppv, npv, f1


def _roc_one_sided(
    values:       np.ndarray,
    labels:       np.ndarray,
    high_is_good: bool,
) -> dict:
    """
    Sweep N_THRESHOLDS thresholds for a one-sided acceptance rule.

    high_is_good=True  → lower bound: accept if value >= threshold
    high_is_good=False → upper bound: accept if value <= threshold
    """
    valid = ~np.isnan(values)
    v, y  = values[valid], labels[valid]

    thr_grid = np.unique(np.percentile(v, np.linspace(1, 99, N_THRESHOLDS)))
    n = len(thr_grid)
    sens_arr = np.zeros(n); spec_arr = np.zeros(n)
    ppv_arr  = np.zeros(n); npv_arr  = np.zeros(n); f1_arr = np.zeros(n)

    for i, thr in enumerate(thr_grid):
        pred = (v >= thr).astype(int) if high_is_good else (v <= thr).astype(int)
        sens_arr[i], spec_arr[i], ppv_arr[i], npv_arr[i], f1_arr[i] = _compute_metrics(pred, y)

    fpr      = 1.0 - spec_arr
    sort_idx = np.argsort(fpr)
    roc_auc  = float(sklearn_auc(fpr[sort_idx], sens_arr[sort_idx]))

    youden = sens_arr + spec_arr - 1.0
    best_i = int(np.argmax(youden))

    return {
        'thresholds':    thr_grid,
        'sens':          sens_arr,
        'spec':          spec_arr,
        'ppv':           ppv_arr,
        'npv':           npv_arr,
        'f1':            f1_arr,
        'fpr':           fpr,
        'auc':           roc_auc,
        'two_sided':     False,
        'abs_transformed': False,
        'youden_thr':    float(thr_grid[best_i]),
        'youden_thr_lo': None,
        'youden_thr_hi': None,
        'youden_idx':    float(youden[best_i]),
        'opt_sens':      float(sens_arr[best_i]),
        'opt_spec':      float(spec_arr[best_i]),
        'opt_ppv':       float(ppv_arr[best_i]),
        'opt_npv':       float(npv_arr[best_i]),
        'opt_f1':        float(f1_arr[best_i]),
    }


def _roc_two_sided(
    values: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Two-sided ROC: find the optimal (lo, hi) acceptance band via a 2D grid search.

    Sweeps all (lo, hi) pairs from an N_THRESHOLDS_2D × N_THRESHOLDS_2D grid
    (p1–p99 percentile range) where lo < hi.  The acceptance rule is:
        lo <= value <= hi
    The Youden-optimal pair is selected jointly, not greedily.

    For the ROC curve and threshold sweep plot, the upper bound is then swept
    with lo held at its optimal value so a 1D curve can be shown.
    """
    valid = ~np.isnan(values)
    v, y  = values[valid], labels[valid]

    thr_grid = np.unique(np.percentile(v, np.linspace(1, 99, N_THRESHOLDS_2D)))

    best_youden = -np.inf
    best_lo = thr_grid[0]
    best_hi = thr_grid[-1]
    best_sens = best_spec = best_ppv = best_npv = best_f1 = 0.0

    # Full 2D grid: evaluate every (lo, hi) pair with lo < hi
    for lo in thr_grid:
        for hi in thr_grid:
            if hi <= lo:
                continue
            pred   = ((v >= lo) & (v <= hi)).astype(int)
            sens, spec, ppv, npv, f1 = _compute_metrics(pred, y)
            youden = sens + spec - 1.0
            if youden > best_youden:
                best_youden = youden
                best_lo = lo;   best_hi = hi
                best_sens = sens; best_spec = spec
                best_ppv = ppv;   best_npv = npv; best_f1 = f1

    # Build 1D ROC / sweep curves: hold lo = best_lo, sweep hi
    hi_candidates = thr_grid[thr_grid > best_lo]
    n = len(hi_candidates)
    sens_arr = np.zeros(n); spec_arr = np.zeros(n)
    ppv_arr  = np.zeros(n); npv_arr  = np.zeros(n); f1_arr = np.zeros(n)
    for i, hi in enumerate(hi_candidates):
        pred = ((v >= best_lo) & (v <= hi)).astype(int)
        sens_arr[i], spec_arr[i], ppv_arr[i], npv_arr[i], f1_arr[i] = _compute_metrics(pred, y)

    fpr      = 1.0 - spec_arr
    sort_idx = np.argsort(fpr)
    roc_auc  = float(sklearn_auc(fpr[sort_idx], sens_arr[sort_idx])) if n >= 2 else 0.5

    return {
        'thresholds':    hi_candidates,   # upper bound sweep for plots (lo fixed)
        'sens':          sens_arr,
        'spec':          spec_arr,
        'ppv':           ppv_arr,
        'npv':           npv_arr,
        'f1':            f1_arr,
        'fpr':           fpr,
        'auc':           roc_auc,
        'two_sided':     True,
        'abs_transformed': False,
        'youden_thr':    best_hi,         # backward-compat single value → upper bound
        'youden_thr_lo': best_lo,
        'youden_thr_hi': best_hi,
        'youden_idx':    best_youden,
        'opt_sens':      best_sens,
        'opt_spec':      best_spec,
        'opt_ppv':       best_ppv,
        'opt_npv':       best_npv,
        'opt_f1':        best_f1,
    }


def _roc_for_metric(
    values: np.ndarray,
    labels: np.ndarray,
    mode:   str,
) -> dict:
    """Dispatch ROC analysis to the right function based on the metric's gating mode."""
    if mode == 'lo_bound':
        return _roc_one_sided(values, labels, high_is_good=True)
    elif mode == 'hi_bound':
        return _roc_one_sided(values, labels, high_is_good=False)
    elif mode == 'abs_hi_bound':
        # Run ROC on |value| with an upper-bound sweep.
        # The resulting youden_thr is the max |value| to accept,
        # which maps to symmetric ±thr bounds in original space.
        res = _roc_one_sided(np.abs(values), labels, high_is_good=False)
        res['abs_transformed'] = True
        return res
    elif mode == 'two_sided':
        return _roc_two_sided(values, labels)
    else:
        raise ValueError(f"Unknown METRIC_MODE '{mode}'")


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def _sensitivity_analysis(
    roc_results: dict,
    windows_df:  pd.DataFrame,
    labels:      np.ndarray,
) -> pd.DataFrame:
    """
    ±10% / ±20% perturbation of the Youden-optimal thresholds.

    For two-sided metrics the acceptance window is expanded or contracted
    symmetrically around its centre.  For abs_hi_bound the threshold on
    |value| is perturbed directly.
    """
    rows = []
    perturbations = [-0.20, -0.10, 0.00, +0.10, +0.20]

    for metric, res in roc_results.items():
        if metric not in windows_df.columns:
            continue
        vals  = windows_df[metric].to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        v, y  = vals[valid], labels[valid]
        mode  = METRIC_MODE.get(metric, 'lo_bound')

        for p in perturbations:
            if res['two_sided']:
                # Perturb the half-width while keeping the band centre fixed.
                lo_opt = res['youden_thr_lo']
                hi_opt = res['youden_thr_hi']
                centre     = (lo_opt + hi_opt) / 2.0
                half_width = (hi_opt - lo_opt) / 2.0
                lo   = centre - half_width * (1.0 + p)
                hi   = centre + half_width * (1.0 + p)
                pred = ((v >= lo) & (v <= hi)).astype(int)
                thr_str = f"[{lo:.3g}, {hi:.3g}]"
            elif res['abs_transformed']:
                thr  = res['youden_thr'] * (1.0 + p)
                pred = (np.abs(v) <= thr).astype(int)
                thr_str = f"|val|≤{thr:.3g}"
            elif mode == 'lo_bound':
                thr  = res['youden_thr'] * (1.0 + p)
                pred = (v >= thr).astype(int)
                thr_str = str(round(thr, 4))
            else:  # hi_bound
                thr  = res['youden_thr'] * (1.0 + p)
                pred = (v <= thr).astype(int)
                thr_str = str(round(thr, 4))

            sens, spec, _, _, _ = _compute_metrics(pred, y)
            rows.append({
                'metric':       metric,
                'perturbation': f'{p:+.0%}',
                'threshold':    thr_str,
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
        ax  = axes_flat[idx]
        res = roc_results[metric]
        sort_idx    = np.argsort(res['fpr'])
        ax.plot(res['fpr'][sort_idx], res['sens'][sort_idx],
                lw=2, color='steelblue', label=f"AUC = {res['auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        opt_fpr = 1.0 - res['opt_spec']
        if res['two_sided']:
            thr_label = f"[{res['youden_thr_lo']:.3g}, {res['youden_thr_hi']:.3g}]"
        elif res['abs_transformed']:
            thr_label = f"|val|≤{res['youden_thr']:.3g}"
        else:
            thr_label = f"thr={res['youden_thr']:.3g}"
        ax.scatter([opt_fpr], [res['opt_sens']], zorder=5, color='crimson', s=60,
                   label=thr_label)
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
    """
    3×3 grid: sensitivity, specificity, F1 vs threshold value.

    For two-sided metrics the x-axis shows the upper-bound sweep with
    lo held fixed at its optimal value.  For abs_hi_bound metrics the
    x-axis shows |value|.
    """
    metrics = list(roc_results.keys())
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 11), dpi=300)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metrics[:9]):
        ax  = axes_flat[idx]
        res = roc_results[metric]
        t   = res['thresholds']
        ax.plot(t, res['sens'], label='Sensitivity', color='steelblue',   lw=1.5)
        ax.plot(t, res['spec'], label='Specificity', color='darkorange',   lw=1.5)
        ax.plot(t, res['f1'],   label='F1',          color='forestgreen',  lw=1.5, ls='--')
        ax.axvline(res['youden_thr'], color='crimson', lw=1.2, ls=':', label='Youden opt')
        ax.set_ylim(0, 1.05)

        label = METRIC_LABELS.get(metric, metric)
        if res['two_sided']:
            xlabel = f"{label} (upper bound; lo fixed = {res['youden_thr_lo']:.3g})"
        elif res['abs_transformed']:
            xlabel = f"|{label}|"
        else:
            xlabel = label
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel('Rate', fontsize=8)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='center right')

    for idx in range(len(metrics), nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Threshold Sweep — Sensitivity / Specificity / F1', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _figure_violin(
    windows_df:  pd.DataFrame,
    labels:      np.ndarray,
    roc_results: dict,
    save_path:   str,
) -> None:
    """
    Violin plots: ACCEPT vs REJECT distributions for each SQI metric.

    Draws the optimal acceptance boundary as dashed lines:
      - One line for one-sided metrics.
      - Two symmetric lines (±thr) for abs_hi_bound metrics (skewness).
      - Two lines (lo, hi) for two-sided metrics (kurtosis).
    """
    metrics = [m for m in roc_results.keys() if m != 'composite_sqi']
    nrows = int(np.ceil(len(metrics) / 3))
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), dpi=300)
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    accept_mask = labels == 1
    reject_mask = labels == 0

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        if metric not in windows_df.columns:
            ax.set_visible(False)
            continue
        vals        = windows_df[metric].to_numpy(dtype=float)
        data_accept = vals[accept_mask & ~np.isnan(vals)]
        data_reject = vals[reject_mask & ~np.isnan(vals)]

        parts = ax.violinplot([data_accept, data_reject], positions=[0, 1],
                              showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        parts['bodies'][0].set_facecolor('steelblue')
        parts['bodies'][1].set_facecolor('crimson')

        res = roc_results[metric]
        if res['two_sided']:
            ax.axhline(res['youden_thr_lo'], color='black', lw=1.2, ls='--',
                       label=f"lo = {res['youden_thr_lo']:.3g}")
            ax.axhline(res['youden_thr_hi'], color='dimgrey', lw=1.2, ls='--',
                       label=f"hi = {res['youden_thr_hi']:.3g}")
        elif res['abs_transformed']:
            # Threshold is on |value| → show symmetric ±thr lines in original space.
            thr = res['youden_thr']
            ax.axhline( thr, color='black',   lw=1.2, ls='--', label=f'+{thr:.3g}')
            ax.axhline(-thr, color='dimgrey', lw=1.2, ls='--', label=f'−{thr:.3g}')
        else:
            opt = res['youden_thr']
            ax.axhline(opt, color='black', lw=1.2, ls='--', label=f'opt = {opt:.3g}')

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
    cmap    = matplotlib.colormaps['tab10']
    metrics = list(roc_results.keys())

    for i, metric in enumerate(metrics):
        res      = roc_results[metric]
        sort_idx = np.argsort(res['fpr'])
        lw  = 2.5 if metric == 'composite_sqi' else 1.0
        col = 'black' if metric == 'composite_sqi' else cmap(i % 10)
        ax.plot(res['fpr'][sort_idx], res['sens'][sort_idx], lw=lw, color=col,
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
# RMSE-optimal threshold search
# ---------------------------------------------------------------------------

def find_rmse_optimal_threshold(
    windows_list:   list,
    random_seed:    int   = 42,
    n_sweep:        int   = 20,
    save_dir:       str   = 'results/sqi_roc',
    include_combos        = None,
    use_ac_dc:      bool  = False,
    svr_C:          float = 100.0,
    svr_epsilon:    float = 0.1,
    svr_gamma               = 'scale',
    gpr_alpha:      float = 1e-6,
    gpr_n_restarts: int   = 5,
) -> dict:
    """
    Find the composite SQI threshold that minimises mean LOSO RMSE directly.

    Unlike Youden-based optimisation (which maximises classification accuracy
    for detecting high-ARMS windows), this sweeps the threshold over the full
    calibration pipeline and picks the value that produces the lowest
    cross-validated RMSE.

    Parameters
    ----------
    windows_list : list of PPGWindow objects (the full pre-SQI window set).
    n_sweep      : number of evenly spaced percentile candidates to evaluate.

    Returns
    -------
    dict with keys: optimal_threshold, optimal_rmse, sweep_results
    """
    from data_extraction import composite_sqi_score, fit_composite_sqi_scaler, \
                                 apply_composite_sqi_fitted
    from calibration import calibrate_all_combos

    if not windows_list:
        return {}

    scaler = fit_composite_sqi_scaler(windows_list)
    lo, hi = scaler

    raw_scores = np.array([composite_sqi_score(w) for w in windows_list])
    candidates_raw = np.percentile(raw_scores, np.linspace(5, 80, n_sweep))
    candidates_raw = np.unique(candidates_raw)

    sweep_results = []
    print(f"\n[RMSE sweep]  {'threshold':>10}  {'n_kept':>7}  {'best_RMSE':>10}")
    for raw_thr in candidates_raw:
        thr_norm = float((raw_thr - lo) / (hi - lo)) if hi > lo else 0.0
        kept = apply_composite_sqi_fitted(windows_list, thr_norm, scaler)
        if len(kept) < 10:
            continue
        cv = calibrate_all_combos(
            kept, random_seed=random_seed,
            include_combos=include_combos, use_ac_dc=use_ac_dc,
            svr_C=svr_C, svr_epsilon=svr_epsilon, svr_gamma=svr_gamma,
            gpr_alpha=gpr_alpha, gpr_n_restarts=gpr_n_restarts,
        )
        # Best mean RMSE across models, evaluated on first (or only) combo
        first_combo = next(iter(cv.values())) if cv else {}
        if not first_combo:
            continue
        best_rmse = min(
            sum(f.rmse for f in folds) / len(folds)
            for folds in first_combo.values() if folds
        )
        sweep_results.append({
            'threshold': thr_norm,
            'n_kept':    len(kept),
            'rmse':      best_rmse,
        })
        print(f"[RMSE sweep]  {thr_norm:>10.4f}  {len(kept):>7}  {best_rmse:>10.4f}")

    if not sweep_results:
        return {}

    best = min(sweep_results, key=lambda r: r['rmse'])
    print(f"[RMSE sweep]  optimal threshold = {best['threshold']:.4f}  "
          f"RMSE = {best['rmse']:.4f}  n_kept = {best['n_kept']}")

    # Persist the RMSE-optimal threshold alongside the Youden thresholds.
    json_path = Path(save_dir) / 'optimal_thresholds.json'
    if json_path.exists():
        with open(json_path) as fh:
            opt_json = json.load(fh)
    else:
        opt_json = {}
    composite_entry = opt_json.setdefault('composite_sqi', {})
    composite_entry['rmse_optimal_threshold'] = best['threshold']
    composite_entry['rmse_optimal_n_kept']    = best['n_kept']
    composite_entry['rmse_optimal_value']     = best['rmse']
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as fh:
        json.dump(opt_json, fh, indent=2)
    print(f"[RMSE sweep]  saved to {json_path}")

    return {
        'optimal_threshold': best['threshold'],
        'optimal_rmse':      best['rmse'],
        'sweep_results':     sweep_results,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_roc_analysis(
    windows_df:            pd.DataFrame,
    arms_reject_threshold: float = ARMS_REJECT_THRESHOLD_DEFAULT,
    save_dir:              str   = 'results/sqi_roc',
) -> pd.DataFrame:
    """
    Run full ROC threshold analysis on SQI metrics.

    Parameters
    ----------
    windows_df : DataFrame with SQI metric columns plus 'spo2_ref' and
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

    labels   = _make_labels(windows_df, arms_reject_threshold)
    n_accept = int(labels.sum())
    n_reject = len(labels) - n_accept
    print(f"[ROC analysis]  ACCEPT = {n_accept}, REJECT = {n_reject}")

    windows_df = windows_df.copy()
    windows_df['composite_sqi'] = compute_composite_sqi(windows_df)

    analyse_metrics = METRICS_SINGLE + ['composite_sqi']
    roc_results: dict = {}

    for metric in analyse_metrics:
        if metric not in windows_df.columns:
            print(f"  [ROC] column '{metric}' not found — skipped")
            continue
        vals = windows_df[metric].to_numpy(dtype=float)
        mode = METRIC_MODE.get(metric, 'lo_bound')
        roc_results[metric] = _roc_for_metric(vals, labels, mode)
        res = roc_results[metric]

        if res['two_sided']:
            thr_str = f"[{res['youden_thr_lo']:.4g}, {res['youden_thr_hi']:.4g}]"
        elif res['abs_transformed']:
            thr_str = f"|val| ≤ {res['youden_thr']:.4g}"
        else:
            thr_str = f"{res['youden_thr']:.4g}"

        print(f"  {METRIC_LABELS.get(metric, metric):<28}  "
              f"AUC={res['auc']:.3f}  Youden_thr={thr_str}  "
              f"Sens={res['opt_sens']:.3f}  Spec={res['opt_spec']:.3f}")

    # --- Summary table --------------------------------------------------
    rows = []
    for metric, res in roc_results.items():
        rows.append({
            'metric':           metric,
            'label':            METRIC_LABELS.get(metric, metric),
            'mode':             METRIC_MODE.get(metric, 'lo_bound'),
            'auc':              round(res['auc'], 4),
            'youden_threshold': round(res['youden_thr'], 4),
            'youden_thr_lo':    round(res['youden_thr_lo'], 4) if res['youden_thr_lo'] is not None else None,
            'youden_thr_hi':    round(res['youden_thr_hi'], 4) if res['youden_thr_hi'] is not None else None,
            'youden_index':     round(res['youden_idx'], 4),
            'sensitivity':      round(res['opt_sens'], 4),
            'specificity':      round(res['opt_spec'], 4),
            'ppv':              round(res['opt_ppv'], 4),
            'npv':              round(res['opt_npv'], 4),
            'f1':               round(res['opt_f1'], 4),
        })
    summary_df   = pd.DataFrame(rows)
    summary_path = out / 'roc_summary_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"[ROC analysis]  saved {summary_path}")

    # --- Sensitivity analysis -------------------------------------------
    sens_df  = _sensitivity_analysis(roc_results, windows_df, labels)
    sens_path = out / 'threshold_sensitivity_table.csv'
    sens_df.to_csv(sens_path, index=False)
    print(f"[ROC analysis]  saved {sens_path}")

    # --- Optimal thresholds JSON ----------------------------------------
    opt_json: dict = {}
    for m, res in roc_results.items():
        entry: dict = {
            'mode':        METRIC_MODE.get(m, 'lo_bound'),
            'auc':         res['auc'],
            'sensitivity': res['opt_sens'],
            'specificity': res['opt_spec'],
            'f1':          res['opt_f1'],
        }
        if res['two_sided']:
            entry['youden_threshold_lo'] = res['youden_thr_lo']
            entry['youden_threshold_hi'] = res['youden_thr_hi']
        elif res['abs_transformed']:
            entry['youden_threshold_abs'] = res['youden_thr']
        else:
            entry['youden_threshold'] = res['youden_thr']
        opt_json[m] = entry

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
    N   = 500

    spo2_ref  = np.asarray(rng.normal(96.0, 2.0, N), dtype=float).clip(88, 100)
    pred_err  = np.asarray(rng.normal(0.0, 2.5, N),  dtype=float)
    spo2_pred = spo2_ref + pred_err

    pi             = np.asarray(rng.uniform(0.001, 0.05, N), dtype=float)
    skewness_ir    = np.asarray(rng.normal(0.0, 1.0, N),     dtype=float)  # centred at 0
    kurtosis_ir    = np.asarray(rng.normal(2.0, 3.0, N),     dtype=float)
    snr_ir         = np.asarray(rng.normal(-5.0, 8.0, N),    dtype=float)
    entropy_ir     = np.asarray(rng.uniform(2.0, 5.0, N),    dtype=float)
    rel_power_ir   = np.asarray(rng.uniform(0.01, 0.5, N),   dtype=float)
    acc_energy_std = np.asarray(rng.exponential(50.0, N),    dtype=float)

    # Corrupt high-error windows: lower PI, higher motion, lower SNR,
    # and push skewness / kurtosis to extremes in both directions.
    high_err = np.abs(pred_err) > 3.5
    pi[high_err]              *= 0.3
    acc_energy_std[high_err]  *= 2.5
    snr_ir[high_err]          -= 6.0
    skewness_ir[high_err]     += rng.choice([-3.0, 3.0], size=int(high_err.sum()))
    kurtosis_ir[high_err]     += rng.choice([-4.0, 8.0], size=int(high_err.sum()))

    df = pd.DataFrame({
        'spo2_ref':       spo2_ref,
        'spo2_pred_mean': spo2_pred,
        'pi':             pi,
        'skewness_ir':    skewness_ir,
        'kurtosis_ir':    kurtosis_ir,
        'snr_ir':         snr_ir,
        'entropy_ir':     entropy_ir,
        'rel_power_ir':   rel_power_ir,
        'acc_energy_std': acc_energy_std,
    })

    summary = run_roc_analysis(df, arms_reject_threshold=3.5, save_dir='results/sqi_roc')
    print("\nROC Summary:")
    print(summary.to_string(index=False))
