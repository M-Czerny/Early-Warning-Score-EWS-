"""
Diagnostic plots for one pipeline run.

Call plot_all() from main.py.  A timestamped sub-folder is created inside
`plots/` so successive runs never overwrite each other.
"""
from __future__ import annotations

import datetime
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')                 # file-only backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as dsp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_run_dir(base: str = 'plots') -> str:
    path = os.path.join(base, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(path, exist_ok=True)
    return path


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = dsp.butter(4, [low / nyq, high / nyq], btype='band', output='ba')  # type: ignore
    return dsp.filtfilt(b, a, x)


def _bad_time_spans(mask: np.ndarray, fs: float) -> list[tuple[float, float]]:
    """(t_start, t_end) pairs in seconds for each contiguous True run in mask."""
    padded = np.concatenate([[False], mask, [False]])
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return [(int(s) / fs, int(e) / fs) for s, e in zip(starts, ends)]


def _shade(axes, spans: list[tuple[float, float]], color: str, alpha: float) -> None:
    for ax in axes:
        for t0, t1 in spans:
            ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0)


def _best_model_name(results: dict) -> str:
    return min(results, key=lambda m: np.mean([f.rmse for f in results[m]]))


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(path)}')


# ---------------------------------------------------------------------------
# Plot 1 — per-record signal overview
# ---------------------------------------------------------------------------

def _plot_signals(
    records:     list,
    all_windows: list,
    kept_ids:    set,
    out_dir:     str,
    window_s:    float,
    bp_low:      float,
    bp_high:     float,
) -> None:
    for rec in records:
        subj = rec['subject']
        ep   = rec['episode']
        fs   = rec['ppg_fs']
        orig = rec['ppg_original']
        bad  = rec['ppg_bad_mask']
        ci   = rec['ppg_clean_indices']   # original index of each clean PPG sample
        win_n = int(window_s * fs)

        n = len(bad)
        t = np.arange(n) / fs

        # Bandpass each channel from the pre-removal signal
        def ac(key: str) -> np.ndarray:
            x = orig[key].astype(float)
            return _bandpass(x, fs, bp_low, bp_high) if len(x) > 3 * win_n else np.zeros(n)

        ac_r = ac('red');   ac_i = ac('ir')
        ac_g = ac('green'); ac_b = ac('blue')
        acc_e = np.sqrt(orig['acc_x'] ** 2.0 + orig['acc_y'] ** 2.0 + orig['acc_z'] ** 2.0)

        # Windows belonging to this record
        rec_wins = [w for w in all_windows if w.subject == subj and w.episode == ep]

        # Map each window's clean-time start → original time span
        def orig_span(w) -> tuple[float, float]:
            cs = min(int(round(w.t_start_s * fs)), len(ci) - 1)
            ce = min(cs + win_n - 1, len(ci) - 1)
            return ci[cs] / fs, (ci[ce] + 1) / fs

        r_kept    = [(orig_span(w), w.R)        for w in rec_wins if id(w) in  kept_ids and not np.isnan(w.R)]
        r_rej     = [(orig_span(w), w.R)        for w in rec_wins if id(w) not in kept_ids and not np.isnan(w.R)]
        spo2_kept = [(orig_span(w), w.spo2_ref) for w in rec_wins if id(w) in  kept_ids and not np.isnan(w.spo2_ref)]
        spo2_rej  = [(orig_span(w), w.spo2_ref) for w in rec_wins if id(w) not in kept_ids and not np.isnan(w.spo2_ref)]

        art_spans = _bad_time_spans(bad, fs)
        sqi_spans = [orig_span(w) for w in rec_wins if id(w) not in kept_ids]

        fig, axes = plt.subplots(7, 1, figsize=(16, 14), sharex=True)
        title = subj + (f'  /  {ep}' if ep else '')
        fig.suptitle(title, fontsize=11, fontweight='bold')

        # Signal rows
        for ax, sig, col, lbl in zip(
            axes[:5],
            [ac_r,    ac_i,        ac_g,          ac_b,        acc_e],
            ['crimson','darkorange','forestgreen','royalblue','purple'],
            ['AC Red','AC IR','AC Green','AC Blue','Acc energy'],
        ):
            ax.plot(t, sig, color=col, lw=0.7)
            ax.set_ylabel(lbl, fontsize=8)

        # R row — horizontal bars per window
        for (t0, t1), val in r_kept:
            axes[5].hlines(val, t0, t1, colors='teal',  lw=2.5)
        for (t0, t1), val in r_rej:
            axes[5].hlines(val, t0, t1, colors='grey', lw=1.2, linestyle='--', alpha=0.55)
        axes[5].set_ylabel('R', fontsize=8)

        # SpO2 row
        for (t0, t1), val in spo2_kept:
            axes[6].hlines(val, t0, t1, colors='navy', lw=2.5)
        for (t0, t1), val in spo2_rej:
            axes[6].hlines(val, t0, t1, colors='grey', lw=1.2, linestyle='--', alpha=0.55)
        axes[6].set_ylabel('SpO2 ref (%)', fontsize=8)
        axes[6].set_xlabel('Time (s)', fontsize=9)

        # Shading
        _shade(axes, art_spans, 'red',    alpha=0.20)
        _shade(axes, sqi_spans, 'orange', alpha=0.15)

        for ax in axes:
            ax.tick_params(labelsize=7)

        legend_handles = [
            mpatches.Patch(color='red',    alpha=0.40, label='Artifact removed'),
            mpatches.Patch(color='orange', alpha=0.40, label='SQI rejected window'),
            plt.Line2D([0], [0], color='teal', lw=2,          label='Kept window'),
            plt.Line2D([0], [0], color='grey', lw=1, ls='--', label='Rejected window'),
        ]
        fig.legend(handles=legend_handles, loc='upper right', fontsize=7, ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        slug = subj + (f'_ep{ep.split()[-1]}' if ep else '')
        _save(fig, os.path.join(out_dir, f'01_signals_{slug}.png'))


# ---------------------------------------------------------------------------
# Plot 2 — per-subject SpO2 predictions
# ---------------------------------------------------------------------------

def _plot_predictions(results: dict, out_dir: str) -> None:
    model_names = list(results.keys())
    palette     = plt.cm.tab10.colors
    subjects    = sorted({f.test_subject for folds in results.values() for f in folds})

    for subj in subjects:
        folds = {m: next((f for f in results[m] if f.test_subject == subj), None)
                 for m in model_names}
        if all(v is None for v in folds.values()):
            continue

        y_true = next(f.y_true for f in folds.values() if f is not None)
        x      = np.arange(len(y_true))

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(x, y_true - 3, y_true + 3,
                        alpha=0.15, color='steelblue', label='±3 % band')
        ax.plot(x, y_true, color='steelblue', lw=2.0, label='Reference SpO2')

        for i, name in enumerate(model_names):
            fold = folds[name]
            if fold is None:
                continue
            ax.plot(x, fold.y_pred,
                    color=palette[(i + 1) % len(palette)], lw=1.5,
                    label=f'{name}  RMSE={fold.rmse:.2f}')

        ax.set_xlabel('Window index', fontsize=9)
        ax.set_ylabel('SpO2 (%)', fontsize=9)
        ax.set_title(f'{subj} — predicted vs reference SpO2', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, f'02_predictions_{subj}.png'))


# ---------------------------------------------------------------------------
# Plot 3 — R vs SpO2 scatter for the best model
# ---------------------------------------------------------------------------

def _plot_scatter(results: dict, kept_windows: list, out_dir: str) -> None:
    best    = _best_model_name(results)
    folds   = results[best]
    palette = plt.cm.tab10.colors

    # Reconstruct per-subject R arrays in the same order used by calibration
    valid_by_subj = {
        subj: [w for w in kept_windows
               if w.subject == subj and not np.isnan(w.R) and not np.isnan(w.spo2_ref)]
        for subj in sorted({f.test_subject for f in folds})
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, fold in enumerate(folds):
        ws = valid_by_subj.get(fold.test_subject, [])
        if len(ws) != len(fold.y_true):
            continue                   # length mismatch — skip this fold
        r_vals = np.array([w.R for w in ws])

        # Clip extreme R outliers for readability
        r_clip = np.clip(r_vals, 0, 5)

        col = palette[i % len(palette)]
        ax.scatter(r_clip, fold.y_true, s=22, color=col, alpha=0.35, marker='o',
                   label=f'{fold.test_subject} true')
        ax.scatter(r_clip, fold.y_pred, s=22, color=col, alpha=0.85, marker='^',
                   label=f'{fold.test_subject} pred')

    ax.set_xlabel('R  (ratio of ratios, clipped to [0, 5])', fontsize=10)
    ax.set_ylabel('SpO2 (%)', fontsize=10)
    ax.set_title(f'R vs SpO2  —  best model: {best}', fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, '03_scatter_best_model.png'))


# ---------------------------------------------------------------------------
# Plot 4 — mean LOSO RMSE per model (with optional raw-baseline comparison)
# ---------------------------------------------------------------------------

def _annotate_bars(ax, bars, means, stds, fontsize=8):
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.04,
            f'{m:.2f}', ha='center', va='bottom', fontsize=fontsize,
        )


def _plot_rmse(
    results:      dict,
    out_dir:      str,
    results_raw:  Optional[dict] = None,
    title_suffix: str = '',
    filename:     str = '04_rmse_comparison.png',
) -> None:
    names = list(results.keys())
    best  = _best_model_name(results)

    means_f = [np.mean([f.rmse for f in results[m]]) for m in names]
    stds_f  = [np.std( [f.rmse for f in results[m]]) for m in names]

    fig, ax = plt.subplots(figsize=(9, 4))

    if results_raw is None:
        colors = [
            'gold' if n == best else plt.cm.Set2.colors[i % 8]
            for i, n in enumerate(names)
        ]
        bars = ax.bar(names, means_f, yerr=stds_f, capsize=5,
                      color=colors, edgecolor='black', lw=0.8)
        _annotate_bars(ax, bars, means_f, stds_f)
        ax.set_title(
            f'Leave-One-Subject-Out RMSE  (mean ± std){title_suffix}\n'
            f'Best model: {best}', fontsize=10,
        )
        y_top = max(m + s for m, s in zip(means_f, stds_f)) * 1.35
    else:
        means_r = [np.mean([f.rmse for f in results_raw[m]]) for m in names]
        stds_r  = [np.std( [f.rmse for f in results_raw[m]]) for m in names]

        x     = np.arange(len(names))
        width = 0.38

        bars_r = ax.bar(
            x - width / 2, means_r, width, yerr=stds_r, capsize=4,
            color='lightgrey', edgecolor='black', lw=0.8,
            label='No preprocessing (raw)',
        )
        filt_colors = [
            'gold' if n == best else plt.cm.Set2.colors[i % 8]
            for i, n in enumerate(names)
        ]
        bars_f = ax.bar(
            x + width / 2, means_f, width, yerr=stds_f, capsize=4,
            color=filt_colors, edgecolor='black', lw=0.8,
            label='Artifact removal + SQI',
        )
        _annotate_bars(ax, bars_r, means_r, stds_r)
        _annotate_bars(ax, bars_f, means_f, stds_f)

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend(fontsize=9)
        ax.set_title(
            f'Leave-One-Subject-Out RMSE  (mean ± std)  —  raw vs preprocessed'
            f'{title_suffix}\nBest preprocessed model: {best}', fontsize=10,
        )
        y_top = max(
            max(m + s for m, s in zip(means_r, stds_r)),
            max(m + s for m, s in zip(means_f, stds_f)),
        ) * 1.35

    ax.set_ylabel('Mean RMSE (SpO2 %)', fontsize=10)
    ax.set_ylim(0, y_top)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, filename))


def _plot_rmse_combos(
    combo_results:     dict,
    out_dir:           str,
    combo_results_raw: Optional[dict] = None,
) -> None:
    """One RMSE comparison plot per LED combination."""
    for combo, results in combo_results.items():
        raw = combo_results_raw.get(combo) if combo_results_raw else None
        safe = combo.replace(' ', '_')
        _plot_rmse(
            results, out_dir,
            results_raw=raw,
            title_suffix=f'  [{combo}]',
            filename=f'06_rmse_{safe}.png',
        )


# ---------------------------------------------------------------------------
# Plot 5 — SQI parameter overview
# ---------------------------------------------------------------------------

def _plot_sqi(
    all_windows: list,
    kept_ids:    set,
    sqi_params:  dict,
    out_dir:     str,
) -> None:
    """
    One subplot per SQI metric.  Every window is plotted (x = window index,
    y = parameter value) in its channel colour: red=crimson, IR=darkorange,
    green=forestgreen, blue=royalblue.  No kept/rejected distinction is shown.
    Threshold lines are drawn as dashed (min) or dotted (max) black lines.
    Each subplot title reports total window count and how many windows fail
    that criterion alone (across all channels in that metric).
    """
    n = len(all_windows)
    x = np.arange(n)
    p = sqi_params

    # channel colour map: label → colour
    CH_COL = {
        'Red':   'crimson',
        'IR':    'darkorange',
        'Green': 'forestgreen',
        'Blue':  'royalblue',
        'PI':    'steelblue',
        'Acc':   'slategrey',
    }

    def _v(field: str) -> np.ndarray:
        return np.array([getattr(w, field) for w in all_windows])

    def _fails(lo, hi, *fields) -> int:
        bad = np.zeros(n, dtype=bool)
        for f in fields:
            v = _v(f)
            if lo is not None: bad |= v < lo
            if hi is not None: bad |= v > hi
        return int(bad.sum())

    specs = [
        dict(title='Perfusion Index (PI)',        ylabel='PI',
             lo=p.get('pi_min'),         hi=p.get('pi_max'),
             channels=[('pi',                'PI',    'o')]),
        dict(title='Skewness',                    ylabel='skewness',
             lo=p.get('skewness_min'),   hi=p.get('skewness_max'),
             channels=[('skewness_red',    'Red',   'o'),
                        ('skewness_ir',     'IR',    '^'),
                        ('skewness_green',  'Green', 's'),
                        ('skewness_blue',   'Blue',  'D')]),
        dict(title='Kurtosis',                    ylabel='kurtosis',
             lo=p.get('kurtosis_min'),   hi=p.get('kurtosis_max'),
             channels=[('kurtosis_red',    'Red',   'o'),
                        ('kurtosis_ir',     'IR',    '^'),
                        ('kurtosis_green',  'Green', 's'),
                        ('kurtosis_blue',   'Blue',  'D')]),
        dict(title='SNR (dB)',                    ylabel='SNR (dB)',
             lo=p.get('snr_min'),        hi=p.get('snr_max'),
             channels=[('snr_red',         'Red',   'o'),
                        ('snr_ir',          'IR',    '^'),
                        ('snr_green',       'Green', 's'),
                        ('snr_blue',        'Blue',  'D')]),
        dict(title='Spectral Entropy (bits)',     ylabel='entropy',
             lo=p.get('entropy_min'),    hi=p.get('entropy_max'),
             channels=[('entropy_red',     'Red',   'o'),
                        ('entropy_ir',      'IR',    '^'),
                        ('entropy_green',   'Green', 's'),
                        ('entropy_blue',    'Blue',  'D')]),
        dict(title='Relative Power',              ylabel='rel. power',
             lo=p.get('rel_power_min'),  hi=p.get('rel_power_max'),
             channels=[('rel_power_red',   'Red',   'o'),
                        ('rel_power_ir',    'IR',    '^'),
                        ('rel_power_green', 'Green', 's'),
                        ('rel_power_blue',  'Blue',  'D')]),
        dict(title='Acceleration Energy Std-Dev', ylabel='acc std',
             lo=None,                   hi=p.get('acc_std_max'),
             channels=[('acc_energy_std',  'Acc',   'o')]),
    ]

    fig, axes = plt.subplots(len(specs), 1,
                              figsize=(14, 3 * len(specs)), sharex=True)
    fig.suptitle('SQI parameter overview  —  all windows',
                 fontsize=11, fontweight='bold')

    for ax, spec in zip(axes, specs):
        lo, hi = spec['lo'], spec['hi']
        chs    = spec['channels']

        n_fails = _fails(lo, hi, *[c[0] for c in chs])
        pct     = 100.0 * n_fails / n if n > 0 else 0.0

        for field, label, marker in chs:
            v   = _v(field)
            col = CH_COL.get(label, 'grey')
            ax.scatter(x, v, s=7, marker=marker, c=col, alpha=0.5, lw=0,
                       label=label)

        if lo is not None:
            ax.axhline(lo, color='black', ls='--', lw=1.0, label=f'min = {lo}')
        if hi is not None:
            ax.axhline(hi, color='black', ls=':',  lw=1.0, label=f'max = {hi}')

        ax.set_title(
            f"{spec['title']}   ·   "
            f"total {n}  |  fails this criterion: {n_fails} ({pct:.1f}%)",
            fontsize=8.5, loc='left',
        )
        ax.set_ylabel(spec['ylabel'], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc='upper right', ncol=len(chs))

    axes[-1].set_xlabel('Window index', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, os.path.join(out_dir, '05_sqi_overview.png'))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_all(
    records:           list,
    all_windows:       list,
    kept_windows:      list,
    results:           dict,
    window_s:          float,
    bp_low:            float,
    bp_high:           float,
    results_raw:       Optional[dict] = None,
    sqi_params:        Optional[dict] = None,
    combo_results:     Optional[dict] = None,
    combo_results_raw: Optional[dict] = None,
    base_dir:          str = 'plots',
) -> str:
    """
    Generate all diagnostic plots for one pipeline run.

    Parameters
    ----------
    records            : preprocessed records (ppg_original, ppg_bad_mask, ppg_clean_indices)
    all_windows        : every PPGWindow before SQI filtering
    kept_windows       : PPGWindows that passed the SQI filter
    results            : Red-IR calibration results (for predictions / scatter / RMSE plots)
    window_s           : window length in seconds
    bp_low/high        : bandpass cut-offs used during feature extraction
    results_raw        : Red-IR results on unfiltered windows for RMSE baseline
    sqi_params         : SQI threshold dict — when supplied, plot 05 is generated
    combo_results      : nested dict combo→model→folds from calibrate_all_combos()
    combo_results_raw  : same but on unfiltered windows; adds raw baseline bars per combo
    base_dir           : parent directory; a timestamped sub-folder is created per run

    Returns
    -------
    Path to the created output directory.
    """
    out_dir  = _make_run_dir(base_dir)
    kept_ids = {id(w) for w in kept_windows}

    print('\nPlotting signals...')
    _plot_signals(records, all_windows, kept_ids, out_dir, window_s, bp_low, bp_high)

    print('Plotting predictions...')
    _plot_predictions(results, out_dir)

    print('Plotting R vs SpO2 scatter...')
    _plot_scatter(results, kept_windows, out_dir)

    print('Plotting RMSE comparison (Red-IR)...')
    _plot_rmse(results, out_dir, results_raw=results_raw)

    if sqi_params is not None:
        print('Plotting SQI overview...')
        _plot_sqi(all_windows, kept_ids, sqi_params, out_dir)

    if combo_results is not None:
        print('Plotting LED combination RMSE plots...')
        _plot_rmse_combos(combo_results, out_dir, combo_results_raw)

    print(f'\nAll plots saved to:  {out_dir}')
    return out_dir
