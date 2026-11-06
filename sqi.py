"""
Signal Quality Index (SQI) filter: decide which windows are clean enough to use.

Each window carries seven quality metrics (computed in data_extraction.py).
A window is dropped when any metric is on the "poor quality" side of its limit.
There are four ways to set those limits, chosen in main.py:

  baseline    (default) Each recording's first minute is a calibration phase.
              A window is dropped when a metric moves too far from that
              person's own calibration values.  See apply_baseline_sqi().
  fixed       Use fixed limits taken from the literature.  See apply_sqi().
  percentile  Drop the worst X % of windows on each metric.
              See percentile_thresholds().
  composite   Combine all metrics into one weighted score and drop windows
              below a cut-off.  See apply_composite_sqi().

Literature behind the metrics and their directions:
  Elgendi M. (2016) Optimal signal quality index for photoplethysmogram
      signals. Bioengineering 3(4):21.               (perfusion, skewness, rel. power)
  Selvaraj N. et al. (2011) Statistical approach for the detection of
      motion/noise artifacts in photoplethysmogram. IEEE EMBC.  (kurtosis, entropy)
  de Haan G., Jeanne V. (2013) Robust pulse rate from chrominance-based rPPG.
      IEEE Trans. Biomed. Eng. 60(10).               (SNR)
  Leys C. et al. (2013) Detecting outliers: use absolute deviation around the
      median. J. Exp. Soc. Psychol. 49(4):764-766.   (median/MAD baseline)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

CHANNELS = ('red', 'ir', 'green', 'blue')

# Which side of each metric means poor quality:
#   'lo_bound'  -> too LOW is bad    'hi_bound' -> too HIGH is bad
#   'two_sided' -> both sides are bad
# This is the one place the directions are defined.
METRICS: dict[str, str] = {
    'pi':        'two_sided',  # perfusion: too low = weak pulse; too high = motion on this prototype
    'skewness':  'lo_bound',   # Elgendi: clean pulses have higher skewness
    'kurtosis':  'hi_bound',   # Selvaraj: artefacts raise kurtosis
    'snr':       'lo_bound',   # de Haan: less power at the pulse rate
    'entropy':   'lo_bound',   # Selvaraj: artefacts lower the entropy
    'rel_power': 'lo_bound',   # Elgendi: less power in the cardiac band
    'acc_std':   'hi_bound',   # accelerometer: more motion
}


def metric_fields(metric: str, ir_only: bool = False) -> list[str]:
    """
    PPGWindow attributes that hold `metric`.

    Perfusion and motion have one value per window.  The others have one value
    per LED; with ir_only=True only the IR channel is checked.
    """
    if metric == 'pi':
        return ['pi']
    if metric == 'acc_std':
        return ['acc_energy_std']
    if ir_only:
        return [f'{metric}_ir']
    return [f'{metric}_{ch}' for ch in CHANNELS]


# ---------------------------------------------------------------------------
# Fixed limits
# ---------------------------------------------------------------------------

def _is_out_of_limits(w, limits: dict, ir_only: bool) -> bool:
    """True if any channel of any metric is outside its limit.  NaN values pass."""
    for metric in METRICS:
        lo = limits.get(f'{metric}_min')
        hi = limits.get(f'{metric}_max')
        for f in metric_fields(metric, ir_only):
            v = getattr(w, f)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                return True
    return False


def apply_sqi(windows: list, limits: dict, ir_only: bool = False,
              verbose: bool = True) -> list:
    """
    Keep windows where every metric is within `limits`.

    `limits` has keys like 'snr_min' or 'kurtosis_max'.  A missing key or a
    value of None means "no limit on that side".
    """
    allowed = {f'{m}_{side}' for m in METRICS for side in ('min', 'max')}
    unknown = set(limits) - allowed
    if unknown:
        raise ValueError(f"unknown SQI limit(s): {sorted(unknown)}")

    kept = [w for w in windows if not _is_out_of_limits(w, limits, ir_only)]
    if verbose:
        n = len(windows)
        print(f"  SQI{' (IR only)' if ir_only else ''}: kept {len(kept)} / {n} windows "
              f"({100 * (n - len(kept)) / max(n, 1):.1f}% rejected)")
    return kept


# ---------------------------------------------------------------------------
# Percentile limits
# ---------------------------------------------------------------------------

def percentile_thresholds(windows: list, pct: float, ir_only: bool = False) -> dict:
    """
    Limits that drop the worst `pct` % of windows on each metric.

    A window fails when ANY of its channels fails, so the percentile is taken
    over each window's worst channel.  Two-sided metrics lose pct/2 % per side.
    Returns a dict that can be passed straight to apply_sqi().

    The overall rejection is larger than pct %, because a window can fail on
    different metrics.  report_percentile_overlap() prints the real total.
    """
    if not 0.0 <= pct < 50.0:
        raise ValueError(f"pct must be in [0, 50), got {pct}")

    limits: dict = {}
    for metric, side in METRICS.items():
        vals = np.array([[getattr(w, f) for f in metric_fields(metric, ir_only)]
                         for w in windows], dtype=float)
        vals = vals[np.isfinite(vals).any(axis=1)]    # skip windows with no value
        lo = hi = None
        if len(vals):
            lowest, highest = np.nanmin(vals, axis=1), np.nanmax(vals, axis=1)
            if side == 'lo_bound':
                lo = float(np.percentile(lowest, pct))
            elif side == 'hi_bound':
                hi = float(np.percentile(highest, 100 - pct))
            else:
                lo = float(np.percentile(lowest, pct / 2))
                hi = float(np.percentile(highest, 100 - pct / 2))
        limits[f'{metric}_min'], limits[f'{metric}_max'] = lo, hi
    return limits


def report_percentile_overlap(windows: list, limits: dict,
                              ir_only: bool = False) -> tuple[int, int]:
    """
    Print how many windows each metric rejects on its own, and in total.

    Returns (number of rejected windows, number of windows).
    """
    union = np.zeros(len(windows), dtype=bool)
    print(f"  {'metric':<12}{'rejects alone':>15}")
    for metric in METRICS:
        single = {k: v for k, v in limits.items() if k.startswith(metric + '_')}
        bad = np.array([_is_out_of_limits(w, single, ir_only) for w in windows])
        union |= bad
        print(f"  {metric:<12}{100 * bad.mean():>14.1f}%")
    print(f"  {'TOTAL':<12}{100 * union.mean():>14.1f}%  ({int(union.sum())} windows)")
    return int(union.sum()), len(windows)


# ---------------------------------------------------------------------------
# Personal baseline (default)
# ---------------------------------------------------------------------------
# Why per person: the metrics depend on skin, tissue and sensor placement as
# much as on signal quality.  On this prototype even clean pulses differ in
# shape between subjects, so no single limit fits everyone.
#
# For each recording, metric and channel, the windows that lie entirely inside
# the first `baseline_s` seconds give:
#     centre = median(values)
#     spread = 1.4826 * MAD(values)     (a robust standard deviation)
# A window is rejected when a value is more than k * spread from the centre,
# on the poor-quality side only.  Median and MAD are used instead of mean and
# SD because a few artefacts inside the calibration minute barely move them.

MAD_SCALE            = 1.4826   # turns the MAD into an SD for normal data
MIN_BASELINE_WINDOWS = 5        # fewer baseline values -> that metric is not checked


def _baseline_fields(ir_only: bool) -> dict[str, tuple[str, str]]:
    """{window attribute: (metric, side)} for every value the baseline gate checks."""
    return {f: (m, side) for m, side in METRICS.items()
            for f in metric_fields(m, ir_only)}


def fit_baselines(windows: list, baseline_s: float, window_s: float, fields,
                  independent: bool = False) -> dict:
    """
    Calibration values per recording:
    {(subject, episode): {field: (median, spread, number of values)}}.

    Uses the windows that end within the first baseline_s seconds.  With
    independent=True only non-overlapping windows are used.
    """
    pools: dict = defaultdict(lambda: defaultdict(list))
    last_start: dict = {}
    for w in windows:                     # windows arrive in time order
        rec = (w.subject, w.episode)
        if independent:
            if rec in last_start and w.t_start_s < last_start[rec] + window_s - 1e-9:
                continue
            last_start[rec] = w.t_start_s
        if w.t_start_s + window_s <= baseline_s + 1e-9:
            for f in fields:
                pools[rec][f].append(getattr(w, f))

    out: dict = {}
    for rec, per_field in pools.items():
        out[rec] = {}
        for f, vals in per_field.items():
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                out[rec][f] = (np.nan, np.nan, 0)
                continue
            med = float(np.median(v))
            out[rec][f] = (med, MAD_SCALE * float(np.median(np.abs(v - med))), len(v))
    return out


def apply_baseline_sqi(windows: list, baseline_s: float = 60.0, window_s: float = 10.0,
                       k: float = 3.0, ir_only: bool = False, verbose: bool = True,
                       independent: bool = False) -> list:
    """
    Keep windows whose metrics stay within k robust SDs of their own
    recording's calibration values, on the poor-quality side only.

    NaN values pass.  A metric with too few calibration values is not checked
    for that recording, and a warning is printed.
    """
    fields = _baseline_fields(ir_only)
    base   = fit_baselines(windows, baseline_s, window_s, fields, independent)

    kept, fails = [], defaultdict(int)
    per_rec = defaultdict(lambda: [0, 0])     # recording -> [kept, total]
    warned  = set()
    for w in windows:
        rec, ok = (w.subject, w.episode), True
        for f, (metric, side) in fields.items():
            med, spread, n = base.get(rec, {}).get(f, (np.nan, np.nan, 0))
            if n < MIN_BASELINE_WINDOWS:
                if verbose and (rec, f) not in warned:
                    print(f"  [baseline] {rec[0]} {rec[1] or ''}: only {n} baseline "
                          f"values for {f}, not checked")
                warned.add((rec, f))
                continue
            v = getattr(w, f)
            if not np.isfinite(v):
                continue
            spread = max(spread, 1e-9 * max(1.0, abs(med)))   # avoid a zero spread
            too_low  = side in ('lo_bound', 'two_sided') and v < med - k * spread
            too_high = side in ('hi_bound', 'two_sided') and v > med + k * spread
            if too_low or too_high:
                fails[metric] += 1
                ok = False
        per_rec[rec][1] += 1
        if ok:
            kept.append(w)
            per_rec[rec][0] += 1

    if verbose:
        n, nk = len(windows), len(kept)
        print(f"  Baseline SQI (first {baseline_s:g} s, k={k:g}"
              f"{', IR only' if ir_only else ''}"
              f"{', non-overlapping baseline' if independent else ''}): "
              f"kept {nk} / {n} windows ({100 * (n - nk) / max(n, 1):.1f} % rejected)")
        for rec, (a, b) in per_rec.items():
            print(f"    {rec[0]:<12}{rec[1] or '':<12} kept {a:>4} / {b:<4}")
        print("    windows failing each metric (a window can fail several): " +
              ", ".join(f"{m} {c}" for m, c in sorted(fails.items(), key=lambda x: -x[1])))
    return kept


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
# One weighted score from the IR-channel metrics.  Higher = better quality.
# The weights are hand-picked; this mode is kept for comparison only.

COMPOSITE_WEIGHTS = (
    # (attribute, weight, lower is better?)
    ('pi',           0.30, False),
    ('skewness_ir',  0.25, False),
    ('snr_ir',       0.20, False),
    ('rel_power_ir', 0.10, False),
    ('kurtosis_ir',  0.10, True),
    ('entropy_ir',   0.05, False),
)


def composite_sqi_score(w) -> float:
    """
    Raw (not yet normalised) composite score of one window.  NaN metrics are skipped.

    To compare windows, normalise with the (lo, hi) range from
    fit_composite_sqi_scaler().
    """
    score = total = 0.0
    for name, weight, lower_is_better in COMPOSITE_WEIGHTS:
        val = getattr(w, name)
        if np.isnan(val):
            continue
        score += weight * (-val if lower_is_better else val)
        total += weight
    return score / total if total > 0 else 0.0


def fit_composite_sqi_scaler(windows: list) -> tuple[float, float]:
    """(lo, hi) range of the raw composite score, used to scale it to [0, 1]."""
    scores = np.array([composite_sqi_score(w) for w in windows])
    lo, hi = float(scores.min()), float(scores.max())
    return (0.0, 1.0) if hi == lo else (lo, hi)


def normalised_composite(windows: list, scaler: tuple[float, float]) -> np.ndarray:
    """Composite score of each window, scaled to [0, 1] with `scaler`."""
    lo, hi = scaler
    scores = np.array([composite_sqi_score(w) for w in windows])
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def composite_percentile_threshold(windows: list, pct: float,
                                   scaler: tuple[float, float]) -> float:
    """Composite cut-off that drops exactly the worst `pct` % of windows."""
    if not 0.0 <= pct < 50.0:
        raise ValueError(f"pct must be in [0, 50), got {pct}")
    return float(np.percentile(normalised_composite(windows, scaler), pct))


def apply_composite_sqi(windows: list, composite_min: float,
                        scaler: tuple[float, float]) -> list:
    """Keep windows whose normalised composite score is at least composite_min."""
    scores = normalised_composite(windows, scaler)
    kept = [w for w, s in zip(windows, scores) if s >= composite_min]
    n = len(windows)
    print(f"  Composite SQI (min={composite_min:.3f}): kept {len(kept)} / {n} "
          f"({100 * (n - len(kept)) / max(n, 1):.1f}% rejected)")
    return kept


# ---------------------------------------------------------------------------
# Independent check
# ---------------------------------------------------------------------------

def report_hr_agreement(all_windows: list, kept_windows: list,
                        tol_bpm: float = 5.0) -> None:
    """
    Check the filter against the reference oximeter, which the filter never sees.

    Prints the share of kept and of rejected windows whose prototype heart rate
    matches the reference heart rate (within tol_bpm).  A useful filter keeps
    clearly more matching windows than it rejects.
    """
    kept_ids = {id(w) for w in kept_windows}

    def share(ws):
        d = np.array([abs(w.hr_ppg - w.hr_ref) for w in ws], float)
        d = d[np.isfinite(d)]
        return (100 * float((d <= tol_bpm).mean()), len(d)) if len(d) else (np.nan, 0)

    k_pct, k_n = share([w for w in all_windows if id(w) in kept_ids])
    r_pct, r_n = share([w for w in all_windows if id(w) not in kept_ids])
    print(f"  HR agreement with reference (<= {tol_bpm:g} bpm): kept {k_pct:.1f} % "
          f"(n={k_n}), rejected {r_pct:.1f} % (n={r_n})")
