"""
Checks for the quality metrics (data_extraction.py) and the SQI filters (sqi.py).

Run:  python test_sqi.py
"""

import numpy as np

from data_extraction import (PPGWindow, perfusion_sqi, relative_power_sqi,
                             shannon_entropy_sqi, skewness_sqi, snr_hr_db)
from sqi import (METRICS, apply_baseline_sqi, apply_sqi, composite_percentile_threshold,
                 fit_baselines, fit_composite_sqi_scaler, normalised_composite,
                 percentile_thresholds, report_percentile_overlap)

FS = 50
T  = np.arange(0, 10, 1 / FS)


# --- quality metrics ---------------------------------------------------------

def _pulse_volume(hr_hz=1.2):
    """Blood-volume-like pulse: fast rise, slow fall."""
    phase = (T * hr_hz) % 1.0
    return np.where(phase < 0.15, phase / 0.15, np.exp(-(phase - 0.15) * 4))


def test_skewness_uses_blood_volume_orientation():
    raw_light = 1000 - 50 * _pulse_volume()     # detected light drops as blood volume rises
    assert skewness_sqi(raw_light - raw_light.mean()) > 0


def test_perfusion():
    raw = 1000 + 5 * np.sin(2 * np.pi * 1.2 * T)
    assert abs(perfusion_sqi(raw, raw - raw.mean()) - 1.0) < 0.01     # 10 / 1000 * 100


def test_shannon_entropy():
    clean = np.sin(2 * np.pi * 1.2 * T)
    spiky = np.zeros_like(T); spiky[::50] = 1.0
    assert 0.8 < shannon_entropy_sqi(clean) <= 1.0
    assert shannon_entropy_sqi(spiky) < 0.5


def test_snr():
    rng   = np.random.default_rng(0)
    clean = np.sin(2 * np.pi * 1.2 * T) + 0.3 * np.sin(2 * np.pi * 2.4 * T)
    noisy = clean + rng.normal(0, 2.0, len(T))
    assert snr_hr_db(clean, FS) > 10 > snr_hr_db(noisy, FS)


def test_relative_power():
    assert relative_power_sqi(np.sin(2 * np.pi * 1.5 * T), FS) > 0.9   # 90 bpm
    assert relative_power_sqi(np.sin(2 * np.pi * 0.3 * T), FS) < 0.1   # breathing-like


# --- fake windows ------------------------------------------------------------

def _win(subject, t, **vals):
    """A window with every metric set to a normal value, unless given in vals."""
    w = PPGWindow.__new__(PPGWindow)
    base = dict(pi=1.0, acc_energy_std=60.0)
    for ch in ('red', 'ir', 'green', 'blue'):
        base.update({f'skewness_{ch}': 0.5, f'kurtosis_{ch}': 2.5, f'snr_{ch}': 5.0,
                     f'entropy_{ch}': 0.9, f'rel_power_{ch}': 0.5})
    base.update(vals)
    for f, v in dict(subject=subject, episode=None, t_start_s=float(t), **base).items():
        setattr(w, f, v)
    return w


def _recording(subject, n=60, level=1.0, seed=0):
    """n windows, 2.5 s apart, with a little noise."""
    rng = np.random.default_rng(seed)
    return [_win(subject, 2.5 * i, pi=level * (1 + rng.normal(0, 0.05)),
                 snr_ir=5 + rng.normal(0, 0.5), acc_energy_std=60 + rng.normal(0, 2))
            for i in range(n)]


def _random_windows(n=400, seed=0):
    """Windows with random, realistic-looking metric values."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        vals = dict(pi=float(rng.lognormal(-5.5, 0.8)),
                    acc_energy_std=float(50 + rng.exponential(32)))
        for ch in ('red', 'ir', 'green', 'blue'):
            vals.update({f'skewness_{ch}': rng.normal(0, 0.63),
                         f'kurtosis_{ch}': rng.normal(3, 1.0),
                         f'snr_{ch}': rng.normal(-8.4, 6.3),
                         f'entropy_{ch}': rng.normal(0.9, 0.05),
                         f'rel_power_{ch}': rng.uniform(0.01, 0.97)})
        out.append(_win(f's{i % 5}', float(i), **vals))
    return out


# --- personal baseline -------------------------------------------------------

def test_baseline_rejects_only_in_bad_direction():
    rec = _recording('s1')
    rec[40].snr_ir = 0.0          # SNR collapse  -> reject (too low is bad)
    rec[41].snr_ir = 20.0         # SNR jump      -> keep   (too high is fine)
    rec[42].pi = 3.0              # PI jump       -> reject (both sides are bad)
    rec[43].acc_energy_std = 300  # movement      -> reject (too high is bad)
    kept = {id(w) for w in apply_baseline_sqi(rec, baseline_s=60, window_s=10,
                                              k=3.0, ir_only=True, verbose=False)}
    assert id(rec[41]) in kept
    assert not {id(rec[i]) for i in (40, 42, 43)} & kept
    assert len(kept) >= len(rec) - 6


def test_baseline_is_per_recording():
    # Subject 2 has 5x the PI of subject 1.  One shared limit would cut one of
    # them; personal baselines keep both.
    wins = _recording('s1', seed=1) + _recording('s2', level=5.0, seed=2)
    kept = apply_baseline_sqi(wins, baseline_s=60, window_s=10, k=3.0,
                              ir_only=True, verbose=False)
    for subj in ('s1', 's2'):
        assert sum(w.subject == subj for w in kept) >= 54, subj


def test_baseline_uses_first_minute_only():
    rec = _recording('s1')
    for w in rec[30:]:
        w.snr_ir = -50.0          # everything after the first minute is bad
    med, _, n = fit_baselines(rec, 60, 10, fields=['snr_ir'])[('s1', None)]['snr_ir']
    assert n == 21 and abs(med - 5) < 0.5, (med, n)   # windows ending by 60 s


def test_baseline_independent_windows_do_not_overlap():
    rec = _recording('s1')                    # 10-s windows every 2.5 s
    base = fit_baselines(rec, 60, 10, fields=['snr_ir'], independent=True)
    assert base[('s1', None)]['snr_ir'][2] == 6   # starts 0, 10, ..., 50 s

    rec = [_win('s1', 1.24 * i) for i in range(60)]  # 5-s windows every 1.24 s
    base = fit_baselines(rec, 60, 5, fields=['snr_ir'], independent=True)
    assert base[('s1', None)]['snr_ir'][2] == 9


# --- fixed and percentile limits ---------------------------------------------

def test_fixed_limits_and_typos():
    wins = [_win('s', 0), _win('s', 1, kurtosis_green=5.0), _win('s', 2, snr_ir=np.nan)]
    kept = apply_sqi(wins, {'kurtosis_max': 3.5}, verbose=False)
    assert kept == [wins[0], wins[2]]               # NaN passes, any channel can fail
    assert len(apply_sqi(wins, {'kurtosis_max': 3.5}, ir_only=True, verbose=False)) == 3
    try:
        apply_sqi(wins, {'kurtosis_maxx': 3.5})
    except ValueError:
        pass
    else:
        raise AssertionError('a misspelt limit must not be ignored')


def test_percentile_cuts_the_requested_share_of_windows():
    wins, pct = _random_windows(), 5.0
    for ir_only in (True, False):
        limits = percentile_thresholds(wins, pct, ir_only)
        for metric in METRICS:
            single = {k: v for k, v in limits.items() if k.startswith(metric + '_')}
            frac = len(apply_sqi(wins, single, ir_only, verbose=False)) / len(wins)
            assert abs(frac - (1 - pct / 100)) < 0.02, (ir_only, metric, frac)


def test_percentile_follows_metric_directions():
    limits = percentile_thresholds(_random_windows(), 5.0, ir_only=True)
    for metric, side in METRICS.items():
        lo, hi = limits[f'{metric}_min'], limits[f'{metric}_max']
        assert (lo is not None) == (side != 'hi_bound'), metric
        assert (hi is not None) == (side != 'lo_bound'), metric


def test_percentile_overlap_matches_filter():
    wins = _random_windows()
    limits = percentile_thresholds(wins, 5.0)
    n_rejected, n = report_percentile_overlap(wins, limits)
    assert n - len(apply_sqi(wins, limits, verbose=False)) == n_rejected
    assert 0.05 * n <= n_rejected < len(METRICS) * 0.05 * n


def test_composite_percentile_cuts_exactly():
    wins   = _random_windows()
    scaler = fit_composite_sqi_scaler(wins)
    cut    = composite_percentile_threshold(wins, 5.0, scaler)
    assert abs((normalised_composite(wins, scaler) >= cut).mean() - 0.95) < 0.01


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS  {name}')
    print('OK')
