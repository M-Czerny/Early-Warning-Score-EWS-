"""Check that each FEATURE_MODE feeds the right features to the models.  Run: python test_calibration.py"""
from dataclasses import fields

import numpy as np

from calibration import LED_COMBOS_ACDC_RATIO, calibrate_all_combos
from data_extraction import PPGWindow


def _window(subject, i, rng):
    vals = {f.name: float(rng.uniform(0.5, 2.0)) for f in fields(PPGWindow)}
    vals.update(subject=subject, episode=None, window_idx=i, t_start_s=float(i))
    for led in ('red', 'ir', 'green', 'blue'):
        vals[f'acdc_{led}'] = vals[f'ac_{led}'] / vals[f'dc_{led}']
    vals['spo2_ref'] = 95 + 3 * (vals['acdc_red'] - vals['acdc_ir'])
    return PPGWindow(**vals)


if __name__ == '__main__':
    assert LED_COMBOS_ACDC_RATIO['Red-IR'] == ['acdc_red', 'acdc_ir']
    assert len(LED_COMBOS_ACDC_RATIO['Red-IR-Green']) == 3
    assert len(LED_COMBOS_ACDC_RATIO['All-4LEDs']) == 4

    rng  = np.random.default_rng(0)
    wins = [_window(s, i, rng) for s in 'ABCD' for i in range(30)]
    rmse = {}
    for mode in ('R', 'AC_DC', 'AC_DC_RATIO'):
        res = calibrate_all_combos(wins, include_combos=['Red-IR'],
                                   feature_mode=mode, gpr_n_restarts=0)
        folds = next(iter(res.values()))['Linear']   # R mode may rename to IR-Red
        rmse[mode] = np.mean([f.rmse for f in folds])
    # Target is linear in AC/DC, so only that mode can fit it exactly.
    assert rmse['AC_DC_RATIO'] < 1e-6 < min(rmse['R'], rmse['AC_DC']), rmse

    try:
        calibrate_all_combos(wins, feature_mode='bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('bad feature_mode accepted')
    print('ok', rmse)
