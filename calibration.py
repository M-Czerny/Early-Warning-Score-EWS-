"""
Calibration: learn to map window features to SpO2, and measure how well it works.

Five regression models (Linear, Quadratic, SVR, GPR, DecisionTree) are tested
for each LED combination with Leave-One-Subject-Out (LOSO) cross-validation:
each subject in turn is held out, the models are trained on everyone else, and
then tested on the held-out subject.  This shows how well the model works on a
person it has never seen.

Afterwards a simple Kalman filter can smooth the predictions over time.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Predictions and scores of one model on one held-out subject."""
    model_name:   str
    test_subject: str
    y_true:       np.ndarray   # reference SpO2
    y_pred:       np.ndarray   # predicted SpO2
    rmse:         float
    mae:          float
    r2:           float
    mean_error:   float        # mean(pred - true); positive = predicts too high

    def rescore(self) -> None:
        """Recompute the scores after y_pred has changed."""
        self.rmse       = float(np.sqrt(np.mean((self.y_true - self.y_pred) ** 2)))
        self.mae        = float(mean_absolute_error(self.y_true, self.y_pred))
        self.r2         = float(r2_score(self.y_true, self.y_pred))
        self.mean_error = float(np.mean(self.y_pred - self.y_true))


Results      = Dict[str, List[FoldResult]]   # model name -> one result per subject
ComboResults = Dict[str, Results]            # LED combination -> Results


def mean_rmse(folds: List[FoldResult]) -> float:
    return float(np.mean([f.rmse for f in folds]))


def best_rmse(results: Results) -> float:
    """Mean RMSE of the best model."""
    return min(mean_rmse(folds) for folds in results.values() if folds)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models(
    svr_C:          float = 100.0,
    svr_epsilon:    float = 0.1,
    svr_gamma             = 'scale',
    gpr_alpha:      float = 1e-6,
    gpr_n_restarts: int   = 5,
) -> Dict[str, Pipeline]:
    """A fresh, untrained copy of each model."""
    return {
        'Linear': Pipeline([
            ('scaler', StandardScaler()),
            ('reg',    LinearRegression()),
        ]),
        'Quadratic': Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
            ('reg',    LinearRegression()),
        ]),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('reg',    SVR(kernel='rbf', C=svr_C, epsilon=svr_epsilon, gamma=svr_gamma)),
        ]),
        'GPR': Pipeline([
            ('scaler', StandardScaler()),
            ('reg',    GaussianProcessRegressor(
                           kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
                           alpha=gpr_alpha,
                           n_restarts_optimizer=gpr_n_restarts,
                           normalize_y=True,
                           random_state=0,
                       )),
        ]),
        'DecisionTree': Pipeline([
            ('reg', DecisionTreeRegressor(max_depth=5, random_state=0)),
        ]),
    }


# ---------------------------------------------------------------------------
# Leave-One-Subject-Out cross-validation
# ---------------------------------------------------------------------------

def calibrate_multi(
    windows,
    fields:      List[str],
    random_seed: int = 42,
    only_models: Optional[Sequence[str]] = None,
    **model_params,
) -> Results:
    """
    LOSO cross-validation with the given PPGWindow fields as features.

    fields       : e.g. ['R_red_ir'], or ['ac_red', 'dc_red', 'ac_ir', 'dc_ir']
    only_models  : train only these models (GPR is slow, so tuning skips it)
    model_params : passed to build_models (svr_C, gpr_alpha, ...)

    Windows with a NaN feature or NaN reference SpO2 are skipped.
    """
    rng = np.random.default_rng(random_seed)

    valid = [w for w in windows
             if not np.isnan(w.spo2_ref)
             and not any(np.isnan(getattr(w, f)) for f in fields)]
    models = build_models(**model_params)
    if only_models is not None:
        unknown = set(only_models) - set(models)
        if unknown:
            raise ValueError(f"unknown model(s): {sorted(unknown)}")
        models = {k: v for k, v in models.items() if k in only_models}
    results: Results = {name: [] for name in models}

    def features(ws):
        return np.array([[getattr(w, f) for f in fields] for w in ws])

    for test_subject in sorted({w.subject for w in valid}):
        train = [w for w in valid if w.subject != test_subject]
        test  = [w for w in valid if w.subject == test_subject]
        if not train:
            continue
        rng.shuffle(train)        # so the order of the data cannot bias training

        X_train, y_train = features(train), np.array([w.spo2_ref for w in train])
        X_test,  y_test  = features(test),  np.array([w.spo2_ref for w in test])

        for name, template in models.items():
            model = clone(template)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                model.fit(X_train, y_train)
            fold = FoldResult(name, test_subject, y_test.copy(), model.predict(X_test),
                              0.0, 0.0, 0.0, 0.0)
            fold.rescore()
            results[name].append(fold)

    return results


# ---------------------------------------------------------------------------
# LED combinations
# ---------------------------------------------------------------------------

# Feature mode 'R': ratio-of-ratios features per combination.
# A pair uses its one R value; a triple uses its three pairwise R values;
# all four LEDs use all six.
LED_COMBOS: Dict[str, List[str]] = {
    'Red-IR':          ['R_red_ir'],
    'Red-Green':       ['R_red_green'],
    'Red-Blue':        ['R_red_blue'],
    'IR-Green':        ['R_ir_green'],
    'IR-Blue':         ['R_ir_blue'],
    'Green-Blue':      ['R_green_blue'],
    'Red-IR-Green':    ['R_red_ir', 'R_red_green', 'R_ir_green'],
    'Red-IR-Blue':     ['R_red_ir', 'R_red_blue',  'R_ir_blue'],
    'Red-Green-Blue':  ['R_red_green', 'R_red_blue', 'R_green_blue'],
    'IR-Green-Blue':   ['R_ir_green',  'R_ir_blue',  'R_green_blue'],
    'All-4LEDs':       ['R_red_ir', 'R_red_green', 'R_red_blue',
                        'R_ir_green', 'R_ir_blue', 'R_green_blue'],
}

# Each pair can also be used upside down (e.g. IR/Red instead of Red/IR).
# pair -> (forward field, inverse field, name of the inverse pair)
PAIR_R_FIELDS: Dict[str, tuple] = {
    'Red-IR':     ('R_red_ir',     'R_ir_red',     'IR-Red'),
    'Red-Green':  ('R_red_green',  'R_green_red',  'Green-Red'),
    'Red-Blue':   ('R_red_blue',   'R_blue_red',   'Blue-Red'),
    'IR-Green':   ('R_ir_green',   'R_green_ir',   'Green-IR'),
    'IR-Blue':    ('R_ir_blue',    'R_blue_ir',    'Blue-IR'),
    'Green-Blue': ('R_green_blue', 'R_blue_green', 'Blue-Green'),
}

# The pairs inside each larger combination (same order as in LED_COMBOS).
COMBO_PAIRS: Dict[str, List[str]] = {
    'Red-IR-Green':   ['Red-IR', 'Red-Green', 'IR-Green'],
    'Red-IR-Blue':    ['Red-IR', 'Red-Blue',  'IR-Blue'],
    'Red-Green-Blue': ['Red-Green', 'Red-Blue',  'Green-Blue'],
    'IR-Green-Blue':  ['IR-Green',  'IR-Blue',   'Green-Blue'],
    'All-4LEDs':      ['Red-IR', 'Red-Green', 'Red-Blue', 'IR-Green', 'IR-Blue', 'Green-Blue'],
}

# Feature mode 'AC_DC': the AC and DC value of every LED in the combination.
LED_COMBOS_ACDC: Dict[str, List[str]] = {
    'Red-IR':          ['ac_red', 'dc_red', 'ac_ir',    'dc_ir'],
    'Red-Green':       ['ac_red', 'dc_red', 'ac_green', 'dc_green'],
    'Red-Blue':        ['ac_red', 'dc_red', 'ac_blue',  'dc_blue'],
    'IR-Green':        ['ac_ir',  'dc_ir',  'ac_green', 'dc_green'],
    'IR-Blue':         ['ac_ir',  'dc_ir',  'ac_blue',  'dc_blue'],
    'Green-Blue':      ['ac_green', 'dc_green', 'ac_blue', 'dc_blue'],
    'Red-IR-Green':    ['ac_red', 'dc_red', 'ac_ir',    'dc_ir',    'ac_green', 'dc_green'],
    'Red-IR-Blue':     ['ac_red', 'dc_red', 'ac_ir',    'dc_ir',    'ac_blue',  'dc_blue'],
    'Red-Green-Blue':  ['ac_red', 'dc_red', 'ac_green', 'dc_green', 'ac_blue',  'dc_blue'],
    'IR-Green-Blue':   ['ac_ir',  'dc_ir',  'ac_green', 'dc_green', 'ac_blue',  'dc_blue'],
    'All-4LEDs':       ['ac_red', 'dc_red', 'ac_ir',    'dc_ir',
                        'ac_green', 'dc_green', 'ac_blue', 'dc_blue'],
}

# Feature mode 'AC_DC_RATIO': AC/DC of every LED in the combination.
LED_COMBOS_ACDC_RATIO: Dict[str, List[str]] = {
    combo: [f'acdc_{f[3:]}' for f in fields if f.startswith('ac_')]
    for combo, fields in LED_COMBOS_ACDC.items()
}

FEATURE_MODES = {'R': LED_COMBOS, 'AC_DC': LED_COMBOS_ACDC,
                 'AC_DC_RATIO': LED_COMBOS_ACDC_RATIO}


def calibrate_all_combos(
    windows,
    include_combos: Optional[List[str]] = None,
    feature_mode:   str = 'R',
    **kwargs,
) -> ComboResults:
    """
    Run LOSO cross-validation for each LED combination.

    include_combos : combination names to run (keys of LED_COMBOS); None = all
    feature_mode   : 'R', 'AC_DC' or 'AC_DC_RATIO'
    kwargs         : passed to calibrate_multi (random_seed, only_models, svr_C, ...)

    In 'R' mode every pair is tried both ways round (e.g. Red/IR and IR/Red)
    and the better one is kept.  If the inverse wins the result is stored
    under the inverse name (e.g. 'IR-Red').  Larger combinations then use the
    winning direction of each of their pairs.

    Returns {combination: {model: [FoldResult per subject]}}.
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode {feature_mode!r}. "
                         f"Valid: {list(FEATURE_MODES)}")
    requested = list(LED_COMBOS) if include_combos is None else include_combos
    unknown = set(requested) - set(LED_COMBOS)
    if unknown:
        raise ValueError(f"Unknown LED combo(s): {unknown}. Valid names: {list(LED_COMBOS)}")
    requested = [c for c in LED_COMBOS if c in requested]    # keep the standard order

    out: ComboResults = {}
    if feature_mode != 'R':
        for combo in requested:
            fields = FEATURE_MODES[feature_mode][combo]
            print(f"  {combo:<18}  [{feature_mode}]  {len(fields)} feature(s): {', '.join(fields)}")
            out[combo] = calibrate_multi(windows, fields, **kwargs)
        return out

    # Step 1: try each needed pair both ways round and keep the better direction.
    needed = {p for c in requested for p in COMBO_PAIRS.get(c, [c])}
    best_field: Dict[str, str] = {}
    for pair, (fwd, inv, inv_name) in PAIR_R_FIELDS.items():
        if pair not in needed:
            continue
        res_fwd = calibrate_multi(windows, [fwd], **kwargs)
        res_inv = calibrate_multi(windows, [inv], **kwargs)
        rmse_fwd, rmse_inv = best_rmse(res_fwd), best_rmse(res_inv)
        if rmse_fwd <= rmse_inv:
            best_field[pair], name, res = fwd, pair, res_fwd
        else:
            best_field[pair], name, res = inv, inv_name, res_inv
        print(f"  {pair:<18}  [R-ratio]  winner: {name}  "
              f"(RMSE {fwd}={rmse_fwd:.3f}, {inv}={rmse_inv:.3f})")
        if pair in requested:
            out[name] = res

    # Step 2: larger combinations use the winning field of each pair.
    for combo in requested:
        if combo in COMBO_PAIRS:
            fields = [best_field[p] for p in COMBO_PAIRS[combo]]
            print(f"  {combo:<18}  [R-ratio]  {len(fields)} feature(s): {', '.join(fields)}")
            out[combo] = calibrate_multi(windows, fields, **kwargs)
    return out


# ---------------------------------------------------------------------------
# Kalman filter (smooths predictions over time)
# ---------------------------------------------------------------------------

def zero_order_kalman(
    measurements: np.ndarray,
    Q:            float = 0.5,
    R:            float = 2.0,
    reject_sigma: float = 3.0,
) -> np.ndarray:
    """
    Smooth a sequence of SpO2 predictions, assuming SpO2 changes slowly.

    Q            : how much the true SpO2 may change from one window to the next
    R            : how noisy a single prediction is
    reject_sigma : a prediction further than this many standard deviations
                   from the current estimate is ignored as an outlier.  The
                   uncertainty still grows, so a real lasting change is
                   accepted after a few windows.
    """
    n = len(measurements)
    if n == 0:
        return measurements.copy()

    out = np.empty(n)
    x = float(measurements[0])   # current estimate
    P = float(R)                 # uncertainty of the estimate
    out[0] = x

    for i in range(1, n):
        P_pred = P + Q
        innov  = float(measurements[i]) - x
        if abs(innov) > reject_sigma * np.sqrt(P_pred + R):
            P = P_pred                     # outlier: keep the estimate
        else:
            K = P_pred / (P_pred + R)
            x = x + K * innov
            P = (1.0 - K) * P_pred
        out[i] = x
    return out


def apply_kalman_to_all_combos(
    combo_results: ComboResults,
    Q:             float = 0.5,
    R:             float = 2.0,
    reject_sigma:  float = 3.0,
) -> ComboResults:
    """Smooth every subject's predictions (in place) and recompute the scores."""
    for results in combo_results.values():
        for folds in results.values():
            for fold in folds:            # one subject at a time
                fold.y_pred = zero_order_kalman(fold.y_pred, Q=Q, R=R,
                                                reject_sigma=reject_sigma)
                fold.rescore()
    return combo_results


# ---------------------------------------------------------------------------
# Printed summaries
# ---------------------------------------------------------------------------

def summarise(results: Results) -> None:
    """Print mean ± std of RMSE, MAE, R² and bias over subjects, per model."""
    print(f"\n  {'Model':<14} {'RMSE':>6}±{'std':>5}   {'MAE':>6}±{'std':>5}   "
          f"{'R²':>6}±{'std':>5}   {'Bias':>6}±{'std':>5}")
    print("  " + "-" * 80)
    for name, folds in results.items():
        rmse = np.array([f.rmse for f in folds])
        mae  = np.array([f.mae for f in folds])
        r2   = np.array([f.r2 for f in folds])
        bias = np.array([f.mean_error for f in folds])
        print(f"  {name:<14} "
              f"{rmse.mean():>6.3f}±{rmse.std():>5.3f}   "
              f"{mae.mean():>6.3f}±{mae.std():>5.3f}   "
              f"{r2.mean():>6.3f}±{r2.std():>5.3f}   "
              f"{bias.mean():>+6.3f}±{bias.std():>5.3f}")


def summarise_combos(combo_results: ComboResults) -> None:
    """Print one line per LED combination with the best model and its RMSE."""
    print(f"\n  {'Combination':<18}  {'Best model':<14}  "
          f"{'Best RMSE':>9}  Per-model mean RMSE")
    print("  " + "-" * 80)
    for combo, results in combo_results.items():
        best = min(results, key=lambda m: mean_rmse(results[m]))
        per_model = "  ".join(f"{m[:4]}={mean_rmse(results[m]):.2f}" for m in results)
        print(f"  {combo:<18}  {best:<14}  {mean_rmse(results[best]):>9.3f}  {per_model}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def _model_rmse(combo_results: ComboResults, model: str) -> float:
    """Mean RMSE of one model, averaged over LED combinations."""
    return float(np.mean([mean_rmse(r[model]) for r in combo_results.values()]))


def grid_search_model_params(
    windows,
    svr_C_grid:       Sequence[float],
    svr_epsilon_grid: Sequence[float],
    gpr_alpha_grid:   Sequence[float],
    **kwargs,
) -> dict:
    """
    Try every SVR (C, epsilon) pair and every GPR alpha; print the RMSE of each.

    kwargs are passed to calibrate_all_combos (include_combos, feature_mode, ...).
    Returns {'svr': rows, 'gpr': rows}, each sorted from best to worst.
    """
    print("\n[Model tuning]  SVR: C × epsilon")
    print(f"  {'C':>10}  {'epsilon':>9}  {'SVR RMSE':>10}")
    svr_rows = []
    for C in svr_C_grid:
        for eps in svr_epsilon_grid:
            cv = calibrate_all_combos(windows, svr_C=C, svr_epsilon=eps,
                                      only_models=['SVR'], **kwargs)
            svr_rows.append({'C': C, 'epsilon': eps, 'rmse': _model_rmse(cv, 'SVR')})
            print(f"  {C:>10.2f}  {eps:>9.3f}  {svr_rows[-1]['rmse']:>10.4f}")
    svr_rows.sort(key=lambda r: r['rmse'])
    print(f"  → best: {svr_rows[0]}")

    print("\n[Model tuning]  GPR: alpha")
    print(f"  {'alpha':>12}  {'GPR RMSE':>10}")
    gpr_rows = []
    for alpha in gpr_alpha_grid:
        cv = calibrate_all_combos(windows, gpr_alpha=alpha, only_models=['GPR'], **kwargs)
        gpr_rows.append({'alpha': alpha, 'rmse': _model_rmse(cv, 'GPR')})
        print(f"  {alpha:>12.4f}  {gpr_rows[-1]['rmse']:>10.4f}")
    gpr_rows.sort(key=lambda r: r['rmse'])
    print(f"  → best: {gpr_rows[0]}")

    return {'svr': svr_rows, 'gpr': gpr_rows}


def sweep_kalman_params(
    combo_results: ComboResults,
    Q_grid:     Sequence[float],
    R_grid:     Sequence[float],
    sigma_grid: Sequence[float],
) -> list:
    """
    Try every Kalman (Q, R, sigma) setting on predictions that have NOT been
    smoothed yet.  The input is copied, never changed.  Returns rows sorted
    from best to worst.
    """
    print(f"\n[Kalman tuning]  Q × R × sigma  "
          f"({len(Q_grid) * len(R_grid) * len(sigma_grid)} combinations)")
    print(f"  {'Q':>6}  {'R':>6}  {'sigma':>6}  {'best_RMSE':>10}")
    rows = []
    for Q in Q_grid:
        for R in R_grid:
            for sigma in sigma_grid:
                smoothed = apply_kalman_to_all_combos(copy.deepcopy(combo_results),
                                                      Q=Q, R=R, reject_sigma=sigma)
                rmse = min(best_rmse(r) for r in smoothed.values())
                rows.append({'Q': Q, 'R': R, 'sigma': sigma, 'rmse': rmse})
                print(f"  {Q:>6.2f}  {R:>6.2f}  {sigma:>6.2f}  {rmse:>10.4f}")
    rows.sort(key=lambda r: r['rmse'])
    print(f"  → best: {rows[0]}")
    return rows
