"""
Calibration: Leave-One-Subject-Out cross-validation for SpO2 estimation.

Five regression models (Linear, Quadratic, SVR, GPR, DecisionTree) are trained
and evaluated for each LED combination.  The feature is the ratio-of-ratios R
value (or raw AC/DC amplitudes when USE_AC_DC=True), and the target is the
mean reference SpO2 over the window.

Leave-One-Subject-Out (LOSO) cross-validation ensures that the model is always
tested on a subject it has never seen during training, giving a realistic
estimate of how well it will generalise to new individuals.

A zero-order Kalman filter is optionally applied after prediction to smooth
the per-window SpO2 estimates over time and reject sudden outliers.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

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
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Predictions and metrics for one left-out subject in one model fold."""
    model_name:   str
    test_subject: str
    y_true:       np.ndarray   # reference SpO2 values
    y_pred:       np.ndarray   # predicted SpO2 values
    rmse:         float
    mae:          float
    r2:           float
    mean_error:   float        # mean(y_pred - y_true); positive = overprediction


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_models(
    svr_C:               float = 100.0,
    svr_epsilon:         float = 0.1,
    svr_gamma:           Union[float, str] = 'scale',
    gpr_alpha:           float = 1e-6,
    gpr_n_restarts:      int   = 5,
) -> Dict[str, Pipeline]:
    """Return a fresh set of sklearn Pipeline objects — one per model type."""
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

def calibrate(
    windows,
    random_seed: int = 42,
) -> Dict[str, List[FoldResult]]:
    """
    Train and evaluate SpO2 regressors using Leave-One-Subject-Out CV.

    For each fold the training windows are shuffled before fitting so that
    any ordering in the data does not bias the learner.

    Feature : R (ratio of ratios)
    Target  : spo2_ref (mean reference SpO2 over the window)

    Windows with NaN R or NaN spo2_ref are silently dropped.

    Parameters
    ----------
    windows     : list of PPGWindow objects (output of apply_sqi)
    random_seed : seed for the training-set shuffle

    Returns
    -------
    Dict mapping model name → list of FoldResult, one entry per left-out subject.
    """
    rng = np.random.default_rng(random_seed)

    valid    = [w for w in windows
                if not np.isnan(w.R_red_ir) and not np.isnan(w.spo2_ref)]
    subjects = sorted({w.subject for w in valid})
    models   = _build_models()
    results  = {name: [] for name in models}

    print(f"\n  Leave-One-Subject-Out CV  ({len(subjects)} subjects, "
          f"{len(valid)} windows)\n")

    for test_subject in subjects:
        train_wins = [w for w in valid if w.subject != test_subject]
        test_wins  = [w for w in valid if w.subject == test_subject]

        if not train_wins or not test_wins:
            continue

        # Shuffle training set
        train_wins = list(train_wins)
        rng.shuffle(train_wins)

        X_train = np.array([w.R_red_ir  for w in train_wins]).reshape(-1, 1)
        y_train = np.array([w.spo2_ref  for w in train_wins])
        X_test  = np.array([w.R_red_ir  for w in test_wins]).reshape(-1, 1)
        y_test  = np.array([w.spo2_ref for w in test_wins])

        fold_row = f"  test={test_subject:<12}  train_n={len(train_wins):>4}  test_n={len(test_wins):>3}"
        metrics  = []

        for name, template in models.items():
            fitted = clone(template)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)

            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            mae  = float(mean_absolute_error(y_test, y_pred))
            r2   = float(r2_score(y_test, y_pred))

            results[name].append(FoldResult(
                model_name   = name,
                test_subject = test_subject,
                y_true       = y_test.copy(),
                y_pred       = y_pred.copy(),
                rmse         = rmse,
                mae          = mae,
                r2           = r2,
                mean_error   = float(np.mean(y_pred - y_test)),
            ))
            metrics.append(f"{name[:4]}={rmse:.2f}")

        print(fold_row + "  |  RMSE: " + "  ".join(metrics))

    return results


# ---------------------------------------------------------------------------
# Multi-LED combination calibration
# ---------------------------------------------------------------------------

# Fields on PPGWindow that correspond to each LED combination.
# Pairs use one R feature; triples use three pairwise R values; quad uses all six.
LED_COMBOS: Dict[str, List[str]] = {
    # 6 pairs
    'Red-IR':          ['R_red_ir'],
    'Red-Green':       ['R_red_green'],
    'Red-Blue':        ['R_red_blue'],
    'IR-Green':        ['R_ir_green'],
    'IR-Blue':         ['R_ir_blue'],
    'Green-Blue':      ['R_green_blue'],
    # 4 triples (all pairwise R values within each triple)
    'Red-IR-Green':    ['R_red_ir', 'R_red_green', 'R_ir_green'],
    'Red-IR-Blue':     ['R_red_ir', 'R_red_blue',  'R_ir_blue'],
    'Red-Green-Blue':  ['R_red_green', 'R_red_blue', 'R_green_blue'],
    'IR-Green-Blue':   ['R_ir_green',  'R_ir_blue',  'R_green_blue'],
    # 1 quadruple (all six pairwise R values)
    'All-4LEDs':       ['R_red_ir', 'R_red_green', 'R_red_blue',
                        'R_ir_green', 'R_ir_blue', 'R_green_blue'],
}

# For each pair combo: (forward R field, inverse R field, inverse combo name).
# The combo name encodes the direction: first LED is numerator, second is denominator.
PAIR_R_FIELDS: Dict[str, tuple] = {
    'Red-IR':     ('R_red_ir',     'R_ir_red',     'IR-Red'),
    'Red-Green':  ('R_red_green',  'R_green_red',  'Green-Red'),
    'Red-Blue':   ('R_red_blue',   'R_blue_red',   'Blue-Red'),
    'IR-Green':   ('R_ir_green',   'R_green_ir',   'Green-IR'),
    'IR-Blue':    ('R_ir_blue',    'R_blue_ir',    'Blue-IR'),
    'Green-Blue': ('R_green_blue', 'R_blue_green', 'Blue-Green'),
}

# Constituent pairs for each multi-LED combo (order matches LED_COMBOS field order).
COMBO_PAIRS: Dict[str, List[str]] = {
    'Red-IR-Green':   ['Red-IR', 'Red-Green', 'IR-Green'],
    'Red-IR-Blue':    ['Red-IR', 'Red-Blue',  'IR-Blue'],
    'Red-Green-Blue': ['Red-Green', 'Red-Blue',  'Green-Blue'],
    'IR-Green-Blue':  ['IR-Green',  'IR-Blue',   'Green-Blue'],
    'All-4LEDs':      ['Red-IR', 'Red-Green', 'Red-Blue', 'IR-Green', 'IR-Blue', 'Green-Blue'],
}

# AC/DC variants — each LED contributes its bandpass std (AC) and lowpass mean (DC).
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


def calibrate_multi(
    windows,
    fields:          List[str],
    random_seed:     int              = 42,
    verbose:         bool             = False,
    svr_C:           float            = 100.0,
    svr_epsilon:     float            = 0.1,
    svr_gamma:       Union[float, str] = 'scale',
    gpr_alpha:       float            = 1e-6,
    gpr_n_restarts:  int              = 5,
) -> Dict[str, List[FoldResult]]:
    """
    LOSO cross-validation using an arbitrary set of PPGWindow fields as features.

    `fields` is a list of PPGWindow attribute names, e.g. ['R_red_ir'] for a
    single pair or ['R_red_ir', 'R_red_green', 'R_ir_green'] for a triple.
    Windows where any listed field or spo2_ref is NaN are silently dropped.
    """
    rng = np.random.default_rng(random_seed)

    valid = [
        w for w in windows
        if all(not np.isnan(getattr(w, f)) for f in fields)
        and not np.isnan(w.spo2_ref)
    ]
    subjects = sorted({w.subject for w in valid})
    models   = _build_models(
        svr_C=svr_C, svr_epsilon=svr_epsilon, svr_gamma=svr_gamma,
        gpr_alpha=gpr_alpha, gpr_n_restarts=gpr_n_restarts,
    )
    results  = {name: [] for name in models}

    if verbose:
        print(f"\n  LOSO CV  ({len(subjects)} subjects, {len(valid)} windows, "
              f"features: {fields})\n")

    for test_subject in subjects:
        train_wins = [w for w in valid if w.subject != test_subject]
        test_wins  = [w for w in valid if w.subject == test_subject]

        if not train_wins or not test_wins:
            continue

        train_wins = list(train_wins)
        rng.shuffle(train_wins)

        X_train = np.array([[getattr(w, f) for f in fields] for w in train_wins])
        y_train = np.array([w.spo2_ref for w in train_wins])
        X_test  = np.array([[getattr(w, f) for f in fields] for w in test_wins])
        y_test  = np.array([w.spo2_ref for w in test_wins])

        for name, template in models.items():
            fitted = clone(template)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)

            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            mae  = float(mean_absolute_error(y_test, y_pred))
            r2   = float(r2_score(y_test, y_pred))

            results[name].append(FoldResult(
                model_name   = name,
                test_subject = test_subject,
                y_true       = y_test.copy(),
                y_pred       = y_pred.copy(),
                rmse         = rmse,
                mae          = mae,
                r2           = r2,
                mean_error   = float(np.mean(y_pred - y_test)),
            ))

    return results


def _best_mean_rmse(results: Dict[str, List[FoldResult]]) -> float:
    """Lowest mean RMSE across folds for the best-performing model."""
    return float(min(np.mean([f.rmse for f in folds]) for folds in results.values()))


def calibrate_all_combos(
    windows,
    random_seed:    int                 = 42,
    include_combos: Optional[List[str]] = None,
    use_ac_dc:      bool                = False,
    svr_C:          float               = 100.0,
    svr_epsilon:    float               = 0.1,
    svr_gamma:      Union[float, str]   = 'scale',
    gpr_alpha:      float               = 1e-6,
    gpr_n_restarts: int                 = 5,
) -> Dict[str, Dict[str, List[FoldResult]]]:
    """
    Run LOSO calibration for LED combinations.

    include_combos : list of combo names to run (must be keys of LED_COMBOS).
                     None (default) runs all combinations.
    use_ac_dc      : False (default) → ratio-of-ratios mode (see below).
                     True            → raw AC and DC values per LED channel.

    R-ratio mode (use_ac_dc=False)
    --------------------------------
    For each pair combo both the forward R and its inverse are evaluated; the
    direction with the lower best-model RMSE is kept.  Triple and quad combos
    then use the winning direction for each of their constituent pairs.

    Returns a nested dict: combo_name → model_name → list[FoldResult].
    """
    if use_ac_dc:
        if include_combos is not None:
            unknown = set(include_combos) - set(LED_COMBOS_ACDC)
            if unknown:
                raise ValueError(f"Unknown LED combo(s): {unknown}. "
                                 f"Valid names: {list(LED_COMBOS_ACDC)}")
        combos = {k: v for k, v in LED_COMBOS_ACDC.items()
                  if include_combos is None or k in include_combos}
        combo_results: Dict[str, Dict[str, List[FoldResult]]] = {}
        for combo, fields in combos.items():
            print(f"  {combo:<18}  [AC/DC]  {len(fields)} feature(s): {', '.join(fields)}")
            combo_results[combo] = calibrate_multi(
                windows, fields, random_seed=random_seed,
                svr_C=svr_C, svr_epsilon=svr_epsilon, svr_gamma=svr_gamma,
                gpr_alpha=gpr_alpha, gpr_n_restarts=gpr_n_restarts,
            )
        return combo_results

    # --- R-ratio mode ---------------------------------------------------------
    if include_combos is not None:
        unknown = set(include_combos) - set(LED_COMBOS)
        if unknown:
            raise ValueError(f"Unknown LED combo(s): {unknown}. "
                             f"Valid names: {list(LED_COMBOS)}")

    requested = set(include_combos) if include_combos is not None else set(LED_COMBOS)

    # Determine which pairs must be evaluated (directly requested + constituent
    # pairs of any requested multi-LED combo).
    needed_pairs: set = set()
    for combo in requested:
        if combo in PAIR_R_FIELDS:
            needed_pairs.add(combo)
        elif combo in COMBO_PAIRS:
            needed_pairs.update(COMBO_PAIRS[combo])

    # Phase 1 — calibrate each needed pair in both directions, pick winner.
    # pair_best_field   : original pair name → winning field (used internally for triples/quads)
    # pair_output_name  : original pair name → output key (flipped if inverse wins)
    pair_best_field:   Dict[str, str]                          = {}
    pair_output_name:  Dict[str, str]                          = {}
    pair_best_results: Dict[str, Dict[str, List[FoldResult]]]  = {}

    for pair, (fwd, inv, inv_name) in PAIR_R_FIELDS.items():
        if pair not in needed_pairs:
            continue
        res_fwd  = calibrate_multi(
            windows, [fwd], random_seed=random_seed,
            svr_C=svr_C, svr_epsilon=svr_epsilon, svr_gamma=svr_gamma,
            gpr_alpha=gpr_alpha, gpr_n_restarts=gpr_n_restarts,
        )
        res_inv  = calibrate_multi(
            windows, [inv], random_seed=random_seed,
            svr_C=svr_C, svr_epsilon=svr_epsilon, svr_gamma=svr_gamma,
            gpr_alpha=gpr_alpha, gpr_n_restarts=gpr_n_restarts,
        )
        rmse_fwd = _best_mean_rmse(res_fwd)
        rmse_inv = _best_mean_rmse(res_inv)
        if rmse_fwd <= rmse_inv:
            pair_best_field[pair]   = fwd
            pair_output_name[pair]  = pair        # forward keeps the original name
            pair_best_results[pair] = res_fwd
            winner_field, loser_field, wr, lr = fwd, inv, rmse_fwd, rmse_inv
            winner_name = pair
        else:
            pair_best_field[pair]   = inv
            pair_output_name[pair]  = inv_name    # inverse flips the name
            pair_best_results[pair] = res_inv
            winner_field, loser_field, wr, lr = inv, fwd, rmse_inv, rmse_fwd
            winner_name = inv_name
        print(f"  {pair:<18}  [R-ratio]  winner: {winner_name} ({winner_field},"
              f" RMSE={wr:.3f})  vs {loser_field} (RMSE={lr:.3f})")

    # Phase 2 — assemble output in LED_COMBOS order.
    combo_results: Dict[str, Dict[str, List[FoldResult]]] = {}
    for combo in LED_COMBOS:
        if combo not in requested:
            continue
        if combo in PAIR_R_FIELDS:
            out_name = pair_output_name[combo]
            combo_results[out_name] = pair_best_results[combo]
        else:
            fields = [pair_best_field[p] for p in COMBO_PAIRS[combo]]
            print(f"  {combo:<18}  [R-ratio]  {len(fields)} feature(s): {', '.join(fields)}")
            combo_results[combo] = calibrate_multi(
                windows, fields, random_seed=random_seed,
                svr_C=svr_C, svr_epsilon=svr_epsilon, svr_gamma=svr_gamma,
                gpr_alpha=gpr_alpha, gpr_n_restarts=gpr_n_restarts,
            )
    return combo_results


def apply_kalman_to_all_combos(
    combo_results: Dict[str, Dict[str, List[FoldResult]]],
    Q:             float = 0.5,
    R:             float = 2.0,
    reject_sigma:  float = 3.0,
) -> Dict[str, Dict[str, List[FoldResult]]]:
    """Apply zero-order Kalman to every combo's per-fold predictions."""
    for results in combo_results.values():
        apply_kalman_to_results(results, Q=Q, R=R, reject_sigma=reject_sigma)
    return combo_results


def summarise_combos(
    combo_results: Dict[str, Dict[str, List[FoldResult]]],
) -> None:
    """Print a compact RMSE table: one row per LED combination."""
    print(f"\n  {'Combination':<18}  {'Best model':<14}  "
          f"{'Best RMSE':>9}  Per-model mean RMSE")
    print("  " + "-" * 80)
    for combo, results in combo_results.items():
        best      = min(results, key=lambda m: np.mean([f.rmse for f in results[m]]))
        best_rmse = np.mean([f.rmse for f in results[best]])
        per_model = "  ".join(
            f"{m[:4]}={np.mean([f.rmse for f in results[m]]):.2f}"
            for m in results
        )
        print(f"  {combo:<18}  {best:<14}  {best_rmse:>9.3f}  {per_model}")


# ---------------------------------------------------------------------------
# Zero-order Kalman filter for prediction smoothing
# ---------------------------------------------------------------------------

def _zero_order_kalman(
    measurements: np.ndarray,
    Q:            float = 0.5,
    R:            float = 2.0,
    reject_sigma: float = 3.0,
) -> np.ndarray:
    """
    Zero-order (constant-model) Kalman filter with outlier rejection.

    Assumes SpO2 is locally constant between consecutive windows.
    If a new prediction deviates more than reject_sigma * sqrt(P + R) from
    the current state estimate it is treated as an outlier: the state is held
    and the covariance is allowed to grow (so a genuine sustained change will
    eventually be accepted).

    Q : process noise variance — how much the true SpO2 can shift per step
    R : measurement noise variance — uncertainty of each regressor prediction
    """
    n = len(measurements)
    if n == 0:
        return measurements.copy()

    filtered = np.empty(n)
    x = float(measurements[0])
    P = float(R)
    filtered[0] = x

    for i in range(1, n):
        P_pred = P + Q
        z      = float(measurements[i])
        innov  = z - x

        if abs(innov) > reject_sigma * np.sqrt(P_pred + R):
            # Outlier: hold state, let covariance grow so future steps can
            # still converge.
            filtered[i] = x
            P = P_pred
        else:
            K = P_pred / (P_pred + R)
            x = x + K * innov
            P = (1.0 - K) * P_pred
            filtered[i] = x

    return filtered


def apply_kalman_to_results(
    results:      dict,
    Q:            float = 0.5,
    R:            float = 2.0,
    reject_sigma: float = 3.0,
) -> dict:
    """
    Apply _zero_order_kalman to y_pred for every fold in results, then
    recalculate RMSE, MAE and R² in-place.

    The filter is applied per-subject fold so the state resets between subjects.
    """
    for folds in results.values():
        for fold in folds:
            fold.y_pred = _zero_order_kalman(
                fold.y_pred, Q=Q, R=R, reject_sigma=reject_sigma,
            )
            fold.rmse       = float(np.sqrt(np.mean((fold.y_true - fold.y_pred) ** 2)))
            fold.mae        = float(mean_absolute_error(fold.y_true, fold.y_pred))
            fold.r2         = float(r2_score(fold.y_true, fold.y_pred))
            fold.mean_error = float(np.mean(fold.y_pred - fold.y_true))
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(results: Dict[str, List[FoldResult]]) -> None:
    """Print mean ± std of RMSE, MAE, R² and mean error across all folds per model."""
    print(f"\n  {'Model':<14} {'RMSE':>6}±{'std':>5}   {'MAE':>6}±{'std':>5}   "
          f"{'R²':>6}±{'std':>5}   {'Bias':>6}±{'std':>5}")
    print("  " + "-" * 80)
    for name, folds in results.items():
        rmses  = np.array([f.rmse       for f in folds])
        maes   = np.array([f.mae        for f in folds])
        r2s    = np.array([f.r2         for f in folds])
        biases = np.array([f.mean_error for f in folds])
        print(f"  {name:<14} "
              f"{rmses.mean():>6.3f}±{rmses.std():>5.3f}   "
              f"{maes.mean():>6.3f}±{maes.std():>5.3f}   "
              f"{r2s.mean():>6.3f}±{r2s.std():>5.3f}   "
              f"{biases.mean():>+6.3f}±{biases.std():>5.3f}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning helpers
# ---------------------------------------------------------------------------

def _model_mean_rmse(
    combo_results: Dict[str, Dict[str, List[FoldResult]]],
    model_name:    str,
) -> float:
    """Mean RMSE for one model across all LED combos."""
    rmses = [
        float(np.mean([f.rmse for f in folds]))
        for results in combo_results.values()
        if model_name in results
        for folds in [results[model_name]]
        if folds
    ]
    return float(np.mean(rmses)) if rmses else float('nan')


def grid_search_model_params(
    windows,
    svr_C_grid:       Sequence[float] = (0.1, 1.0, 10.0, 100.0, 500.0),
    svr_epsilon_grid: Sequence[float] = (0.01, 0.1, 0.5, 1.0),
    gpr_alpha_grid:   Sequence[float] = (0.001, 0.01, 0.1, 0.5, 1.0, 2.0),
    random_seed:      int             = 42,
    include_combos                    = None,
    use_ac_dc:        bool            = False,
    svr_gamma                         = 'scale',
    gpr_n_restarts:   int             = 5,
) -> dict:
    """
    Sweep SVR (C, epsilon) and GPR (alpha) independently.

    Runs a 2-D grid over SVR C × epsilon (GPR alpha held at its median grid
    value), then a 1-D sweep over GPR alpha (SVR held at current defaults).
    Reports the per-model RMSE for each config and prints a sorted table.

    Returns dict with keys 'svr' and 'gpr', each a list of result dicts
    sorted by ascending RMSE.
    """
    gpr_alpha_default = sorted(gpr_alpha_grid)[len(gpr_alpha_grid) // 2]
    svr_C_default     = 100.0
    svr_eps_default   = 0.1

    # --- SVR: C × epsilon grid -----------------------------------------------
    svr_rows: list = []
    print(f"\n[Model tuning]  SVR: C × epsilon  "
          f"(GPR alpha fixed at {gpr_alpha_default})")
    print(f"  {'C':>10}  {'epsilon':>9}  {'SVR RMSE':>10}")
    for C in svr_C_grid:
        for eps in svr_epsilon_grid:
            cv   = calibrate_all_combos(
                windows, random_seed=random_seed,
                include_combos=include_combos, use_ac_dc=use_ac_dc,
                svr_C=C, svr_epsilon=eps, svr_gamma=svr_gamma,
                gpr_alpha=gpr_alpha_default, gpr_n_restarts=gpr_n_restarts,
            )
            rmse = _model_mean_rmse(cv, 'SVR')
            svr_rows.append({'C': C, 'epsilon': eps, 'rmse': rmse})
            print(f"  {C:>10.2f}  {eps:>9.3f}  {rmse:>10.4f}")

    svr_rows.sort(key=lambda r: r['rmse'])
    best = svr_rows[0]
    print(f"  → best: C={best['C']}, epsilon={best['epsilon']}, RMSE={best['rmse']:.4f}")

    # --- GPR: alpha sweep ----------------------------------------------------
    gpr_rows: list = []
    print(f"\n[Model tuning]  GPR: alpha  "
          f"(SVR C={svr_C_default}, epsilon={svr_eps_default})")
    print(f"  {'alpha':>12}  {'GPR RMSE':>10}")
    for alpha in gpr_alpha_grid:
        cv   = calibrate_all_combos(
            windows, random_seed=random_seed,
            include_combos=include_combos, use_ac_dc=use_ac_dc,
            svr_C=svr_C_default, svr_epsilon=svr_eps_default, svr_gamma=svr_gamma,
            gpr_alpha=alpha, gpr_n_restarts=gpr_n_restarts,
        )
        rmse = _model_mean_rmse(cv, 'GPR')
        gpr_rows.append({'alpha': alpha, 'rmse': rmse})
        print(f"  {alpha:>12.4f}  {rmse:>10.4f}")

    gpr_rows.sort(key=lambda r: r['rmse'])
    best = gpr_rows[0]
    print(f"  → best: alpha={best['alpha']}, RMSE={best['rmse']:.4f}")

    return {'svr': svr_rows, 'gpr': gpr_rows}


def sweep_kalman_params(
    combo_results_base: Dict[str, Dict[str, List[FoldResult]]],
    Q_grid:     Sequence[float] = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0),
    R_grid:     Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    sigma_grid: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
) -> list:
    """
    Sweep Kalman filter parameters on pre-computed LOSO predictions.

    combo_results_base must be the output of calibrate_all_combos() BEFORE
    any Kalman filter has been applied so that y_pred holds raw predictions.

    Deep-copies the results for each parameter combination so the originals
    are never mutated.  Returns a list of result dicts sorted by best RMSE.
    """
    n_combos = len(Q_grid) * len(R_grid) * len(sigma_grid)
    print(f"\n[Kalman tuning]  Q × R × sigma  ({n_combos} combinations)")
    print(f"  {'Q':>6}  {'R':>6}  {'sigma':>6}  {'best_RMSE':>10}")

    rows: list = []
    for Q in Q_grid:
        for R in R_grid:
            for sigma in sigma_grid:
                cv_copy  = copy.deepcopy(combo_results_base)
                apply_kalman_to_all_combos(cv_copy, Q=Q, R=R, reject_sigma=sigma)
                best_rmse = min(
                    float(np.mean([f.rmse for f in folds]))
                    for results in cv_copy.values()
                    for folds in results.values()
                    if folds
                )
                rows.append({'Q': Q, 'R': R, 'sigma': sigma, 'rmse': best_rmse})
                print(f"  {Q:>6.2f}  {R:>6.2f}  {sigma:>6.2f}  {best_rmse:>10.4f}")

    rows.sort(key=lambda r: r['rmse'])
    best = rows[0]
    print(f"  → best: Q={best['Q']}, R={best['R']}, sigma={best['sigma']}, "
          f"RMSE={best['rmse']:.4f}")
    return rows
