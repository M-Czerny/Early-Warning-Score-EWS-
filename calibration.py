from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.base import clone
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
    y_true:       np.ndarray   # reference SpO2
    y_pred:       np.ndarray   # predicted SpO2
    rmse:         float
    mae:          float
    r2:           float


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_models() -> Dict[str, Pipeline]:
    return {
        'Linear': Pipeline([
            ('reg', LinearRegression()),
        ]),
        'Quadratic': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('reg',  LinearRegression()),
        ]),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('reg',    SVR(kernel='rbf', C=100.0, epsilon=0.1, gamma='scale')),
        ]),
        'GPR': Pipeline([
            ('scaler', StandardScaler()),
            ('reg',    GaussianProcessRegressor(
                           kernel=ConstantKernel(1.0) * RBF(1.0),
                           n_restarts_optimizer=5,
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
                if not np.isnan(w.R) and not np.isnan(w.spo2_ref)]
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

        X_train = np.array([w.R        for w in train_wins]).reshape(-1, 1)
        y_train = np.array([w.spo2_ref for w in train_wins])
        X_test  = np.array([w.R        for w in test_wins]).reshape(-1, 1)
        y_test  = np.array([w.spo2_ref for w in test_wins])

        fold_row = f"  test={test_subject:<12}  train_n={len(train_wins):>4}  test_n={len(test_wins):>3}"
        metrics  = []

        for name, template in models.items():
            fitted = clone(template)
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
            ))
            metrics.append(f"{name[:4]}={rmse:.2f}")

        print(fold_row + "  |  RMSE: " + "  ".join(metrics))

    return results


# ---------------------------------------------------------------------------
# Multi-LED combination calibration
# ---------------------------------------------------------------------------

# Fields on PPGWindow that correspond to each LED combination.
# Pairs use one feature; triples use three pairwise R values; quad uses all six.
LED_COMBOS: Dict[str, List[str]] = {
    # 6 pairs
    'Red-IR':          ['R'],
    'Red-Green':       ['R_red_green'],
    'Red-Blue':        ['R_red_blue'],
    'IR-Green':        ['R_ir_green'],
    'IR-Blue':         ['R_ir_blue'],
    'Green-Blue':      ['R_green_blue'],
    # 4 triples (all pairwise R values within each triple)
    'Red-IR-Green':    ['R', 'R_red_green', 'R_ir_green'],
    'Red-IR-Blue':     ['R', 'R_red_blue',  'R_ir_blue'],
    'Red-Green-Blue':  ['R_red_green', 'R_red_blue', 'R_green_blue'],
    'IR-Green-Blue':   ['R_ir_green',  'R_ir_blue',  'R_green_blue'],
    # 1 quadruple (all six pairwise R values)
    'All-4LEDs':       ['R', 'R_red_green', 'R_red_blue',
                        'R_ir_green', 'R_ir_blue', 'R_green_blue'],
}


def calibrate_multi(
    windows,
    fields:      List[str],
    random_seed: int  = 42,
    verbose:     bool = False,
) -> Dict[str, List[FoldResult]]:
    """
    LOSO cross-validation using an arbitrary set of R-ratio fields as features.

    `fields` is a list of PPGWindow attribute names (e.g. ['R'] or
    ['R', 'R_red_green', 'R_ir_green']).  Windows where any listed field or
    spo2_ref is NaN are silently dropped.
    """
    rng = np.random.default_rng(random_seed)

    valid = [
        w for w in windows
        if all(not np.isnan(getattr(w, f)) for f in fields)
        and not np.isnan(w.spo2_ref)
    ]
    subjects = sorted({w.subject for w in valid})
    models   = _build_models()
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
            ))

    return results


def calibrate_all_combos(
    windows,
    random_seed: int = 42,
) -> Dict[str, Dict[str, List[FoldResult]]]:
    """
    Run LOSO calibration for every entry in LED_COMBOS.

    Returns a nested dict: combo_name → model_name → list[FoldResult].
    """
    combo_results: Dict[str, Dict[str, List[FoldResult]]] = {}
    for combo, fields in LED_COMBOS.items():
        n_feat = len(fields)
        print(f"  {combo:<18}  {n_feat} feature(s): {', '.join(fields)}")
        combo_results[combo] = calibrate_multi(windows, fields,
                                               random_seed=random_seed,
                                               verbose=False)
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
            fold.rmse = float(np.sqrt(np.mean((fold.y_true - fold.y_pred) ** 2)))
            fold.mae  = float(mean_absolute_error(fold.y_true, fold.y_pred))
            fold.r2   = float(r2_score(fold.y_true, fold.y_pred))
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(results: Dict[str, List[FoldResult]]) -> None:
    """Print mean ± std of RMSE, MAE and R² across all folds per model."""
    print(f"\n  {'Model':<14} {'RMSE':>6}±{'std':>5}   {'MAE':>6}±{'std':>5}   {'R²':>6}±{'std':>5}")
    print("  " + "-" * 62)
    for name, folds in results.items():
        rmses = np.array([f.rmse for f in folds])
        maes  = np.array([f.mae  for f in folds])
        r2s   = np.array([f.r2   for f in folds])
        print(f"  {name:<14} "
              f"{rmses.mean():>6.3f}±{rmses.std():>5.3f}   "
              f"{maes.mean():>6.3f}±{maes.std():>5.3f}   "
              f"{r2s.mean():>6.3f}±{r2s.std():>5.3f}")
