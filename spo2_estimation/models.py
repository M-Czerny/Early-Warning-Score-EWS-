"""
models.py
~~~~~~~~~
Model definitions and cross-validated evaluation for SpO2 estimation.
 
Changes vs. original
---------------------
* SQI-weighted fitting: windows with higher signal quality get more
  influence during training (optional, falls back gracefully).
* Motion-aware evaluation: metrics are reported separately for
  low-motion and high-motion windows.
* Added ElasticNet and GradientBoosting regressors.
* Cleaner metric aggregation with per-fold records.
"""
 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Custom transformer
# ─────────────────────────────────────────────────────────────────────────────
 
class PolynomialFeatures2(BaseEstimator, TransformerMixin):
    """Appends X² to X (single-feature quadratic expansion)."""
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.column_stack([X, X ** 2])
 
    def fit_transform(self, X, y=None):
        return self.transform(X)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
 
def get_models() -> Dict[str, Any]:
    return {
        "linear": LinearRegression(),
        "quadratic": Pipeline([
            ("poly2", PolynomialFeatures2()),
            ("lr",    LinearRegression()),
        ]),
        "huber": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  HuberRegressor(max_iter=300)),
        ]),
        "ransac": RANSACRegressor(
            estimator=LinearRegression(), random_state=42
        ),
        "elasticnet": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  ElasticNet(max_iter=2000)),
        ]),
        "svr_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(kernel="rbf", C=10.0, epsilon=0.5)),
        ]),
        "gbr": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, random_state=42,
            )),
        ]),
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class FoldResult:
    fold:        int
    combo:       str
    model:       str
    metrics:     Dict[str, float]
    y_true:      np.ndarray
    y_pred:      np.ndarray
    time_center: np.ndarray
    motion_flag: Optional[np.ndarray] = field(default=None)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    suffix: str = "",
) -> Dict[str, float]:
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae    = float(mean_absolute_error(y_true, y_pred))
    bias   = float(np.mean(y_pred - y_true))
    sd_err = float(np.std(y_pred - y_true, ddof=1)) if len(y_true) > 1 else 0.0
    r2     = float(r2_score(y_true, y_pred))
    mape   = float(np.mean(np.abs((y_true - y_pred) /
                   np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {
        f"RMSE{suffix}":     rmse,
        f"MAE{suffix}":      mae,
        f"Bias{suffix}":     bias,
        f"SD_error{suffix}": sd_err,
        f"R2{suffix}":       r2,
        f"MAPE_pct{suffix}": mape,
    }
 
 
def _sqi_weights(features_df: pd.DataFrame, xcol: str) -> Optional[np.ndarray]:
    """
    Build per-sample weights from SQI columns when available.
    Weights are the mean SQI of the two channels in the combination.
    Falls back to None (uniform weights) if SQI columns are absent.
    """
    # xcol is like "R_red_ir"  → channels are "red" and "ir"
    parts = xcol.split("_")          # ['R', 'red', 'ir']
    if len(parts) < 3:
        return None
 
    ch_a, ch_b = parts[1], parts[2]
    col_a = f"sqi_{ch_a}"
    col_b = f"sqi_{ch_b}"
 
    has_a = col_a in features_df.columns
    has_b = col_b in features_df.columns
    if not (has_a or has_b):
        return None
 
    cols = [c for c in (col_a, col_b) if c in features_df.columns]
    sqi  = features_df[cols].mean(axis=1).to_numpy(dtype=float)
    sqi  = np.clip(sqi, 0.0, 1.0)
    # avoid zero-weight issues
    sqi  = np.maximum(sqi, 0.05)
    return sqi
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────
 
def cross_validate_feature_table(
    features_df: pd.DataFrame,
    combinations: List[Tuple[str, str]],
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, List[FoldResult]]:
    """
    Time-series cross-validation over all LED combinations and models.
 
    Returns
    -------
    summary_df   – aggregated metrics (mean ± std per combo/model)
    all_results  – list of FoldResult objects with raw predictions
    """
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    y            = features_df["spo2"].to_numpy(dtype=float)
    time_center  = features_df["time_center"].to_numpy(dtype=float)
    has_motion   = "high_motion_window" in features_df.columns
    motion_flag  = (
        features_df["high_motion_window"].to_numpy(dtype=int)
        if has_motion else np.zeros(len(features_df), dtype=int)
    )
 
    models      = get_models()
    all_results: List[FoldResult] = []
 
    for combo in combinations:
        combo_name = f"{combo[0]}_{combo[1]}"
        xcol       = f"R_{combo[0]}_{combo[1]}"
 
        if xcol not in features_df.columns:
            print(f"  [cv] Column '{xcol}' missing — skipping.")
            continue
 
        X      = features_df[[xcol]].to_numpy(dtype=float)
        w_full = _sqi_weights(features_df, xcol)
 
        for fold_idx, (tr, te) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X[tr], X[te]
            y_train, y_test = y[tr], y[te]
            t_test          = time_center[te]
            m_test          = motion_flag[te]
            w_train         = w_full[tr] if w_full is not None else None
 
            for model_name, model in models.items():
                mdl = clone(model)
 
                # SQI-weighted fit where supported
                try:
                    if w_train is not None:
                        mdl.fit(X_train, y_train, **_fit_weight_kwargs(mdl, w_train))
                    else:
                        mdl.fit(X_train, y_train)
                except TypeError:
                    mdl.fit(X_train, y_train)
 
                y_pred   = mdl.predict(X_test)
                metrics  = compute_metrics(y_test, y_pred)
 
                # Separate metrics for low/high motion on test set
                lo_idx = np.where(m_test == 0)[0]
                hi_idx = np.where(m_test == 1)[0]
                if len(lo_idx) > 1:
                    metrics.update(compute_metrics(y_test[lo_idx], y_pred[lo_idx], suffix="_lo"))
                if len(hi_idx) > 1:
                    metrics.update(compute_metrics(y_test[hi_idx], y_pred[hi_idx], suffix="_hi"))
 
                all_results.append(FoldResult(
                    fold=fold_idx,
                    combo=combo_name,
                    model=model_name,
                    metrics=metrics,
                    y_true=y_test,
                    y_pred=y_pred,
                    time_center=t_test,
                    motion_flag=m_test,
                ))
 
    # ── Aggregate metrics ────────────────────────────────────────────────────
    metrics_rows = []
    for r in all_results:
        row = {"fold": r.fold, "combo": r.combo, "model": r.model}
        row.update(r.metrics)
        metrics_rows.append(row)
 
    metrics_df = pd.DataFrame(metrics_rows)
 
    # Only aggregate columns that are numeric
    num_cols = [c for c in metrics_df.columns
                if c not in ("fold", "combo", "model")
                and pd.api.types.is_numeric_dtype(metrics_df[c])]
 
    agg_dict = {c: ["mean", "std"] for c in num_cols}
    summary  = metrics_df.groupby(["combo", "model"], as_index=False).agg(agg_dict)
    summary.columns = [
        "_".join(filter(None, col)).strip("_") for col in summary.columns
    ]
    summary = summary.sort_values("RMSE_mean").reset_index(drop=True)
 
    return summary, all_results
 
 
def _fit_weight_kwargs(estimator, sample_weight: np.ndarray) -> dict:
    """
    Return the correct keyword argument for passing sample weights,
    handling sklearn Pipelines vs. plain estimators.
    """
    import inspect
    from sklearn.pipeline import Pipeline as SKPipeline
 
    if isinstance(estimator, SKPipeline):
        last_step = estimator.steps[-1][0]
        return {f"{last_step}__sample_weight": sample_weight}
 
    sig = inspect.signature(estimator.fit)
    if "sample_weight" in sig.parameters:
        return {"sample_weight": sample_weight}
 
    return {}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Train best model on full dataset
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class TrainedModel:
    """
    A model fitted on the full training-subject feature table,
    ready to be applied to a new subject.
    """
    combo:      str            # e.g. "red_ir"
    model_name: str            # e.g. "quadratic"
    estimator:  Any            # fitted sklearn estimator
    xcol:       str            # R-value column name, e.g. "R_red_ir"
    train_rmse: float          # in-sample RMSE for reference
    cv_rmse:    float          # cross-validated RMSE from summary_df
 
 
def train_best_model(
    features_df: pd.DataFrame,
    summary_df:  pd.DataFrame,
) -> TrainedModel:
    """
    Identify the best combo/model from *summary_df* (lowest RMSE_mean),
    then fit that model on the **entire** feature table so all available
    data from the training subject informs the coefficients.
 
    Parameters
    ----------
    features_df : window feature table produced by build_window_features.
    summary_df  : cross-validation summary from cross_validate_feature_table.
 
    Returns
    -------
    TrainedModel with a fitted estimator ready for predict().
    """
    combo_col = "combo_" if "combo_" in summary_df.columns else "combo"
    model_col = "model_" if "model_" in summary_df.columns else "model"
    rmse_col  = "RMSE_mean"
 
    best = summary_df.sort_values(rmse_col).iloc[0]
    best_combo = str(best[combo_col])
    best_model = str(best[model_col])
    cv_rmse    = float(best[rmse_col])
 
    xcol = f"R_{best_combo}"
    if xcol not in features_df.columns:
        raise KeyError(
            f"Feature column '{xcol}' not found in features_df. "
            f"Available R columns: {[c for c in features_df.columns if c.startswith('R_')]}"
        )
 
    # Drop windows where this R value is NaN (SQI-gated)
    valid = features_df[[xcol, "spo2"]].dropna()
    if len(valid) < 5:
        raise RuntimeError(
            f"Only {len(valid)} valid windows for combo '{best_combo}' — "
            "cannot train a reliable model."
        )
 
    X = valid[[xcol]].to_numpy(dtype=float)
    y = valid["spo2"].to_numpy(dtype=float)
 
    # SQI weights if available
    w = _sqi_weights(valid, xcol)
 
    models   = get_models()
    estimator = clone(models[best_model])
    try:
        if w is not None:
            estimator.fit(X, y, **_fit_weight_kwargs(estimator, w))
        else:
            estimator.fit(X, y)
    except TypeError:
        estimator.fit(X, y)
 
    y_pred_train = estimator.predict(X)
    train_rmse   = float(np.sqrt(mean_squared_error(y, y_pred_train)))
 
    print(f"  [train] Best combo: {best_combo}  model: {best_model}")
    print(f"  [train] CV RMSE: {cv_rmse:.3f}%  |  In-sample RMSE: {train_rmse:.3f}%")
    print(f"  [train] Trained on {len(valid)} windows")
 
    return TrainedModel(
        combo=best_combo,
        model_name=best_model,
        estimator=estimator,
        xcol=xcol,
        train_rmse=train_rmse,
        cv_rmse=cv_rmse,
    )
 
 
def apply_trained_model(
    trained: TrainedModel,
    features_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Apply a TrainedModel to a new subject's feature table.
 
    Parameters
    ----------
    trained     : TrainedModel from train_best_model().
    features_df : feature table for the new subject.
 
    Returns
    -------
    time_center : centre time of each window (s)
    y_true      : reference SpO2 for each window
    y_pred      : estimated SpO2 for each window
    metrics     : dict with RMSE, MAE, Bias, SD_error, R2, MAPE_pct
    """
    if trained.xcol not in features_df.columns:
        raise KeyError(
            f"Feature column '{trained.xcol}' not found in new subject features. "
            f"Ensure the same LED combinations are used."
        )
 
    valid = features_df[["time_center", trained.xcol, "spo2"]].dropna()
    if len(valid) == 0:
        raise RuntimeError(
            f"No valid windows for column '{trained.xcol}' in new subject data."
        )
 
    X            = valid[[trained.xcol]].to_numpy(dtype=float)
    y_true       = valid["spo2"].to_numpy(dtype=float)
    time_center  = valid["time_center"].to_numpy(dtype=float)
 
    y_pred  = trained.estimator.predict(X)
    metrics = compute_metrics(y_true, y_pred)
 
    print(f"  [transfer] Subject windows: {len(valid)}")
    print(f"  [transfer] RMSE : {metrics['RMSE']:.3f} %")
    print(f"  [transfer] MAE  : {metrics['MAE']:.3f} %")
    print(f"  [transfer] Bias : {metrics['Bias']:+.3f} %")
    print(f"  [transfer] R²   : {metrics['R2']:.3f}")
 
    return time_center, y_true, y_pred, metrics