"""
pipeline.py
~~~~~~~~~~~
Top-level pipeline orchestration.
"""
 
from pathlib import Path
from typing import Dict, Tuple
 
import numpy as np
import pandas as pd
 
from .config import FileConfig, ModelConfig, SignalConfig
from .features import build_window_features
from .io import load_data
from .models import (
    TrainedModel,
    apply_trained_model,
    cross_validate_feature_table,
    train_best_model,
)
from .plotting import (
    plot_acc_motion,
    plot_all_prototype_ppg,
    plot_best_model_predictions,
    plot_fold_detail,
    plot_metrics_summary,
    plot_reference_vs_prototype_signals,
    plot_scatter_best,
    plot_transfer_error_summary,
    plot_transfer_scatter,
    plot_transfer_time_series,
)
from .preprocessing import preprocess_and_merge
 
 
def run_pipeline(
    file_cfg: FileConfig,
    sig_cfg: SignalConfig,
    model_cfg: ModelConfig,
    out_dir: str = "spo2_results",
) -> Tuple[pd.DataFrame, pd.DataFrame, TrainedModel]:
    """
    End-to-end pipeline:
 
    1. Load raw data.
    2. Preprocess + merge onto a common time grid (includes ANC + alignment).
    3. Save diagnostic plots.
    4. Extract windowed features (motion-gated).
    5. Cross-validate all model/combo pairs.
    6. Save plots + CSV artefacts.
 
    Returns
    -------
    feat_df     – per-window feature table (motion-filtered)
    summary_df  – cross-validated metric summary
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
 
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[pipeline] Loading data …")
    ref_df, proto_df = load_data(file_cfg)
    print(f"  Reference : {len(ref_df):,} samples")
    print(f"  Prototype : {len(proto_df):,} samples")
 
    # ── 2. Preprocess ────────────────────────────────────────────────────────
    print("\n[pipeline] Preprocessing …")
    merged = preprocess_and_merge(ref_df, proto_df, sig_cfg)
    merged.to_csv(out_path / "merged_signals.csv", index=False)
    print(f"  Merged grid: {len(merged):,} samples")
 
    # ── 3. Diagnostic plots ──────────────────────────────────────────────────
    print("\n[pipeline] Generating diagnostic plots …")
    merged_plot = merged.rename(columns={
        "red":   "proto_red",
        "ir":    "proto_ir",
        "green": "proto_green",
        "blue":  "proto_blue",
    })
    plot_reference_vs_prototype_signals(merged_plot, out_path)
    plot_all_prototype_ppg(merged_plot, out_path)
    plot_acc_motion(merged, out_path)           # uses original column names
 
    # ── 4. Feature extraction ────────────────────────────────────────────────
    print("\n[pipeline] Extracting windowed features …")
    feat = build_window_features(
        merged,
        combinations=model_cfg.led_combinations,
        fs=sig_cfg.fs_common,
        window_sec=sig_cfg.window_sec,
        overlap_sec=sig_cfg.overlap_sec,
        motion_threshold=sig_cfg.motion_threshold,
        drop_motion_windows=sig_cfg.drop_motion_windows,
        align_channels=sig_cfg.align_channels,
        max_align_shift=sig_cfg.max_align_shift,
        min_channel_sqi=sig_cfg.min_channel_sqi,
    )
    feat.to_csv(out_path / "window_features.csv", index=False)
    print(f"  Feature windows: {len(feat):,}")
 
    if len(feat) < model_cfg.cv_splits * 2:
        raise RuntimeError(
            f"Too few windows ({len(feat)}) for {model_cfg.cv_splits}-fold CV. "
            "Reduce window_sec, overlap_sec, or cv_splits."
        )
 
    # ── 5. Cross-validation ──────────────────────────────────────────────────
    print("\n[pipeline] Running cross-validation …")
    summary_df, fold_results = cross_validate_feature_table(
        feat,
        combinations=model_cfg.led_combinations,
        n_splits=model_cfg.cv_splits,
    )
    summary_df.to_csv(out_path / "cv_summary.csv", index=False)
 
    # ── 6. Result plots ──────────────────────────────────────────────────────
    print("\n[pipeline] Saving result plots …")
    plot_metrics_summary(summary_df, out_path)
    plot_best_model_predictions(fold_results, summary_df, out_path)
    plot_fold_detail(fold_results, summary_df, out_path, fold=model_cfg.cv_splits)
    plot_scatter_best(fold_results, summary_df, out_path)
 
    # -- 7. Train best model on full dataset
    print("\n[pipeline] Training best model on full dataset ...")
    trained = train_best_model(feat, summary_df)
 
    print("\n[pipeline] Top model-combination results:")
    print(summary_df.head(10).to_string(index=False))
 
    return feat, summary_df, trained
 
 
def apply_to_subject(
    trained:       TrainedModel,
    file_cfg:      FileConfig,
    sig_cfg:       SignalConfig,
    model_cfg:     ModelConfig,
    out_dir:       str,
    train_subject: str = "train",
    test_subject:  str = "test",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Apply a pre-trained model to a new subject and return errors + plots.
 
    The new subject data is loaded, preprocessed, and windowed using the
    same configuration as the training subject.  The trained model predicts
    SpO2 from the R-value of the best LED combination, and errors are
    computed against the reference SpO2.
 
    Parameters
    ----------
    trained        : TrainedModel returned by run_pipeline().
    file_cfg       : FileConfig pointing to the new subject's files.
    sig_cfg        : SignalConfig (same settings as training).
    model_cfg      : ModelConfig (led_combinations must include trained.combo).
    out_dir        : directory for output plots and CSV.
    train_subject  : label used in plot titles.
    test_subject   : label used in plot titles.
 
    Returns
    -------
    time_center, y_true, y_pred, metrics
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
 
    # -- 1. Load ---------------------------------------------------------------
    print(f"\n[apply] Loading subject {test_subject} ...")
    ref_df, proto_df = load_data(file_cfg)
    print(f"  Reference : {len(ref_df):,} samples")
    print(f"  Prototype : {len(proto_df):,} samples")
 
    # -- 2. Preprocess ---------------------------------------------------------
    print(f"\n[apply] Preprocessing subject {test_subject} ...")
    merged = preprocess_and_merge(ref_df, proto_df, sig_cfg)
    merged.to_csv(out_path / f"subject{test_subject}_merged_signals.csv", index=False)
 
    # -- 3. Diagnostic plots ---------------------------------------------------
    merged_plot = merged.rename(columns={
        "red": "proto_red", "ir": "proto_ir",
        "green": "proto_green", "blue": "proto_blue",
    })
    plot_reference_vs_prototype_signals(merged_plot, out_path)
    plot_all_prototype_ppg(merged_plot, out_path)
    plot_acc_motion(merged, out_path)
 
    # -- 4. Features -----------------------------------------------------------
    print(f"\n[apply] Extracting features for subject {test_subject} ...")
    feat = build_window_features(
        merged,
        combinations=model_cfg.led_combinations,
        fs=sig_cfg.fs_common,
        window_sec=sig_cfg.window_sec,
        overlap_sec=sig_cfg.overlap_sec,
        motion_threshold=sig_cfg.motion_threshold,
        drop_motion_windows=sig_cfg.drop_motion_windows,
        align_channels=sig_cfg.align_channels,
        max_align_shift=sig_cfg.max_align_shift,
        min_channel_sqi=sig_cfg.min_channel_sqi,
    )
    feat.to_csv(out_path / f"subject{test_subject}_features.csv", index=False)
    print(f"  Feature windows: {len(feat):,}")
 
    # -- 5. Apply model --------------------------------------------------------
    print(f"\n[apply] Applying trained model to subject {test_subject} ...")
    time_center, y_true, y_pred, metrics = apply_trained_model(trained, feat)
 
    pred_df = pd.DataFrame({
        "time_center":    time_center,
        "spo2_reference": y_true,
        "spo2_estimated": y_pred,
        "residual":       y_pred - y_true,
    })
    pred_df.to_csv(out_path / f"subject{test_subject}_predictions.csv", index=False)
 
    # -- 6. Transfer plots -----------------------------------------------------
    print(f"\n[apply] Saving transfer plots ...")
    plot_transfer_time_series(
        time_center, y_true, y_pred, metrics,
        trained.combo, trained.model_name,
        train_subject, test_subject, out_path,
    )
    plot_transfer_scatter(
        y_true, y_pred, metrics,
        trained.combo, trained.model_name,
        train_subject, test_subject, out_path,
    )
    plot_transfer_error_summary(
        metrics, trained.combo, trained.model_name,
        train_subject, test_subject,
        cv_rmse=trained.cv_rmse,
        out_dir=out_path,
    )
 
    # -- 7. Summary ------------------------------------------------------------
    print(f"\n[apply] Transfer results -- subject {test_subject}:")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:+.3f}")
 
    return time_center, y_true, y_pred, metrics
 