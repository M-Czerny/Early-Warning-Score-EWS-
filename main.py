"""
main.py
~~~~~~~
Entry point.
 
Two modes:
  1. TRAIN    -- run cross-validation on the training subject, fit the best
                 model on all data, and save the TrainedModel for reuse.
  2. TRANSFER -- load a different subject, apply the saved TrainedModel,
                 produce error metrics and plots.
 
Set TRANSFER_SUBJECT = None to run training only.
"""
 
import os
import pickle
from pathlib import Path
 
from spo2_estimation.config import FileConfig, ModelConfig, SignalConfig
from spo2_estimation.pipeline import apply_to_subject, run_pipeline
from spo2_estimation.utils import timediff
 
 
# =============================================================================
# SUBJECT SELECTION  <-- edit these two lines
# =============================================================================
TRAIN_SUBJECT    = "5"
TRANSFER_SUBJECT = "6"    # set to None to skip transfer evaluation
# =============================================================================
 
WINDOW_SEC  = 5
OVERLAP_SEC = 4
BASE_ROOT   = Path(r"D:\Early-Warning-Score-EWS-\SpO2\Pilot Data\session3")
 
REFERENCE_SIGNAL_KEYS = {
    "red":  "84:2E:14:0C:D8:EF/raw/channel_9", # Which way round?
    "ir":   "84:2E:14:0C:D8:EF/raw/channel_10",
    "spo2": "84:2E:14:0C:D8:EF/raw/channel_11",
}
PROTOTYPE_SIGNAL_KEYS = {
    "timestamp": "timestamp",
    "green": "Green", "ir": "IR", "red": "Red", "blue": "Blue",
    "accx": "ACCx", "accy": "ACCy", "accz": "ACCz",
}
LED_COMBINATIONS = [
    ("red", "ir"), ("red", "blue"), ("red", "green"),
    ("green", "ir"), ("green", "red"), ("green", "blue"),
    ("blue", "ir"), ("blue", "red"), ("blue", "green"),
    ("ir", "red"), ("ir", "green"), ("ir", "blue"),
]
 
 
def make_configs(subject: str):
    base_dir = BASE_ROOT / f"subject{subject}"
    files = sorted(
        f for f in os.listdir(base_dir) if os.path.isfile(base_dir / f)
    )
    reference_file = base_dir / files[1]
    prototype_file = base_dir / files[0]
 
    time_diff = timediff(
        base_dir / files[2], prototype_file, files[0],
        WINDOW_SEC, ref_timezone_offset_hours=2.0,
    )
 
    file_cfg = FileConfig(
        reference_path=str(reference_file),
        prototype_path=str(prototype_file),
        reference_format="h5",
        prototype_format="txt",
        reference_signal_keys=REFERENCE_SIGNAL_KEYS,
        prototype_signal_keys=PROTOTYPE_SIGNAL_KEYS,
    )
    sig_cfg = SignalConfig(
        fs_reference=100.0, fs_prototype=50.0, fs_common=50.0,
        time_diff=time_diff,
        bandpass_low_hz=0.5, bandpass_high_hz=5.0, bandpass_order=5,
        window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC,
        min_valid_spo2=92.0, max_valid_spo2=100.0, max_spo2_jump_pct=3.0,
        prototype_spike_zscore=6.0, prototype_spike_kernel=11,
        prototype_saturation_quantile=0.999,
        enable_alignment=True, max_alignment_lag_sec=10.0,
        # Motion
        acc_motion_low_hz=0.5, acc_motion_high_hz=10.0,
        motion_threshold=0.95, drop_motion_windows=True,
        enable_anc=False, anc_filter_order=32, anc_mu=0.005,
        # Inter-channel alignment
        align_channels=True, max_align_shift=8,
        # Signal quality improvements
        warmup_discard_sec=40.0,
        remove_baseline_wander=True, baseline_wander_window_sec=0.6,
        rolling_dc_window_sec=2.0,
        min_channel_sqi=0.15,
    )
    return file_cfg, sig_cfg, base_dir
 
 
def main():
 
    # ── Training subject ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  TRAINING  subject {TRAIN_SUBJECT}\n{'='*60}")
 
    train_file_cfg, train_sig_cfg, train_base = make_configs(TRAIN_SUBJECT)
    train_out = train_base / "spo2_results"
    train_out.mkdir(parents=True, exist_ok=True)
 
    model_cfg = ModelConfig(
        led_combinations=LED_COMBINATIONS, cv_splits=5, random_state=42,
    )
 
    feat_df, summary_df, trained = run_pipeline(
        file_cfg=train_file_cfg,
        sig_cfg=train_sig_cfg,
        model_cfg=model_cfg,
        out_dir=str(train_out),
    )
 
    feat_df.to_csv(train_out / "window_features_export.csv", index=False)
    summary_df.to_csv(train_out / "cv_summary_export.csv", index=False)
 
    model_path = train_out / "trained_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(trained, fh)
 
    print(f"\nTrained model saved to: {model_path}")
    print(f"  Best combo : {trained.combo}")
    print(f"  Best model : {trained.model_name}")
    print(f"  CV RMSE    : {trained.cv_rmse:.3f} %")
 
    # ── Transfer subject ──────────────────────────────────────────────────────
    if TRANSFER_SUBJECT is None:
        print("\nNo transfer subject selected. Done.")
        return
 
    print(f"\n{'='*60}\n  TRANSFER  subject {TRANSFER_SUBJECT}\n{'='*60}")
 
    # To reuse a previously saved model without re-training, comment out the
    # run_pipeline block above and uncomment the two lines below:
    #   with open(train_out / "trained_model.pkl", "rb") as fh:
    #       trained = pickle.load(fh)
 
    test_file_cfg, test_sig_cfg, test_base = make_configs(TRANSFER_SUBJECT)
    test_out = test_base / "spo2_results"
 
    time_center, y_true, y_pred, metrics = apply_to_subject(
        trained=trained,
        file_cfg=test_file_cfg,
        sig_cfg=test_sig_cfg,
        model_cfg=model_cfg,
        out_dir=str(train_out),
        train_subject=TRAIN_SUBJECT,
        test_subject=TRANSFER_SUBJECT,
    )
 
    print(f"\nDone.  {len(y_true)} windows evaluated on subject {TRANSFER_SUBJECT}.")
 
 
if __name__ == "__main__":
    main()