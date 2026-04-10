"""
config.py
~~~~~~~~
"""


from dataclasses import dataclass, field
from typing import Dict, List, Tuple
 
 
@dataclass
class FileConfig:
    reference_path: str
    prototype_path: str
    reference_format: str = 'auto'   # auto | h5 | txt
    prototype_format: str = 'auto'   # auto | csv | txt
 
    reference_signal_keys: Dict[str, str] = field(default_factory=lambda: {
        'time': 'time',
        'red': 'red',
        'ir': 'ir',
        'spo2': 'spo2',
    })
 
    prototype_signal_keys: Dict[str, str] = field(default_factory=lambda: {
        'timestamp': 'timestamp',
        'red': 'red',
        'ir': 'ir',
        'green': 'green',
        'blue': 'blue',
        'accx': 'ACCx',
        'accy': 'ACCy',
        'accz': 'ACCz',
    })
 
 
@dataclass
class SignalConfig:
    fs_reference: float = 100.0
    fs_prototype: float = 50.0
    fs_common: float = 50.0
    time_diff: float = 0.0
 
    # Bandpass for AC component
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 5.0
    bandpass_order: int = 4
 
    # Windowing
    window_sec: float = 10.0
    overlap_sec: float = 7.5
 
    # SpO2 validity
    min_valid_spo2: float = 90.0
    max_valid_spo2: float = 100.0
    max_spo2_jump_pct: float = 5.0
 
    # Prototype spike removal
    prototype_spike_zscore: float = 6.0
    prototype_spike_kernel: int = 11
    prototype_saturation_quantile: float = 0.999
 
    # Alignment
    enable_alignment: bool = True
    max_alignment_lag_sec: float = 10.0
 
    # ── Motion / ACC ──────────────────────────────────────────
    # Frequency band used to compute motion energy from ACC
    acc_motion_low_hz: float = 0.5
    acc_motion_high_hz: float = 10.0
 
    # Fraction (0–1) used as a percentile cutoff on the motion energy
    # distribution.  Windows above this percentile are flagged high-motion.
    # e.g. 0.5 = top 50% flagged, 0.75 = top 25% flagged.
    # Scale-independent — works for raw ADC counts or calibrated g values.
    motion_threshold: float = 0.97
 
    # If True, high-motion windows are dropped from model training
    # but their features are still saved to CSV.
    drop_motion_windows: bool = True
 
    # Adaptive noise cancellation: if True, use ACC as a reference
    # channel in an LMS filter to suppress motion from PPG.
    enable_anc: bool = True
    anc_filter_order: int = 32     # LMS taps
    anc_mu: float = 0.005          # LMS step-size (smaller = more stable)
 
    # ── Inter-channel alignment ───────────────────────────────────────────────
    # Correct for LED time-multiplexing offsets and wavelength-dependent
    # scattering by aligning channel pairs via cross-correlation before
    # computing the AC component of each ratio-of-ratios window.
    align_channels: bool = True
    max_align_shift: int = 8       # max shift in samples (+/-); 8 @ 50Hz = 160ms
 
    # ── Warm-up discard ───────────────────────────────────────────────────────
    # Discard the first N seconds of the merged recording.  Removes the ADC
    # startup glitch (IR dropout then overshoot) and the LED/sensor warm-up
    # period during which DC drifts rapidly.  60 s is conservative but safe.
    warmup_discard_sec: float = 60.0
 
    # ── Baseline wander removal ───────────────────────────────────────────────
    # Apply morphological baseline removal before bandpass filtering.
    # This suppresses breathing and vasomotion artefacts (0.05–0.4 Hz) that
    # dominate signal variance and cannot be fully removed by bandpass alone.
    remove_baseline_wander: bool = True
    baseline_wander_window_sec: float = 0.6   # must be > one cardiac cycle
 
    # ── Rolling DC for ratio-of-ratios ───────────────────────────────────────
    # Use a short rolling mean as the DC estimate instead of the window mean.
    # Prevents slow within-window drift from inflating or deflating R.
    rolling_dc_window_sec: float = 2.0
 
    # ── Per-channel SQI gate ──────────────────────────────────────────────────
    # Minimum spectral SNR (0-1) required for a channel to be used in a combo.
    # Green and blue typically fall below 0.15 in reflectance mode; setting
    # this to 0.15 silently skips those combos for that window rather than
    # producing a biased R value.
    min_channel_sqi: float = 0.15
 
 
@dataclass
class ModelConfig:
    led_combinations: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('red', 'ir'),
        ('green', 'ir'),
        ('blue', 'ir'),
        ('red', 'green'),
        ('red', 'blue'),
        ('green', 'blue'),
        ('ir', 'red'),
        ('ir', 'green'),
        ('ir', 'blue'),
        ('blue', 'red'),
        ('blue', 'green'),
        ('green', 'red'),
    ])
    cv_splits: int = 5
    random_state: int = 42