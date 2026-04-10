"""
plotting.py
~~~~~~~~~~~
All matplotlib output for the SpO2 pipeline.
 
Figures produced
-----------------
01_reference_vs_prototype_signals.png  – reference vs prototype RED + IR + SpO2
02_prototype_all_wavelengths.png       – all 4 prototype channels
03_acc_motion.png                      – ACC axes, magnitude and motion energy
04_metrics_summary.png                 – bar chart: RMSE per combo/model
best_model_fold_{n}.png                – time-series prediction per fold
fold_{n}_detail.png                    – last fold detail
best_scatter.png                       – scatter: reference vs estimated SpO2
"""
 
from pathlib import Path
from typing import List, Optional
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from .models import FoldResult
from .utils import normalize_signal
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _shade_motion(ax, df: pd.DataFrame, alpha: float = 0.15) -> None:
    """Shade regions where high_motion == 1."""
    if "high_motion" not in df.columns:
        return
    t = df["time"].to_numpy()
    m = df["high_motion"].to_numpy()
    in_region = False
    t_start = 0.0
    for i, (ti, mi) in enumerate(zip(t, m)):
        if mi and not in_region:
            t_start  = ti
            in_region = True
        elif not mi and in_region:
            ax.axvspan(t_start, ti, color="red", alpha=alpha, label="_motion")
            in_region = False
    if in_region:
        ax.axvspan(t_start, t[-1], color="red", alpha=alpha)
 
 
def _best_combo_model(summary_df: pd.DataFrame) -> tuple[str, str]:
    combo_col = "combo_" if "combo_" in summary_df.columns else "combo"
    model_col = "model_" if "model_" in summary_df.columns else "model"
    rmse_col  = "RMSE_mean" if "RMSE_mean" in summary_df.columns else "rmse_mean"
    row       = summary_df.sort_values(rmse_col).iloc[0]
    return str(row[combo_col]), str(row[model_col])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Figure 01 – reference vs prototype optical + SpO2
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_reference_vs_prototype_signals(
    df: pd.DataFrame,
    out_dir: Path,
    duration_sec: float = 60.0,
) -> None:
    sel = df[df["time"] <= df["time"].min() + duration_sec].copy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
 
    for ax, ref_col, proto_col, title in (
        (axes[0], "ref_red", "proto_red", "RED"),
        (axes[1], "ref_ir",  "proto_ir",  "IR"),
    ):
        if ref_col in sel.columns:
            ax.plot(sel["time"], normalize_signal(sel[ref_col]),
                    label=f"Reference {title}", lw=1.2)
        if proto_col in sel.columns:
            ax.plot(sel["time"], normalize_signal(sel[proto_col]),
                    label=f"Prototype {title}", alpha=0.8, lw=1.0)
        _shade_motion(ax, sel)
        ax.set_ylabel("Normalised amplitude")
        ax.set_title(f"Reference vs Prototype – {title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    if "spo2" in sel.columns:
        axes[2].plot(sel["time"], sel["spo2"], label="Reference SpO2", lw=1.2)
        _shade_motion(axes[2], sel)
        axes[2].set_ylabel("SpO2 (%)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Cleaned Reference SpO2")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
 
    fig.tight_layout()
    fig.savefig(out_dir / "01_reference_vs_prototype_signals.png", dpi=200)
    plt.close(fig)
    print("  [plot] 01_reference_vs_prototype_signals.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Figure 02 – all 4 prototype wavelengths
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_all_prototype_ppg(
    df: pd.DataFrame,
    out_dir: Path,
    duration_sec: float = 60.0,
) -> None:
    sel = df[df["time"] <= df["time"].min() + duration_sec].copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"proto_red": "red", "proto_ir": "darkred",
              "proto_green": "green", "proto_blue": "blue"}
    for c, color in colors.items():
        if c in sel.columns:
            ax.plot(sel["time"], normalize_signal(sel[c]),
                    label=c, color=color, lw=1.0)
    _shade_motion(ax, sel)
    ax.set_title("Prototype – all wavelengths (shaded = high motion)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_prototype_all_wavelengths.png", dpi=200)
    plt.close(fig)
    print("  [plot] 02_prototype_all_wavelengths.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Figure 03 – ACC channels and motion energy
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_acc_motion(
    df: pd.DataFrame,
    out_dir: Path,
    duration_sec: float = 60.0,
) -> None:
    acc_cols = [c for c in ("accx", "accy", "accz") if c in df.columns]
    if not acc_cols and "acc_mag" not in df.columns:
        print("  [plot] No ACC data — skipping acc_motion plot.")
        return
 
    sel   = df[df["time"] <= df["time"].min() + duration_sec].copy()
    n_row = len(acc_cols) + (1 if "motion_energy" in sel.columns else 0) + 1  # +1 for magnitude
    fig, axes = plt.subplots(n_row, 1, figsize=(14, 3 * n_row), sharex=True)
    if n_row == 1:
        axes = [axes]
 
    idx = 0
    colors = {"accx": "steelblue", "accy": "coral", "accz": "seagreen"}
    for c in acc_cols:
        axes[idx].plot(sel["time"], sel[c], lw=0.8, color=colors.get(c, "grey"), label=c)
        axes[idx].set_ylabel("ACC (ADC)")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        idx += 1
 
    if "acc_mag" in sel.columns:
        axes[idx].plot(sel["time"], sel["acc_mag"], lw=0.9, color="purple", label="|ACC|")
        axes[idx].set_ylabel("|ACC|")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        idx += 1
 
    if "motion_energy" in sel.columns:
        axes[idx].plot(sel["time"], sel["motion_energy"], lw=0.9, color="orange",
                       label="Motion energy")
        if "high_motion" in sel.columns:
            thr = sel["motion_energy"][sel["high_motion"] == 1]
            if len(thr):
                thresh_val = thr.min()
                axes[idx].axhline(thresh_val, color="red", ls="--", lw=0.8,
                                  label=f"threshold ≈ {thresh_val:.3f}")
        axes[idx].set_ylabel("Motion energy")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlabel("Time (s)")
        idx += 1
 
    axes[0].set_title("Accelerometer channels and derived motion energy")
    fig.tight_layout()
    fig.savefig(out_dir / "03_acc_motion.png", dpi=200)
    plt.close(fig)
    print("  [plot] 03_acc_motion.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Figure 04 – metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_metrics_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    df = summary_df.copy()
    combo_col = "combo_" if "combo_" in df.columns else "combo"
    model_col = "model_" if "model_" in df.columns else "model"
    rmse_col  = "RMSE_mean"
 
    if rmse_col not in df.columns:
        print("  [plot] RMSE_mean not found — skipping metrics summary.")
        return
 
    df["label"] = df[combo_col].astype(str) + " | " + df[model_col].astype(str)
    df = df.sort_values(rmse_col)
 
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(df["label"], df[rmse_col], color="steelblue", alpha=0.8)
 
    if "RMSE_std" in df.columns:
        ax.errorbar(
            df["label"], df[rmse_col], yerr=df["RMSE_std"],
            fmt="none", ecolor="black", capsize=3, lw=1.2,
        )
 
    ax.set_ylabel("Mean RMSE (SpO₂ %)")
    ax.set_title("Model / LED combination comparison — cross-validated RMSE")
    ax.tick_params(axis="x", rotation=65, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "04_metrics_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] 04_metrics_summary.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Best-model time-series per fold
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_best_model_predictions(
    results: List[FoldResult],
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    best_combo, best_model = _best_combo_model(summary_df)
    selected = [r for r in results if r.combo == best_combo and r.model == best_model]
 
    if not selected:
        print(f"  [plot] No results for combo={best_combo}, model={best_model}.")
        return
 
    for r in selected:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(r.y_true, label="Reference SpO₂", lw=1.3)
        ax.plot(r.y_pred, label="Estimated SpO₂", lw=1.0, alpha=0.85)
 
        # shade high-motion windows if available
        if r.motion_flag is not None:
            for i, mf in enumerate(r.motion_flag):
                if mf:
                    ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15, label="_m")
 
        rmse = r.metrics.get("RMSE", float("nan"))
        ax.set_title(
            f"Best model – fold {r.fold}: {best_combo} | {best_model}  "
            f"(RMSE={rmse:.2f}%)"
        )
        ax.set_xlabel("Window index")
        ax.set_ylabel("SpO₂ (%)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"best_model_fold_{r.fold}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
 
    print(f"  [plot] best_model_fold_*.png ({len(selected)} folds)")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Last-fold detail
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_fold_detail(
    results: List[FoldResult],
    summary_df: pd.DataFrame,
    out_dir: Path,
    fold: int = 5,
) -> None:
    best_combo, best_model = _best_combo_model(summary_df)
    r = next(
        (x for x in results if x.combo == best_combo and x.model == best_model and x.fold == fold),
        None,
    )
    if r is None:
        print(f"  [plot] No result for fold={fold}, combo={best_combo}, model={best_model}.")
        return
 
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(r.y_true, label="Reference SpO₂", lw=1.3)
    ax.plot(r.y_pred, label="Estimated SpO₂", lw=1.0, alpha=0.85)
    ax.set_title(f"Fold {fold} detail: {best_combo} | {best_model}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("SpO₂ (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"fold_{fold}_detail.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] fold_{fold}_detail.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Scatter – reference vs estimated across all folds
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_scatter_best(
    results: List[FoldResult],
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    best_combo, best_model = _best_combo_model(summary_df)
    selected = [r for r in results if r.combo == best_combo and r.model == best_model]
 
    if not selected:
        print(f"  [plot] No scatter results for combo={best_combo}, model={best_model}.")
        return
 
    y_true = np.concatenate([r.y_true for r in selected])
    y_pred = np.concatenate([r.y_pred for r in selected])
    motion = (
        np.concatenate([r.motion_flag for r in selected if r.motion_flag is not None])
        if all(r.motion_flag is not None for r in selected)
        else None
    )
 
    fig, ax = plt.subplots(figsize=(6, 6))
    if motion is not None and motion.sum() > 0:
        lo = motion == 0
        hi = motion == 1
        ax.scatter(y_true[lo], y_pred[lo], alpha=0.6, s=18, label="Low motion", color="steelblue")
        ax.scatter(y_true[hi], y_pred[hi], alpha=0.6, s=18, label="High motion", color="red",
                   marker="x")
        ax.legend(fontsize=8)
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, s=18, color="steelblue")
 
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.0, label="Identity")
    ax.set_xlabel("Reference SpO₂ (%)")
    ax.set_ylabel("Estimated SpO₂ (%)")
    ax.set_title(f"Scatter (all folds): {best_combo} | {best_model}")
    fig.tight_layout()
    fig.savefig(out_dir / "best_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [plot] best_scatter.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Transfer-subject plots
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_transfer_time_series(
    time_center: "np.ndarray",
    y_true:      "np.ndarray",
    y_pred:      "np.ndarray",
    metrics:     dict,
    trained_combo: str,
    trained_model: str,
    train_subject: str,
    test_subject:  str,
    out_dir: Path,
) -> None:
    """
    Time-series of reference vs estimated SpO2 for the transfer subject,
    annotated with key error metrics.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
 
    # ── Top panel: predicted vs reference ────────────────────────────────────
    axes[0].plot(time_center, y_true, lw=1.4, label="Reference SpO₂", color="#185FA5")
    axes[0].plot(time_center, y_pred, lw=1.0, alpha=0.85,
                 label="Estimated SpO₂", color="#D85A30", linestyle="--")
    axes[0].fill_between(time_center, y_true, y_pred, alpha=0.12, color="#D85A30")
 
    rmse = metrics.get("RMSE", float("nan"))
    bias = metrics.get("Bias", float("nan"))
    r2   = metrics.get("R2",   float("nan"))
    axes[0].set_title(
        f"Transfer evaluation — trained on subject {train_subject}, "
        f"tested on subject {test_subject}"
        f"combo: {trained_combo}  |  model: {trained_model}  |  "
        f"RMSE={rmse:.2f}%  Bias={bias:+.2f}%  R²={r2:.3f}",
        fontsize=10,
    )
    axes[0].set_ylabel("SpO₂ (%)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
 
    # ── Bottom panel: residuals ───────────────────────────────────────────────
    residuals = y_pred - y_true
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1].fill_between(time_center, residuals, alpha=0.4, color="#D85A30")
    axes[1].plot(time_center, residuals, lw=0.8, color="#993C1D")
    axes[1].set_ylabel("Residual (%)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
 
    fig.tight_layout()
    fname = out_dir / f"transfer_subject{test_subject}_timeseries.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {fname.name}")
 
 
def plot_transfer_scatter(
    y_true:      "np.ndarray",
    y_pred:      "np.ndarray",
    metrics:     dict,
    trained_combo: str,
    trained_model: str,
    train_subject: str,
    test_subject:  str,
    out_dir: Path,
) -> None:
    """
    Bland-Altman + scatter panel for the transfer subject.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    # ── Left: scatter ─────────────────────────────────────────────────────────
    axes[0].scatter(y_true, y_pred, alpha=0.55, s=20, color="#185FA5", label="Windows")
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1.0, label="Identity")
    rmse = metrics.get("RMSE", float("nan"))
    r2   = metrics.get("R2",   float("nan"))
    axes[0].set_xlabel("Reference SpO₂ (%)")
    axes[0].set_ylabel("Estimated SpO₂ (%)")
    axes[0].set_title(f"Scatter  RMSE={rmse:.2f}%  R²={r2:.3f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
 
    # ── Right: Bland-Altman ────────────────────────────────────────────────────
    mean_vals = (y_true + y_pred) / 2.0
    diff_vals = y_pred - y_true
    bias      = float(np.mean(diff_vals))
    sd        = float(np.std(diff_vals, ddof=1))
    loa_hi    = bias + 1.96 * sd
    loa_lo    = bias - 1.96 * sd
 
    axes[1].scatter(mean_vals, diff_vals, alpha=0.55, s=20, color="#0F6E56")
    axes[1].axhline(bias,   color="#D85A30", lw=1.2, linestyle="-",  label=f"Bias {bias:+.2f}%")
    axes[1].axhline(loa_hi, color="#D85A30", lw=0.9, linestyle="--", label=f"+1.96 SD {loa_hi:+.2f}%")
    axes[1].axhline(loa_lo, color="#D85A30", lw=0.9, linestyle="--", label=f"−1.96 SD {loa_lo:+.2f}%")
    axes[1].axhline(0, color="black", lw=0.6, linestyle=":")
    axes[1].set_xlabel("Mean of reference and estimated SpO₂ (%)")
    axes[1].set_ylabel("Difference (estimated − reference) (%)")
    axes[1].set_title("Bland–Altman")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
 
    fig.suptitle(
        f"Transfer: trained on subject {train_subject}, "
        f"tested on subject {test_subject}  |  "
        f"{trained_combo} / {trained_model}",
        fontsize=10,
    )
    fig.tight_layout()
    fname = out_dir / f"transfer_subject{test_subject}_scatter_ba.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {fname.name}")
 
 
def plot_transfer_error_summary(
    metrics:       dict,
    trained_combo: str,
    trained_model: str,
    train_subject: str,
    test_subject:  str,
    cv_rmse:       float,
    out_dir: Path,
) -> None:
    """
    Horizontal bar chart comparing transfer-subject errors to the
    cross-validated training errors, so generalisation gap is visible.
    """
    labels  = ["RMSE (%)", "MAE (%)", "|Bias| (%)", "SD error (%)"]
    transfer = [
        metrics.get("RMSE",     float("nan")),
        metrics.get("MAE",      float("nan")),
        abs(metrics.get("Bias", float("nan"))),
        metrics.get("SD_error", float("nan")),
    ]
 
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, transfer, color="#185FA5", alpha=0.8, height=0.5)
    ax.axvline(cv_rmse, color="#D85A30", lw=1.5, linestyle="--",
               label=f"CV RMSE on training subject ({cv_rmse:.2f}%)")
 
    # Value labels on bars
    for bar, val in zip(bars, transfer):
        if np.isfinite(val):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", fontsize=9)
 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Error (%)")
    ax.set_title(
        f"Transfer errors — trained on subject {train_subject}, "
        f"tested on subject {test_subject}"
        f"combo: {trained_combo}  |  model: {trained_model}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fname = out_dir / f"transfer_subject{test_subject}_error_summary.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {fname.name}")