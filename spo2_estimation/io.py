"""
io.py
~~~~~~~~
"""


from pathlib import Path
 
import numpy as np
import pandas as pd
import h5py
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def infer_file_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".h5", ".hdf5"):
        return "h5"
    if ext == ".csv":
        return "csv"
    if ext in (".txt", ".tsv"):
        return "txt"
    return ext.lstrip(".")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Reference loader  (H5)
# ─────────────────────────────────────────────────────────────────────────────
 
def load_reference_h5(path: str, mapping: dict, fs_reference: float = 100.0) -> pd.DataFrame:
    """
    Load reference red / ir / spo2 from H5 file.
 
    mapping example::
 
        {
            "red":  "84:2E:14:0C:D8:EF/raw/channel_9",
            "ir":   "84:2E:14:0C:D8:EF/raw/channel_10",
            "spo2": "84:2E:14:0C:D8:EF/raw/channel_11",
        }
    """
    with h5py.File(path, "r") as f:
        print("\nAvailable H5 datasets:")
 
        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                #print(" ", name)
                name
 
        f.visititems(_visitor)
 
        missing = [k for k in ("red", "ir", "spo2") if mapping.get(k) not in f]
        if missing:
            raise KeyError(
                "The following mapped dataset paths were not found in the H5 file:\n"
                + "\n".join(f"  {k} -> {mapping[k]}" for k in missing)
            )
 
        red  = np.asarray(f[mapping["red"]][:].squeeze(),  dtype=float)
        ir   = np.asarray(f[mapping["ir"]][:].squeeze(),   dtype=float)
        spo2 = np.asarray(f[mapping["spo2"]][:].squeeze(), dtype=float)
 
    n    = min(len(red), len(ir), len(spo2))
    time = np.arange(n, dtype=float) / fs_reference
 
    return pd.DataFrame({"time": time, "red": red[:n], "ir": ir[:n], "spo2": spo2[:n]})
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Prototype loader  (TXT / CSV)
# ─────────────────────────────────────────────────────────────────────────────
 
def load_prototype_table(
    path: str,
    mapping: dict,
    fs_prototype: float = 50.0,
    sep: str | None = None,
) -> pd.DataFrame:
    """
    Load prototype TXT/CSV table.
 
    Optical channels (red, ir, green, blue) and ACC channels
    (accx, accy, accz) are extracted when present in *mapping*.
    A relative time axis (seconds, starting at 0) is produced from
    either the 'timestamp' column or ``fs_prototype``.
    """
    path = str(path)
 
    if sep is None:
        if Path(path).suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
    else:
        df = pd.read_csv(path, sep=sep, engine="python")
 
    print("\nPrototype columns:", df.columns.tolist())
 
    out = pd.DataFrame()
 
    # ── optical channels ─────────────────────────────────────────────────────
    for key in ("red", "ir", "green", "blue"):
        if key in mapping:
            col = mapping[key]
            if col not in df.columns:
                raise KeyError(f"Prototype column '{col}' not found (mapped from '{key}').")
            out[key] = pd.to_numeric(df[col], errors="coerce")
 
    # ── accelerometer channels ───────────────────────────────────────────────
    for key in ("accx", "accy", "accz"):
        if key in mapping:
            col = mapping[key]
            if col in df.columns:
                out[key] = pd.to_numeric(df[col], errors="coerce")
            else:
                print(f"  [warn] ACC column '{col}' not found — skipping '{key}'.")
 
    # ── time axis ────────────────────────────────────────────────────────────
    if "timestamp" in mapping and mapping["timestamp"] in df.columns:
        t = pd.to_numeric(df[mapping["timestamp"]], errors="coerce").to_numpy(dtype=float)
        # ms → s heuristic
        if np.nanmedian(np.abs(t)) > 1e6:
            t = t / 1000.0
        t = t - np.nanmin(t)
        out["time"] = t
    else:
        out["time"] = np.arange(len(df), dtype=float) / fs_prototype
 
    # reorder so time is first
    optical = [c for c in ("red", "ir", "green", "blue") if c in out.columns]
    acc     = [c for c in ("accx", "accy", "accz")       if c in out.columns]
    cols    = ["time"] + optical + acc
    return out[cols].copy()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def load_data(file_cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ref_df, proto_df)."""
    ref_fmt = (
        infer_file_format(file_cfg.reference_path)
        if file_cfg.reference_format == "auto"
        else file_cfg.reference_format
    )
    proto_fmt = (
        infer_file_format(file_cfg.prototype_path)
        if file_cfg.prototype_format == "auto"
        else file_cfg.prototype_format
    )
 
    if ref_fmt == "h5":
        ref_df = load_reference_h5(
            file_cfg.reference_path,
            file_cfg.reference_signal_keys,
            fs_reference=getattr(file_cfg, "fs_reference", 100.0),
        )
    else:
        raise ValueError(f"Unsupported reference format: '{ref_fmt}'. Expected 'h5'.")
 
    if proto_fmt in ("txt", "csv"):
        proto_df = load_prototype_table(
            file_cfg.prototype_path,
            file_cfg.prototype_signal_keys,
            fs_prototype=getattr(file_cfg, "fs_prototype", 50.0),
        )
    else:
        raise ValueError(f"Unsupported prototype format: '{proto_fmt}'.")
 
    return ref_df, proto_df