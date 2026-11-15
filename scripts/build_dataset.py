"""Build data/dataset.csv from the published Hannover HDF5 database.

Source: Denkena, Bergmann, Buhl, Handrup (2026). "Multivariate time series dataset
for predicting surface roughness and dimensional error in longitudinal turning".
LUIS, Leibniz University Hannover. https://doi.org/10.25835/9xffqwnc  (CC BY 3.0)

HDF5 layout (verified by inspection):
    /<SampleId>                      group per workpiece; attrs: SampleMaterial, HV, Tool, Description, ...
    /<SampleId>/Scope_<timestamp>    group per cut; attrs: vc, f, ap, re, Ra, Rz, d0, d1, d0_nominal,
                                     d1_nominal, d_error, T_error, T_length, Measurement_ID, num_segments
    /<SampleId>/Scope_<timestamp>/<signal>   1-D time series (FX, FY, FZ, IX, IZ, I4, MX, M4, PosX, PosZ,
                                     VelX, VelZ, eX, eZ, n4 @ 1 kHz from the NC; aX, aY, aZ @ ~20 kHz)

One row per cut is written. All scalar attributes are copied unchanged. In addition,
simple per-cut statistics (mean / standard deviation / RMS) of selected time series
are computed as optional "process-monitoring" features. MRR is NOT written here; it
is calculated in the application (src/mrr_calculator.py) and labelled as calculated.

Usage:
    python scripts/build_dataset.py path/to/database.h5 [--download]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://data.uni-hannover.de/dataset/2b6899b5-3ff7-4a52-8f47-2dd785682312/resource/"
       "40d471dd-1f81-4bca-a332-9d6fd88be31e/download/database.h5")

NC_SIGNALS_MEAN_STD = ["FX", "FY", "FZ", "I4"]  # NC drive signals (units as recorded by the control, not documented)
ACC_SIGNALS_RMS = ["aX", "aY", "aZ"]            # accelerometer signals


def download(dest: Path) -> Path:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dest


def build(h5_path: Path) -> pd.DataFrame:
    import h5py
    rows = []
    with h5py.File(h5_path, "r") as f:
        for sample_id in f:
            wp = f[sample_id]
            wp_attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in wp.attrs.items()}
            for cut_no, cut_name in enumerate(sorted(wp.keys()), start=1):
                cut = wp[cut_name]
                row = {
                    "sample_id": sample_id,
                    "material": wp_attrs.get("SampleMaterial"),
                    "hardness_HV": float(wp_attrs.get("HV", np.nan)),
                    "tool": wp_attrs.get("Tool"),
                    "cut_index": cut_no,
                    "scope_name": cut_name,
                }
                for k, v in cut.attrs.items():
                    row[k] = v.item() if hasattr(v, "item") else v
                for s in NC_SIGNALS_MEAN_STD:
                    if s in cut:
                        x = cut[s][()].astype(float)
                        row[f"{s}_mean"] = float(np.nanmean(x))
                        row[f"{s}_std"] = float(np.nanstd(x))
                for s in ACC_SIGNALS_RMS:
                    if s in cut:
                        x = cut[s][()].astype(float)
                        row[f"{s}_rms"] = float(np.sqrt(np.nanmean((x - np.nanmean(x)) ** 2)))
                rows.append(row)
    df = pd.DataFrame(rows)
    # Process parameters are stored as float32 in the HDF5 file (e.g. 0.52 -> 0.5199999809).
    # Round them to the resolution at which they were set; measured values are left untouched.
    for c in ("vc", "f", "ap", "re", "d0_nominal", "d1_nominal"):
        if c in df.columns:
            df[c] = df[c].astype(float).round(3)
    front = ["sample_id", "material", "hardness_HV", "tool", "cut_index", "Measurement_ID", "vc", "f", "ap", "re", "Ra", "Rz"]
    df = df[[c for c in front if c in df.columns] + [c for c in df.columns if c not in front]]
    return df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("h5", nargs="?", default=str(ROOT / "data" / "raw" / "database.h5"))
    p.add_argument("--download", action="store_true", help="download the HDF5 file (~982 MB) first")
    p.add_argument("--out", default=str(ROOT / "data" / "dataset.csv"))
    a = p.parse_args(argv)
    h5 = Path(a.h5)
    if a.download and not h5.exists():
        print(f"Downloading {URL} -> {h5}")
        download(h5)
    if not h5.exists():
        sys.exit(f"HDF5 file not found: {h5}. Use --download or pass the path.")
    df = build(h5)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns -> {a.out}")


if __name__ == "__main__":
    main()
