"""Material Removal Rate for longitudinal (external cylindrical) turning.

    MRR = 1000 * Vc * f * ap        [mm^3/min]

    Vc : cutting speed  [m/min]   (x1000 converts m -> mm)
    f  : feed           [mm/rev]
    ap : depth of cut   [mm]

This is the standard first-order turning relation (cutting speed x
uncut chip cross-section ap*f). It is valid ONLY for turning; it must not be
applied to milling, drilling, grinding or non-conventional processes.

In this project MRR is a *calculated* (derived) target: the Hannover dataset
does not contain a measured MRR. It is computed from the nominal process
parameters recorded for every cut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MRR_UNIT = "mm³/min"
MRR_EQUATION = "MRR = 1000 × Vc × f × ap"

# Plausibility limits for conventional turning (used for input validation only)
VC_RANGE = (1.0, 2000.0)   # m/min
F_RANGE = (0.005, 2.0)     # mm/rev
AP_RANGE = (0.01, 20.0)    # mm


def validate_inputs(vc: float, f: float, ap: float) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    errs = []
    for name, val, (lo, hi), unit in (
        ("Cutting speed Vc", vc, VC_RANGE, "m/min"),
        ("Feed f", f, F_RANGE, "mm/rev"),
        ("Depth of cut ap", ap, AP_RANGE, "mm"),
    ):
        if val is None or not np.isfinite(val):
            errs.append(f"{name} is not a number.")
        elif val <= 0:
            errs.append(f"{name} must be positive.")
        elif not (lo <= val <= hi):
            errs.append(f"{name} = {val} {unit} is outside the plausible turning range [{lo}, {hi}] {unit}.")
    return errs


def calculate_mrr(vc, f, ap):
    """Vectorised MRR (mm^3/min). Accepts scalars, numpy arrays or Series."""
    return 1000.0 * np.asarray(vc, dtype=float) * np.asarray(f, dtype=float) * np.asarray(ap, dtype=float)


def add_mrr_column(df: pd.DataFrame, vc_col: str, f_col: str, ap_col: str,
                   out_col: str = "MRR_mm3_min") -> pd.DataFrame:
    """Return a copy of *df* with a calculated MRR column. Raises KeyError if a column is missing."""
    missing = [c for c in (vc_col, f_col, ap_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot calculate MRR, missing column(s): {missing}")
    out = df.copy()
    out[out_col] = calculate_mrr(pd.to_numeric(out[vc_col], errors="coerce"),
                                 pd.to_numeric(out[f_col], errors="coerce"),
                                 pd.to_numeric(out[ap_col], errors="coerce"))
    return out
