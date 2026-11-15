"""Dataset loading, metadata, validation and column mapping."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .mrr_calculator import add_mrr_column

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "dataset.csv"

DATASET_DOI = "10.25835/9xffqwnc"
DATASET_URL = "https://doi.org/10.25835/9xffqwnc"
DATASET_TITLE = ("Multivariate time series dataset for predicting surface roughness and dimensional "
                 "error in longitudinal turning")
DATASET_CITATION = ("Denkena, B., Bergmann, B., Buhl, H., Handrup, M. (2026). " + DATASET_TITLE +
                    " [Data set]. LUIS, Leibniz University Hannover (SFB 1153). "
                    "https://doi.org/10.25835/9xffqwnc. Licence: CC BY 3.0.")

REFERENCES = [
    DATASET_CITATION,
    "Benardos, P.G., Vosniakos, G.-C. (2003). Predicting surface roughness in machining: a review. "
    "International Journal of Machine Tools and Manufacture, 43(8), 833–844.",
    "Özel, T., Karpat, Y. (2005). Predictive modeling of surface roughness and tool wear in hard turning using "
    "regression and neural networks. International Journal of Machine Tools and Manufacture, 45(4–5), 467–479.",
    "Kalpakjian, S., Schmid, S.R. (2014). Manufacturing Engineering and Technology, 7th ed. Pearson.",
    "Shaw, M.C. (2005). Metal Cutting Principles, 2nd ed. Oxford University Press.",
    "ISO 4287:1997. Geometrical Product Specifications (GPS) — Surface texture: Profile method — Terms, "
    "definitions and surface texture parameters.",
    "Breiman, L., Friedman, J.H., Olshen, R.A., Stone, C.J. (1984). Classification and Regression Trees. Wadsworth.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
    "Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.",
    "Drucker, H., Burges, C.J.C., Kaufman, L., Smola, A., Vapnik, V. (1997). Support vector regression machines. "
    "Advances in Neural Information Processing Systems 9, 155–161.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95.",
    "Waskom, M.L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.",
]

# Column dictionary for the Hannover dataset (verified against the HDF5 attributes)
COLUMN_INFO = {
    "sample_id":   ("Workpiece ID", "—", "Identifier of the turned bar (one per material)"),
    "material":    ("Workpiece material", "—", "20MnCr5 (case-hardening steel), 41Cr4 (QT steel), EN AW-6082 (Al alloy)"),
    "hardness_HV": ("Workpiece hardness", "HV", "Vickers hardness stored per workpiece"),
    "tool":        ("Insert", "—", "ISO insert designation"),
    "cut_index":   ("Cut sequence no.", "—", "Order of the cut on the workpiece (derived from timestamp)"),
    "Measurement_ID": ("Measurement ID", "—", "ID from the original database"),
    "vc":          ("Cutting speed Vc", "m/min", "Nominal process parameter"),
    "f":           ("Feed f", "mm/rev", "Nominal process parameter"),
    "ap":          ("Depth of cut ap", "mm", "Nominal; verified: d0_nominal − d1_nominal = 2·ap"),
    "re":          ("Corner radius rε", "mm", "Tool nose radius (constant 0.8 mm)"),
    "Ra":          ("Surface roughness Ra", "µm", "MEASURED target — arithmetic mean roughness"),
    "Rz":          ("Surface roughness Rz", "µm", "Measured mean roughness depth"),
    "d0":          ("Measured initial diameter", "mm", "Measured"),
    "d1":          ("Measured final diameter", "mm", "Measured"),
    "d0_nominal":  ("Nominal initial diameter", "mm", ""),
    "d1_nominal":  ("Nominal final diameter", "mm", ""),
    "d_error":     ("Diameter error", "not documented", "Stored as in source; not used"),
    "T_error":     ("T_error", "not documented", "Stored as in source; not used"),
    "T_length":    ("T_length", "not documented", "Stored as in source; not used"),
    "num_segments": ("Segments", "—", "Number of signal segments"),
    "MRR_mm3_min": ("Material removal rate", "mm³/min", "CALCULATED target: 1000·Vc·f·ap"),
}
SIGNAL_NOTE = ("Columns ending in _mean/_std (FX, FY, FZ, I4: NC drive signals) and _rms (aX, aY, aZ: acceleration) "
               "are per-cut statistics computed by this project from the raw time series. The dataset does not document "
               "their physical units, so they are shown as recorded and are not used by the parameter-based simulators.")

# Aliases for automatic column recognition in uploaded CSV files
ALIASES = {
    "vc": ["vc", "v_c", "cutting_speed", "cutting speed", "speed", "v", "cuttingspeed", "vc_m_min"],
    "f": ["f", "feed", "feed_rate", "feed rate", "fz", "feedrate", "f_mm_rev"],
    "ap": ["ap", "a_p", "doc", "depth_of_cut", "depth of cut", "depth", "depthofcut", "ap_mm"],
    "Ra": ["ra", "r_a", "surface_roughness", "surface roughness", "roughness", "ra_um"],
}


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def guess_column(df: pd.DataFrame, key: str) -> str | None:
    cols = {_norm(c): c for c in df.columns}
    for a in ALIASES.get(key, [key]):
        if _norm(a) in cols:
            return cols[_norm(a)]
    return None


def load_default() -> pd.DataFrame | None:
    if not DEFAULT_DATASET.exists():
        return None
    return pd.read_csv(DEFAULT_DATASET)


def read_uploaded_csv(file) -> pd.DataFrame:
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if not raw:
        raise ValueError("The uploaded file is empty.")
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep)
        except Exception:
            continue
        if df.shape[1] > 1:
            break
    else:
        raise ValueError("Could not parse the file as CSV (tried comma, semicolon and tab separators).")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("The CSV has a header but no data rows.")
    return df


def standardise(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    """Rename mapped columns to vc/f/ap/Ra, coerce to numeric and add calculated MRR.

    mapping: {'vc': colname|None, 'f': ..., 'ap': ..., 'Ra': ...}. Returns (df, warnings).
    """
    warns = []
    out = df.copy()
    rename = {}
    for k, col in mapping.items():
        if col and col in out.columns and col != k:
            if k in out.columns:
                out = out.drop(columns=[k])
            rename[col] = k
    out = out.rename(columns=rename)
    for k in ("vc", "f", "ap", "Ra"):
        if k in out.columns:
            before = out[k].isna().sum()
            out[k] = pd.to_numeric(out[k], errors="coerce")
            bad = int(out[k].isna().sum() - before)
            if bad:
                warns.append(f"{bad} non-numeric value(s) in '{k}' were treated as missing.")
            if (out[k] <= 0).any():
                warns.append(f"'{k}' contains zero/negative values; check units.")
    if all(k in out.columns for k in ("vc", "f", "ap")):
        out = add_mrr_column(out, "vc", "f", "ap")
        if out["vc"].median() < 10:
            warns.append("Median cutting speed < 10 — is Vc in m/s instead of m/min? MRR assumes m/min.")
        if out["f"].median() > 2:
            warns.append("Median feed > 2 — is feed in mm/min instead of mm/rev? MRR assumes mm/rev.")
    else:
        warns.append("Vc, f and ap are not all mapped, so MRR cannot be calculated.")
    if "Ra" not in out.columns:
        warns.append("No surface-roughness column mapped; Ra models cannot be trained.")
    return out, warns


def quality_report(df: pd.DataFrame) -> dict:
    miss = df.isna().sum()
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing_total": int(miss.sum()),
        "missing": pd.DataFrame({"Missing": miss, "Missing %": (miss / max(len(df), 1) * 100).round(2)})
                     .query("Missing > 0"),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(exclude="number").columns.tolist(),
    }


def column_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        info = COLUMN_INFO.get(c)
        s = df[c]
        rows.append({
            "Column": c,
            "Meaning": info[0] if info else ("Signal statistic" if c.endswith(("_mean", "_std", "_rms")) else "—"),
            "Unit": info[1] if info else ("as recorded" if c.endswith(("_mean", "_std", "_rms")) else "—"),
            "Type": str(s.dtype),
            "Unique": int(s.nunique()),
            "Min": float(s.min()) if pd.api.types.is_numeric_dtype(s) else None,
            "Max": float(s.max()) if pd.api.types.is_numeric_dtype(s) else None,
            "Note": info[2] if info else "",
        })
    return pd.DataFrame(rows)


def is_hannover(df: pd.DataFrame) -> bool:
    return {"sample_id", "material", "vc", "f", "ap", "Ra", "Rz"}.issubset(df.columns)


def useful_features(df: pd.DataFrame, exclude: tuple = ()) -> list[str]:
    """Columns that may serve as model inputs (excludes identifiers, targets and constants)."""
    banned = {"sample_id", "Measurement_ID", "scope_name", "tool", "Ra", "Rz", "MRR_mm3_min", "d0", "d1",
              "d_error", "T_error", "T_length", "num_segments", *exclude}
    out = []
    for c in df.columns:
        if c in banned or df[c].nunique(dropna=True) <= 1:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 20:
            continue
        out.append(c)
    return out
