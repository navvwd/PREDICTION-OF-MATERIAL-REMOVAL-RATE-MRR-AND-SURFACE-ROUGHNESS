"""Preparation of the analysis table and default feature sets per target.

The actual numeric preprocessing (imputation, scaling, one-hot encoding) lives
inside the scikit-learn pipelines in ``model_training.py`` so that it is fitted
on training folds only (no data leakage). This module prepares the table.
"""
from __future__ import annotations

import pandas as pd

from .data_loader import guess_column, standardise

TARGETS = {
    "MRR": {"column": "MRR_mm3_min", "unit": "mm³/min", "label": "Material removal rate (calculated)",
            "kind": "calculated"},
    "Ra": {"column": "Ra", "unit": "µm", "label": "Surface roughness Ra (measured)", "kind": "measured"},
}

# Process parameters used by the simulators. Material is included for Ra when available,
# because the dataset contains three different workpiece materials.
DEFAULT_FEATURES = {
    "MRR": ["vc", "f", "ap"],
    "Ra": ["vc", "f", "ap", "material"],
}


def prepare(df: pd.DataFrame, mapping: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Map columns (auto-detect if *mapping* is None) and add calculated MRR."""
    if mapping is None:
        mapping = {k: guess_column(df, k) for k in ("vc", "f", "ap", "Ra")}
    return standardise(df, mapping)


def default_features(df: pd.DataFrame, target_kind: str) -> list[str]:
    return [c for c in DEFAULT_FEATURES[target_kind] if c in df.columns and df[c].nunique() > 1]
