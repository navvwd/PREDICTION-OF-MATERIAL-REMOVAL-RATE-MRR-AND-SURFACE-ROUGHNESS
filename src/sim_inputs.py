"""Input widgets shared by the two simulators (driven by the trained model's feature list)."""
from __future__ import annotations

import numpy as np
import streamlit as st

LABELS = {
    "vc": ("Cutting speed Vc", "m/min", 5.0, "%.1f"),
    "f": ("Feed f", "mm/rev", 0.005, "%.3f"),
    "ap": ("Depth of cut ap", "mm", 0.01, "%.2f"),
    "hardness_HV": ("Workpiece hardness", "HV", 1.0, "%.0f"),
    "cut_index": ("Cut sequence no.", "", 1.0, "%.0f"),
    "material": ("Workpiece material", "", None, None),
}


def label(feat: str) -> str:
    l, u, *_ = LABELS.get(feat, (feat, "", None, None))
    return f"{l} [{u}]" if u else l


def render_inputs(bundle: dict, key: str, df=None) -> dict:
    """Render one widget per model feature; defaults = training medians. Returns {feature: value}."""
    feats = bundle["features"]
    ranges = bundle.get("feature_ranges", {})
    cats = bundle.get("categories", {})
    vals = {}
    cols = st.columns(min(3, max(1, len(feats))))
    for i, f in enumerate(feats):
        with cols[i % len(cols)]:
            if f in bundle["categorical_features"]:
                opts = cats.get(f) or (sorted(df[f].astype(str).unique()) if df is not None and f in df else [])
                vals[f] = st.selectbox(label(f), opts, key=f"{key}_{f}")
            else:
                lo, hi, med = ranges.get(f, (0.0, 1.0, 0.5))
                _, _, step, fmt = LABELS.get(f, (f, "", None, "%.4g"))
                span = hi - lo if hi > lo else max(abs(hi), 1.0)
                step = step or float(10 ** np.floor(np.log10(span / 100))) if span > 0 else 0.01
                vals[f] = st.number_input(label(f), min_value=0.0, value=float(med), step=float(step), format=fmt,
                                          key=f"{key}_{f}", help=f"Training range: {lo:g} – {hi:g}")
                st.caption(f"trained on {lo:g} – {hi:g}")
    return vals
