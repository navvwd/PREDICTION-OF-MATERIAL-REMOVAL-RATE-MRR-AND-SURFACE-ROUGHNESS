import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.data_loader import (DATASET_CITATION, SIGNAL_NOTE, column_table, guess_column, load_default, quality_report,
                             read_uploaded_csv)
from src.insights import design_summary
from src.preprocessing import prepare

ui.header("Page 2 · Data", "Dataset Explorer",
          "Inspect the experimental data: structure, quality, statistics, correlations and distributions. "
          "Switch to your own CSV file if required.")

ss = st.session_state

# ------------------------------------------------------------------ source selection
with st.expander("Data source", expanded=ss.get("df") is None):
    src = st.radio("Source", ["Published Hannover dataset (bundled)", "Upload a CSV file"], horizontal=True,
                   index=0 if ss.get("is_default", True) else 1)
    if src.startswith("Published"):
        st.caption(DATASET_CITATION)
        if not ss.get("is_default"):
            if st.button("Load published dataset", type="primary"):
                raw = load_default()
                if raw is None:
                    st.error("data/dataset.csv not found. Rebuild it with `python scripts/build_dataset.py --download`.")
                else:
                    df, warns = prepare(raw)
                    ss.df, ss.dataset_warnings, ss.is_default = df, warns, True
                    ss.dataset_name = "Hannover longitudinal turning dataset (DOI 10.25835/9xffqwnc)"
                    ss.bundles, ss.runs = {}, {}
                    from src.prediction import load_bundle
                    for kind in ("MRR", "Ra"):
                        b = load_bundle(kind)
                        if b is not None and "Hannover" in str(b.get("dataset", "")):
                            ss.bundles[kind], ss.runs[kind] = b, b.get("run")
                    st.rerun()
        else:
            st.success("The published dataset is loaded.")
    else:
        up = st.file_uploader("CSV file (comma, semicolon or tab separated)", type=["csv", "txt"])
        if up is not None:
            try:
                raw = read_uploaded_csv(up)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                raw = None
            if raw is not None:
                st.write(f"Detected **{raw.shape[0]} rows × {raw.shape[1]} columns**. Map the columns "
                         "(units must be Vc in m/min, f in mm/rev, ap in mm, Ra in µm):")
                opts = ["— not available —"] + list(raw.columns)
                cols = st.columns(4)
                mapping = {}
                for col, (k, lbl) in zip(cols, [("vc", "Cutting speed Vc"), ("f", "Feed f"), ("ap", "Depth of cut ap"),
                                                ("Ra", "Surface roughness Ra")]):
                    g = guess_column(raw, k)
                    with col:
                        sel = st.selectbox(lbl, opts, index=opts.index(g) if g in opts else 0, key=f"map_{k}")
                    mapping[k] = None if sel.startswith("—") else sel
                chosen = [v for v in mapping.values() if v]
                if len(chosen) != len(set(chosen)):
                    st.error("The same column is mapped to more than one variable.")
                elif st.button("Use this dataset", type="primary"):
                    df, warns = prepare(raw, mapping)
                    if not any(k in df.columns for k in ("MRR_mm3_min", "Ra")):
                        st.error("Neither MRR (needs Vc, f, ap) nor Ra could be established — no model can be trained.")
                    else:
                        ss.df, ss.dataset_warnings, ss.is_default = df, warns, False
                        ss.dataset_name = f"Uploaded file: {up.name}"
                        ss.bundles, ss.runs = {}, {}  # models from another dataset no longer apply
                        st.rerun()

if ui.need_data():
    st.stop()

df: pd.DataFrame = ss.df
for w in ss.get("dataset_warnings", []):
    st.warning(w)

q = quality_report(df)
c = st.columns(5)
with c[0]:
    ui.kpi("Rows", f"{q['rows']}", ss.get("dataset_name", ""))
with c[1]:
    ui.kpi("Columns", f"{q['cols']}", f"{len(q['numeric_cols'])} numeric · {len(q['categorical_cols'])} text")
with c[2]:
    ui.kpi("Missing values", f"{q['missing_total']}", "cells")
with c[3]:
    ui.kpi("Duplicate rows", f"{q['duplicates']}", "exact duplicates")
with c[4]:
    ui.kpi("Targets", " · ".join([t for t, col in (("MRR", "MRR_mm3_min"), ("Ra", "Ra")) if col in df.columns]) or "none",
           "MRR calculated · Ra measured")

if {"vc", "f", "ap"}.issubset(df.columns):
    st.caption("Experimental design: " + design_summary(df))

tabs = st.tabs(["Preview", "Columns", "Data quality", "Statistics", "Correlation", "Distributions", "Design space"])

with tabs[0]:
    n = st.slider("Rows to show", 5, min(500, len(df)), min(25, len(df))) if len(df) > 5 else len(df)
    st.dataframe(df.head(n), width="stretch", hide_index=True)
    st.download_button("Download full table (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "machining_dataset_with_MRR.csv", "text/csv")

with tabs[1]:
    st.dataframe(column_table(df), width="stretch", hide_index=True)
    if any(c.endswith(("_mean", "_std", "_rms")) for c in df.columns):
        st.caption(SIGNAL_NOTE)

with tabs[2]:
    l, r = st.columns(2)
    with l:
        st.markdown("**Missing values per column**")
        if q["missing"].empty:
            st.success("No missing values in any column.")
        else:
            st.dataframe(q["missing"], width="stretch")
            st.caption("Rows with a missing target are dropped only for that target; missing inputs are median-imputed "
                       "inside the training pipeline.")
    with r:
        st.markdown("**Duplicate detection**")
        if q["duplicates"] == 0:
            st.success("No exact duplicate rows.")
        else:
            st.warning(f"{q['duplicates']} exact duplicate rows found.")
            st.dataframe(df[df.duplicated(keep=False)].head(50), width="stretch")
        if {"vc", "f", "ap"}.issubset(df.columns):
            key = [k for k in ("material", "vc", "f", "ap") if k in df.columns]
            rep = df.groupby(key).size()
            st.markdown("**Repeated parameter settings**")
            st.write(f"{(rep > 1).sum()} of {len(rep)} parameter combinations were cut more than once "
                     f"(max {rep.max()} repetitions). Repetitions are genuine replicate cuts, not duplicates; the "
                     "*grouped* split on the training page keeps them on the same side of the train/test split.")

with tabs[3]:
    st.dataframe(df.describe().T.round(4), width="stretch")
    if "material" in df.columns and "Ra" in df.columns:
        st.markdown("**Measured Ra by material [µm]**")
        st.dataframe(df.groupby("material")["Ra"].describe().round(3), width="stretch")

with tabs[4]:
    num = df.select_dtypes("number").columns.tolist()
    default = [c for c in ["vc", "f", "ap", "hardness_HV", "cut_index", "Ra", "Rz", "MRR_mm3_min"] if c in num] or num[:8]
    sel = st.multiselect("Columns", num, default=default, key="corr_cols")
    if len(sel) >= 2:
        ui.show_fig(viz.correlation_heatmap(df[sel]))
    else:
        st.info("Select at least two numeric columns.")

with tabs[5]:
    allc = df.columns.tolist()
    default = [c for c in ["vc", "f", "ap", "material", "Ra", "MRR_mm3_min"] if c in allc] or allc[:6]
    sel = st.multiselect("Columns", allc, default=default, key="dist_cols")
    if sel:
        ui.show_fig(viz.distributions(df, sel))

with tabs[6]:
    if {"vc", "f"}.issubset(df.columns):
        import matplotlib.pyplot as plt
        hue = "material" if "material" in df.columns else None
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        groups = df.groupby(hue) if hue else [("all", df)]
        for i, (k, g) in enumerate(groups):
            axes[0].scatter(g["vc"], g["f"], s=18 + 30 * (g["ap"] if "ap" in g else 0), alpha=.6,
                            color=viz.PALETTE[i % 6], edgecolor=viz.NAVY, linewidth=.3, label=str(k))
            if "Ra" in g:
                axes[1].scatter(g["f"], g["Ra"], s=16, alpha=.7, color=viz.PALETTE[i % 6], label=str(k))
        axes[0].set_xlabel("Vc [m/min]"); axes[0].set_ylabel("f [mm/rev]"); axes[0].set_title("Tested parameter combinations (size ∝ ap)")
        axes[1].set_xlabel("f [mm/rev]"); axes[1].set_ylabel("Ra [µm]"); axes[1].set_title("Measured Ra vs feed")
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
        fig.tight_layout()
        ui.show_fig(fig)
    else:
        st.info("Cutting speed and feed columns are required for this view.")
