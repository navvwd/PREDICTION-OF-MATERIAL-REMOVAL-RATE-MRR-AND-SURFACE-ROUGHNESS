import datetime as dt

import pandas as pd

import streamlit as st

from src import insights as ins
from src import ui
from src.data_loader import column_table, is_hannover
from src.report_generator import build_report

ui.header("Page 8 · Output", "Report Generator",
          "Generate a PDF project report containing the dataset description, equations, methodology, the actual "
          "metrics and figures of the models trained in this session, discussion, conclusion and references.")

if ui.need_data():
    st.stop()
df = ui.get_df()
runs = {k: ui.get_run(k) for k in ("MRR", "Ra")}

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Included results**")
    for k in ("Ra", "MRR"):
        r = runs[k]
        if r:
            st.markdown(f"✅ {k}: {r.best_model_name} — test R² {r.best['test_metrics']['R2']:.4f} ({len(r.results)} algorithms)")
        else:
            st.markdown(f"⚠️ {k}: not trained — this section will state that no results are available")
with c2:
    authors = st.text_input("Student name(s) / roll numbers (optional)")
    inst = st.text_input("Department / institution (optional)")

if not any(runs.values()):
    st.warning("No model has been trained yet. The report can still be generated, but it will contain no results. "
               "Train models on the **Model Training** page first.")


def dataset_info():
    key_cols = [c for c in ["material", "hardness_HV", "vc", "f", "ap", "re", "Ra", "Rz", "MRR_mm3_min"] if c in df.columns]
    ct = column_table(df[key_cols])[["Column", "Meaning", "Unit", "Min", "Max", "Unique"]]
    for c in ("Min", "Max"):
        ct[c] = ct[c].map(lambda v: f"{v:.4g}" if isinstance(v, (int, float)) and pd.notna(v) else "—")
    desc = ("The database (HDF5, 982 MB) contains three workpieces — 20MnCr5, 41Cr4 and EN AW-6082 — each turned with "
            "151 longitudinal cuts using a DNMG150608 insert (corner radius 0.8 mm). For every cut the cutting speed, feed "
            "and depth of cut, the measured roughness values Ra and Rz, the measured diameters and multivariate process "
            "time series are stored. Units were verified from the data: the nominal diameter reduction equals 2·ap for "
            "every cut (ap in mm), and measured Ra agrees in magnitude with the kinematic value f²/(32·rε) (f in mm/rev, "
            "Ra in µm).") if is_hannover(df) else ("User-supplied CSV dataset (" + st.session_state.get("dataset_name", "") +
                                                   "). Column meanings and units were mapped by the user.")
    disc = ins.ra_observations(df) + ins.mrr_observations(df) + ins.algorithm_observations(runs)
    concl = []
    for k, unit in (("Ra", "µm"), ("MRR", "mm³/min")):
        r = runs[k]
        if r:
            m = r.best["test_metrics"]
            concl.append(f"For {k}, {r.best_model_name} was selected by cross-validation and achieved R² = {m['R2']:.3f}, "
                         f"MAE = {m['MAE']:.4g} {unit} and RMSE = {m['RMSE']:.4g} {unit} on {r.n_test} unseen test samples.")
    if "Ra" in df.columns:
        corr = ins.correlation_table(df, "Ra", by=None)
        if not corr.empty:
            top = corr.loc[corr["Pearson r"].abs().idxmax()]
            concl.append(f"In this dataset the parameter most strongly associated with Ra is {ins.PARAM_LABELS.get(top['Parameter'], top['Parameter'])} "
                         f"(|r| = {abs(top['Pearson r']):.2f}), whereas MRR is by definition proportional to all three parameters.")
    concl.append("The models are valid only within the tested parameter ranges, workpiece materials and tool.")
    return {"n_rows": len(df), "description": desc, "columns_table": ct, "design_text": "Experimental design: " + ins.design_summary(df),
            "discussion": disc, "conclusion": " ".join(concl),
            "heatmap_cols": [c for c in ["vc", "f", "ap", "hardness_HV", "Ra", "Rz", "MRR_mm3_min"] if c in df.columns]}


if st.button("Generate PDF report", type="primary"):
    try:
        with st.spinner("Rendering figures and building the PDF…"):
            st.session_state.report_pdf = build_report(df, dataset_info(), runs, authors, inst)
        st.success(f"Report generated ({len(st.session_state.report_pdf)/1024:.0f} kB).")
    except Exception as e:
        st.error(f"Report generation failed: {e}")

if st.session_state.get("report_pdf"):
    st.download_button("Download PDF report", st.session_state.report_pdf,
                       f"MEDX55_Project08_ML_Machining_Report_{dt.date.today():%Y%m%d}.pdf", "application/pdf", type="primary")

st.divider()
st.markdown("**Data exports**")
e1, e2 = st.columns(2)
with e1:
    st.download_button("Dataset with calculated MRR (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "dataset_with_MRR.csv", "text/csv")
with e2:
    import pandas as pd
    from src.evaluation import metrics_table
    frames = [metrics_table(r.results).assign(Target=k) for k, r in runs.items() if r]
    if frames:
        st.download_button("All model metrics (CSV)", pd.concat(frames).to_csv(index=False).encode("utf-8"),
                           "all_model_metrics.csv", "text/csv")
