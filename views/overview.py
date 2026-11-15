import streamlit as st

from src import ui
from src.data_loader import DATASET_CITATION, DATASET_TITLE, DATASET_URL
from src.mrr_calculator import MRR_EQUATION

ui.header("Advanced Machining Processes · MEDX 55 · Simulation Based Project No. 08",
          "ML Prediction of Material Removal Rate & Surface Roughness",
          "Regression and machine-learning models trained on a published longitudinal-turning experiment to predict "
          "productivity (MRR) and surface quality (Ra) from cutting speed, feed and depth of cut.")

df = ui.get_df()
c1, c2, c3, c4 = st.columns(4)
if df is not None:
    with c1:
        ui.kpi("Experimental cuts", f"{len(df)}", "rows in the loaded dataset")
    with c2:
        n_mat = df["material"].nunique() if "material" in df.columns else "—"
        ui.kpi("Workpiece materials", f"{n_mat}", ", ".join(sorted(df["material"].astype(str).unique())) if "material" in df.columns else "")
    with c3:
        if "Ra" in df.columns:
            ui.kpi("Measured Ra range", f"{df['Ra'].min():.2f}–{df['Ra'].max():.2f}", "µm")
    with c4:
        if "MRR_mm3_min" in df.columns:
            ui.kpi("Calculated MRR range", f"{df['MRR_mm3_min'].min()/1000:.1f}–{df['MRR_mm3_min'].max()/1000:.1f}", "cm³/min (×1000 mm³/min)")
else:
    st.info("No dataset loaded yet.")

st.write("")
a, b = st.columns([1.15, 1])
with a:
    ui.card("Introduction", """<p>Turning is the most common process for producing rotational parts. Two indicators
dominate process planning: <b>productivity</b>, measured by the material removal rate (MRR), and <b>surface quality</b>,
usually specified by the arithmetic mean roughness R<sub>a</sub>. Higher feed and depth of cut raise MRR but feed also
increases roughness, so the parameters must be chosen as a compromise.</p>
<p>This project learns these relationships from real experimental data. Several regression algorithms are trained,
compared on an independent test set, and deployed in interactive simulators.</p>""", accent=True)
with b:
    ui.card("Objectives", """<ol style="margin:0;padding-left:1.1rem">
<li>Predict Material Removal Rate (MRR).</li>
<li>Predict surface roughness (R<sub>a</sub>).</li>
<li>Compare Linear Regression, Decision Tree, Random Forest, Gradient Boosting and SVR.</li>
<li>Visualise actual vs. predicted results and residuals.</li>
<li>Evaluate with R², MAE, MSE and RMSE.</li>
<li>Provide interactive prediction from machining parameters.</li>
<li>Explain how the parameters affect machining performance.</li></ol>""")

st.subheader("Machining performance indicators")
m1, m2 = st.columns(2)
with m1:
    ui.card("Material Removal Rate <span class='tag calc'>calculated target</span>", f"""
<div class="eq">{MRR_EQUATION} &nbsp; [mm³/min]</div>
<p style="margin-top:.6rem">V<sub>c</sub> cutting speed (m/min), f feed (mm/rev), a<sub>p</sub> depth of cut (mm).
The factor 1000 converts metres to millimetres. MRR is the volume of chip removed per minute.</p>
<p>The dataset does <b>not</b> contain a measured MRR. It is computed from the recorded nominal parameters of every cut
and is therefore labelled as a calculated target throughout the application.</p>""")
with m2:
    ui.card("Surface roughness R<sub>a</sub> <span class='tag meas'>measured target</span>", """
<div class="eq">Ra = (1/l) ∫₀ˡ |z(x)| dx &nbsp; [µm]</div>
<p style="margin-top:.6rem">R<sub>a</sub> is the arithmetic mean deviation of the profile from its mean line (ISO 4287).
It was measured on the workpiece after each cut and is the experimentally observed target.</p>
<p>Ideal (kinematic) theory for a round-nose tool gives R<sub>a,th</sub> ≈ f² / (32·r<sub>ε</sub>). Real surfaces
deviate because of tool wear, vibration, material side-flow and built-up edge.</p>""")

st.subheader("Dataset source")
ui.card(DATASET_TITLE, f"""<p>{DATASET_CITATION}</p>
<p>Longitudinal turning on an NTX 1000 machine (as recorded in the file metadata) with a DNMG150608 insert (corner radius 0.8 mm) on three workpiece materials.
For each cut the database stores the cutting parameters, the measured surface roughness (Ra, Rz), measured diameters and
multivariate time series (drive forces, currents, positions, accelerations). The raw 982 MB HDF5 file was parsed into one
row per cut by <code>scripts/build_dataset.py</code>; measured values were not modified.</p>
<p><a href="{DATASET_URL}" target="_blank">{DATASET_URL}</a></p>""", accent=True)

st.subheader("Technologies")
t = st.columns(4)
for col, (name, desc) in zip(t, [
    ("Python 3 · pandas · NumPy", "Data extraction from HDF5 (h5py), cleaning and feature tables"),
    ("scikit-learn", "Pipelines, GridSearchCV, k-fold CV, five regression algorithms, permutation importance"),
    ("Matplotlib · Seaborn", "Scientific figures shared by the dashboard and the PDF report"),
    ("Streamlit · ReportLab", "Interactive engineering dashboard and downloadable PDF report"),
]):
    with col:
        ui.card(name, f"<p>{desc}</p>")

st.write("")
ui.note("<b>Workflow:</b> Dataset Explorer → Model Training (MRR and Ra) → Performance Analysis → Simulators → "
        "Engineering Insights → Report Generator. Pre-trained models for both targets are included, so the simulators work immediately.")
