import numpy as np
import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.insights import kinematic_ra
from src.mrr_calculator import calculate_mrr, validate_inputs
from src.prediction import extrapolation_warnings, predict_one
from src.sim_inputs import label, render_inputs

# ISO 1302 roughness grade numbers (Ra upper limits in µm)
N_GRADES = [(0.025, "N1"), (0.05, "N2"), (0.1, "N3"), (0.2, "N4"), (0.4, "N5"), (0.8, "N6"), (1.6, "N7"),
            (3.2, "N8"), (6.3, "N9"), (12.5, "N10"), (25, "N11"), (50, "N12")]

ui.header("Page 5 · Simulator", "Surface Roughness Prediction",
          "Predict the arithmetic mean roughness Ra for a chosen material and set of cutting parameters, and compare "
          "it with the kinematic theory and the nearest real measurements.")

if ui.need_model("Ra"):
    st.stop()
bundle = ui.get_bundle("Ra")
df = ui.get_df()
m = bundle["test_metrics"]
st.markdown(f"<span class='tag'>{bundle['model_name']}</span><span class='tag meas'>trained on measured Ra</span>"
            f"<span class='tag grey'>test R² {m['R2']:.3f} · RMSE {m['RMSE']:.3f} µm · MAE {m['MAE']:.3f} µm</span>",
            unsafe_allow_html=True)
st.write("")

with st.form("ra_form"):
    vals = render_inputs(bundle, "ra", df)
    submitted = st.form_submit_button("Predict surface roughness", type="primary")

if submitted:
    errs = validate_inputs(vals["vc"], vals["f"], vals["ap"]) if all(k in vals for k in ("vc", "f", "ap")) else []
    if errs:
        for e in errs:
            st.error(e)
    else:
        try:
            st.session_state.ra_pred = (vals, predict_one(bundle, vals))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("ra_pred")
if not (res and set(res[0]) == set(bundle["features"])):
    st.info("Choose the parameters and press **Predict surface roughness**.")
    st.stop()

vals, y = res
grade = next((g for lim, g in N_GRADES if y <= lim), ">N12")
r_eps = float(df["re"].iloc[0]) if df is not None and "re" in df.columns and df["re"].nunique() == 1 else None
ra_th = float(kinematic_ra(vals["f"], r_eps)) if (r_eps and "f" in vals) else None

l, r = st.columns([1.1, 1])
with l:
    st.markdown(f"""<div class="result"><div class="label">Predicted surface roughness</div>
<div class="value">Ra {y:.3f}<span class="unit">µm</span></div>
<div class="meta">ISO 1302 grade {grade} · model: {bundle['model_name']} · typical test error ±{m['RMSE']:.2f} µm (RMSE)</div></div>""",
                unsafe_allow_html=True)
    for w in extrapolation_warnings(bundle, vals):
        st.warning(w)
    if y <= 0:
        st.warning("The model returned a non-physical value (≤ 0); the input lies outside the region it can represent.")
with r:
    rows = [{"Parameter": label(k), "Value": v} for k, v in vals.items()]
    if ra_th is not None:
        rows.append({"Parameter": f"Kinematic Ra f²/(32·rε), rε={r_eps:g} mm [µm]", "Value": round(ra_th, 3)})
    if all(k in vals for k in ("vc", "f", "ap")):
        rows.append({"Parameter": "MRR (calculated) [mm³/min]", "Value": round(float(calculate_mrr(vals["vc"], vals["f"], vals["ap"])), 1)})
    tbl = pd.DataFrame(rows).astype({"Value": str})
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.download_button("Export prediction (CSV)", pd.concat([tbl, pd.DataFrame([{"Parameter": "Predicted Ra [µm]", "Value": f"{y:.4f}"}])])
                       .to_csv(index=False).encode("utf-8"), "ra_prediction.csv", "text/csv")

st.subheader("Prediction explanation")
expl = [f"The {bundle['model_name']} model was trained on {bundle['n_train']} measured cuts using "
        f"{', '.join(label(f) for f in bundle['features'])}, and on the held-out test cuts it achieved R² = {m['R2']:.3f} "
        f"with RMSE = {m['RMSE']:.3f} µm."]
if ra_th is not None:
    expl.append(f"Kinematic theory predicts Ra ≈ f²/(32·rε) = {vals['f']:g}²/(32·{r_eps:g}) × 1000 = {ra_th:.3f} µm "
                f"for a perfectly sharp tool with no vibration. The model predicts {y/ra_th:.2f}× this value, because it also "
                "reflects the real effects present in the experiments (material behaviour, tool condition, dynamics).")
ui.note(" ".join(expl))

# nearest real measurements
if df is not None and all(k in df.columns for k in ("vc", "f", "ap", "Ra")):
    st.subheader("Nearest experimental measurements")
    d = df.copy()
    if "material" in vals and "material" in d.columns:
        d = d[d["material"].astype(str) == str(vals["material"])]
    if not d.empty:
        dist = sum(((d[k] - vals[k]) / (df[k].max() - df[k].min() or 1)) ** 2 for k in ("vc", "f", "ap"))
        near = d.assign(distance=np.sqrt(dist)).nsmallest(5, "distance")
        cols = [c for c in ["sample_id", "material", "cut_index", "vc", "f", "ap", "Ra", "Rz", "distance"] if c in near.columns]
        st.dataframe(near[cols].round(4), hide_index=True, width="stretch")
        st.caption("Real measured cuts closest in normalised (Vc, f, ap) space — use them to judge the plausibility of the prediction.")

# response curve
num_feats = bundle["numeric_features"]
if num_feats:
    sweep = st.selectbox("Sensitivity: vary one parameter (others fixed)", num_feats,
                         index=num_feats.index("f") if "f" in num_feats else 0, format_func=label)
    lo, hi, _ = bundle["feature_ranges"][sweep]
    xs = np.linspace(lo, hi, 60)
    X = pd.DataFrame([{**vals, sweep: x} for x in xs])[bundle["features"]]
    ys = bundle["estimator"].predict(X)
    an = kinematic_ra(xs, r_eps) if (sweep == "f" and r_eps) else None
    fig = viz.response_curve(xs, ys, label(sweep), "Ra [µm]", point=(vals[sweep], y),
                             title=f"Ra response to {label(sweep)}", analytic=an)
    if an is not None:
        fig.axes[0].get_lines()[1].set_label("kinematic f²/(32·rε)")
        fig.axes[0].legend(fontsize=8)
    ui.show_fig(fig)
