import numpy as np
import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.mrr_calculator import MRR_EQUATION, calculate_mrr, validate_inputs
from src.prediction import extrapolation_warnings, predict_one
from src.sim_inputs import label, render_inputs

ui.header("Page 4 · Simulator", "MRR Prediction Simulator",
          "Enter turning parameters and obtain the material removal rate predicted by the trained regression model, "
          "verified against the analytical equation.")

if ui.need_model("MRR"):
    st.stop()
bundle = ui.get_bundle("MRR")
m = bundle["test_metrics"]
st.markdown(f"<span class='tag'>{bundle['model_name']}</span><span class='tag calc'>target calculated from 1000·Vc·f·ap</span>"
            f"<span class='tag grey'>test R² {m['R2']:.4f} · RMSE {m['RMSE']:.4g} mm³/min</span>", unsafe_allow_html=True)
st.write("")

with st.form("mrr_form"):
    vals = render_inputs(bundle, "mrr", ui.get_df())
    submitted = st.form_submit_button("Predict MRR", type="primary")

if submitted:
    errs = []
    if all(k in vals for k in ("vc", "f", "ap")):
        errs = validate_inputs(vals["vc"], vals["f"], vals["ap"])
    errs += [f"{label(k)} must be positive." for k, v in vals.items()
             if k not in ("vc", "f", "ap") and isinstance(v, float) and v <= 0 and k != "cut_index"]
    if errs:
        for e in errs:
            st.error(e)
    else:
        try:
            st.session_state.mrr_pred = (vals, predict_one(bundle, vals))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("mrr_pred")
if res and set(res[0]) == set(bundle["features"]):
    vals, y = res
    has_eq = all(k in vals for k in ("vc", "f", "ap"))
    analytic = float(calculate_mrr(vals["vc"], vals["f"], vals["ap"])) if has_eq else None
    l, r = st.columns([1.1, 1])
    with l:
        st.markdown(f"""<div class="result"><div class="label">Predicted material removal rate</div>
<div class="value">{y:,.0f}<span class="unit">mm³/min</span></div>
<div class="meta">= {y/1000:,.2f} cm³/min · model: {bundle['model_name']}</div></div>""", unsafe_allow_html=True)
        for w in extrapolation_warnings(bundle, vals):
            st.warning(w)
    with r:
        rows = [{"Parameter": label(k), "Value": v} for k, v in vals.items()]
        if analytic is not None:
            rows.append({"Parameter": "Analytical MRR [mm³/min]", "Value": round(analytic, 1)})
            rows.append({"Parameter": "Model − analytical [%]", "Value": round((y - analytic) / analytic * 100, 2)})
        tbl = pd.DataFrame(rows).astype({"Value": str})
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.download_button("Export prediction (CSV)", tbl.to_csv(index=False).encode("utf-8"), "mrr_prediction.csv", "text/csv")

    st.subheader("Prediction explanation")
    if analytic is not None:
        ui.note(f"<b>Equation:</b> {MRR_EQUATION} = 1000 × {vals['vc']:g} m/min × {vals['f']:g} mm/rev × {vals['ap']:g} mm "
                f"= <b>{analytic:,.1f} mm³/min</b>. The {bundle['model_name']} model, which learned this relation only from "
                f"the {bundle['n_train']} training rows, returns {y:,.1f} mm³/min "
                f"({(y-analytic)/analytic*100:+.2f} %). For MRR the equation is exact for nominal parameters; the ML model "
                "is a demonstration of learning a known physical law and becomes unreliable outside the training range.")
    else:
        ui.note("This model was trained without all of Vc, f and ap, so no analytical comparison is possible.", warn=True)

    num_feats = [f for f in bundle["numeric_features"]]
    if num_feats:
        sweep = st.selectbox("Sensitivity: vary one parameter (others fixed at the values above)", num_feats,
                             format_func=label)
        lo, hi, _ = bundle["feature_ranges"][sweep]
        xs = np.linspace(lo, hi, 60)
        X = pd.DataFrame([{**vals, sweep: x} for x in xs])[bundle["features"]]
        ys = bundle["estimator"].predict(X)
        an = calculate_mrr(X["vc"], X["f"], X["ap"]) if has_eq else None
        ui.show_fig(viz.response_curve(xs, ys, label(sweep), "MRR [mm³/min]", point=(vals[sweep], y),
                                       title=f"MRR response to {label(sweep)}", analytic=an))
else:
    st.info("Set the parameters and press **Predict MRR**.")
