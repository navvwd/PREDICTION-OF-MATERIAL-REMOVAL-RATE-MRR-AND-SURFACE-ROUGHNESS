import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.evaluation import metrics_table
from src.model_training import feature_importance
from src.preprocessing import TARGETS

ui.header("Page 6 · Evaluation", "Model Performance Analysis",
          "Hold-out test results of the trained models: actual vs predicted, residuals, algorithm comparison, "
          "feature importance and the full metrics table.")

runs = {k: ui.get_run(k) for k in ("MRR", "Ra")}
if not any(runs.values()):
    st.warning("No trained models are available. Train at least one target on the **Model Training** page.")
    st.page_link("views/model_training.py", label="Go to Model Training", icon=":material/model_training:")
    st.stop()


@st.cache_data(show_spinner="Computing permutation importance…")
def _importance(kind: str, model_name: str, _run, run_id: int):
    return feature_importance(_run, model_name)


# ---- overview of both targets
st.subheader("Selected models")
cols = st.columns(2)
for col, kind in zip(cols, ("MRR", "Ra")):
    with col:
        run = runs[kind]
        if run is None:
            ui.card(kind, "<p>Not trained yet.</p>")
            continue
        b = run.best
        unit = TARGETS[kind]["unit"]
        ui.card(f"{kind} — {run.best_model_name}",
                f"<p>Test R² <b>{b['test_metrics']['R2']:.4f}</b> · RMSE <b>{b['test_metrics']['RMSE']:.4g} {unit}</b> · "
                f"MAE {b['test_metrics']['MAE']:.4g} {unit}<br>split: {run.split_mode} · train {run.n_train} / test {run.n_test} · "
                f"features: {', '.join(run.features)}</p>", accent=True)

# ---- actual vs predicted (both)
st.subheader("Actual vs predicted (hold-out test set)")
cols = st.columns(2)
for col, kind in zip(cols, ("MRR", "Ra")):
    with col:
        run = runs[kind]
        if run is not None:
            b = run.best
            ui.show_fig(viz.actual_vs_predicted(b["y_test"], b["y_pred_test"], f"{kind} — {run.best_model_name}",
                                                TARGETS[kind]["unit"], b["test_metrics"]))
        else:
            st.info(f"Train an {kind} model to see this plot.")

# ---- detailed per target
st.subheader("Detailed analysis")
kind = st.radio("Target", [k for k in ("Ra", "MRR") if runs[k] is not None], horizontal=True)
run = runs[kind]
unit = TARGETS[kind]["unit"]
names = list(run.results)
model_name = st.selectbox("Model", names, index=names.index(run.best_model_name))
r = run.results[model_name]

tabs = st.tabs(["Residuals", "Model comparison", "Feature importance", "Correlation heatmap", "Metrics table", "Predictions"])
with tabs[0]:
    ui.show_fig(viz.residual_plots(r["y_test"], r["y_pred_test"], f"{kind} residuals — {model_name}", unit))
    res = r["y_test"] - r["y_pred_test"]
    st.caption(f"Mean residual {res.mean():+.4g} {unit} (bias) · std {res.std():.4g} {unit} · "
               f"max |residual| {abs(res).max():.4g} {unit}.")
with tabs[1]:
    table = metrics_table(run.results)
    ui.show_fig(viz.model_comparison(table, f"{kind}: algorithm comparison"))
    st.caption("Bars: hold-out test metrics. Diamonds: mean cross-validated RMSE on the training set, which is the "
               "criterion used to select the model.")
with tabs[2]:
    if model_name in ("Decision Tree", "Random Forest", "Gradient Boosting"):
        st.caption("Permutation importance is shown (model-agnostic, on the test set). Impurity-based importances "
                   "of tree models can be biased and are not used.")
    imp = _importance(kind, model_name, run, id(run))
    ui.show_fig(viz.feature_importance_plot(imp, f"{kind}: permutation importance — {model_name}"))
    st.dataframe(imp.round(4), hide_index=True, width="stretch")
with tabs[3]:
    df = ui.get_df()
    if df is not None:
        cols_ = [c for c in run.numeric_features + [run.target] if c in df.columns]
        if len(cols_) >= 2:
            ui.show_fig(viz.correlation_heatmap(df[cols_], f"Correlation of model inputs with {kind}"))
with tabs[4]:
    table = metrics_table(run.results)
    st.dataframe(table.style.format({c: "{:.4g}" for c in table.columns if c != "Model"}), hide_index=True, width="stretch")
    st.download_button("Export metrics (CSV)", table.to_csv(index=False).encode("utf-8"), f"{kind}_metrics.csv", "text/csv")
with tabs[5]:
    pred = r["X_test"].copy()
    pred[f"actual_{kind}"] = r["y_test"]
    pred[f"predicted_{kind}"] = r["y_pred_test"]
    pred["residual"] = pred[f"actual_{kind}"] - pred[f"predicted_{kind}"]
    st.dataframe(pred.round(4), hide_index=True, width="stretch")
    st.download_button("Export test predictions (CSV)", pred.to_csv(index=False).encode("utf-8"),
                       f"{kind}_{model_name.replace(' ', '_')}_test_predictions.csv", "text/csv")
