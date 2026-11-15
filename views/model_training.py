import io

import joblib
import streamlit as st

from src import ui
from src import visualization as viz
from src.data_loader import useful_features
from src.evaluation import metrics_table
from src.model_training import ALGORITHMS, MIN_SAMPLES, RANDOM_STATE, train_models, validate_training_data
from src.prediction import MODEL_FILES, bundle_from_run, save_bundle
from src.preprocessing import TARGETS, default_features

ui.header("Page 3 · Machine Learning", "Model Training",
          "Train and compare five regression algorithms with an 80/20 hold-out split, k-fold cross-validated "
          "hyper-parameter tuning and leakage-free preprocessing pipelines.")

if ui.need_data():
    st.stop()
ss = st.session_state
df = ss.df

available = [k for k in ("MRR", "Ra") if TARGETS[k]["column"] in df.columns]
if not available:
    st.error("The loaded dataset has neither a Ra column nor the Vc/f/ap columns needed to calculate MRR.")
    st.stop()

with st.container(border=True):
    c1, c2 = st.columns([1, 2])
    with c1:
        kind = st.radio("Prediction target", available, horizontal=True,
                        format_func=lambda k: f"{k} — {TARGETS[k]['label']}")
        tinfo = TARGETS[kind]
        st.markdown(f"<span class='tag {'calc' if tinfo['kind']=='calculated' else 'meas'}'>{tinfo['kind']} target</span>"
                    f"<span class='tag grey'>{tinfo['column']} [{tinfo['unit']}]</span>", unsafe_allow_html=True)
    with c2:
        cand = useful_features(df)
        feats = st.multiselect("Input features", cand, default=[f for f in default_features(df, kind) if f in cand],
                               key=f"feats_{kind}",
                               help="The simulators ask for exactly these inputs. Use process parameters only "
                                    "(vc, f, ap, material) for a model that can be used before machining.")
    algos = st.multiselect("Algorithms", list(ALGORITHMS), default=list(ALGORITHMS), key="algos")
    c3, c4, c5 = st.columns(3)
    with c3:
        split = st.radio("Train/test split (80/20, seed 42)", ["grouped", "random"],
                         format_func=lambda s: {"grouped": "Grouped by parameter setting (recommended)",
                                                "random": "Random rows"}[s])
    with c4:
        folds = st.slider("Cross-validation folds", 3, 10, 5)
    with c5:
        tune = st.checkbox("Hyper-parameter tuning (grid search)", value=True)
        st.caption(f"Minimum {MIN_SAMPLES} valid rows required.")

    if kind == "MRR":
        ui.note("MRR is a deterministic function of Vc, f and ap. Very high scores are expected and only show that a "
                "model can reproduce the equation — they do not indicate experimental accuracy.", warn=True)
    if any(f in feats for f in ("Rz",)):
        st.warning("Rz is another roughness measurement made after cutting; using it to predict Ra would be target leakage.")

    go = st.button("Train models", type="primary", disabled=not (feats and algos))

if go:
    errs = validate_training_data(df, feats, tinfo["column"])
    if errs:
        for e in errs:
            st.error(e)
    else:
        bar = st.progress(0.0, text="Preparing…")

        def cb(i, n, name):
            bar.progress(i / n, text=f"Training {name} ({i + 1}/{n})…" if name != "done" else "Finished")

        try:
            with st.spinner("Fitting pipelines with cross-validation…"):
                run = train_models(df, feats, tinfo["column"], algos, split_mode=split, cv_folds=folds, tune=tune,
                                   progress_cb=cb)
        except Exception as e:  # keep the app alive on unexpected data problems
            bar.empty()
            st.error(f"Training failed: {e}")
            st.stop()
        ss.runs[kind] = run
        ss.bundles[kind] = bundle_from_run(run, target_kind=kind, dataset_name=ss.get("dataset_name", ""))
        ss[f"saved_{kind}"] = False
        st.toast(f"{kind}: trained {len(algos)} model(s). Best by CV RMSE: {run.best_model_name}")

run = ui.get_run(kind)
if run is None:
    st.info(f"No {kind} model trained yet in this session. Configure the options above and press **Train models**.")
    st.stop()

st.subheader(f"Results — {kind}")
for w in run.warnings:
    st.warning(w)
b = run.best
c = st.columns(5)
with c[0]:
    ui.kpi("Selected model", run.best_model_name, "lowest CV RMSE")
with c[1]:
    ui.kpi("Test R²", f"{b['test_metrics']['R2']:.4f}", f"n_test = {run.n_test}")
with c[2]:
    ui.kpi("Test RMSE", f"{b['test_metrics']['RMSE']:.4g}", tinfo["unit"])
with c[3]:
    ui.kpi("Test MAE", f"{b['test_metrics']['MAE']:.4g}", tinfo["unit"])
with c[4]:
    ui.kpi("Split", run.split_mode, f"train {run.n_train} / test {run.n_test}")
st.caption(f"Features: {', '.join(run.features)} · {run.results[run.best_model_name]['cv_folds']}-fold CV on the training set · "
           f"random seed {RANDOM_STATE}. Model selection uses CV RMSE only; the test set is used once for reporting.")

table = metrics_table(run.results)
show = table[["Model", "CV RMSE (mean)", "CV RMSE (std)", "Train R2", "Test R2", "Test MAE", "Test MSE", "Test RMSE",
              "Test MAPE_%", "Fit time (s)"]]
st.dataframe(show.style.format({c: "{:.4g}" for c in show.columns if c != "Model"})
             .apply(lambda r: ["background-color:#eff5ff;font-weight:600" if r["Model"] == run.best_model_name else ""
                               for _ in r], axis=1), width="stretch", hide_index=True)
st.download_button("Export metrics (CSV)", table.to_csv(index=False).encode("utf-8"), f"{kind}_model_metrics.csv", "text/csv")

ui.show_fig(viz.model_comparison(table, f"{kind}: algorithm comparison"))

with st.expander("Tuned hyper-parameters"):
    for name, r in run.results.items():
        st.write(f"**{name}** — {r['best_params'] or 'defaults (no grid)'}")

st.subheader("Save model")
names = list(run.results)
cur = (ss.bundles.get(kind) or {}).get("model_name")
cc1, cc2, cc3 = st.columns([1.3, 1, 1])
with cc1:
    choice = st.selectbox("Model used by the simulator", names,
                          index=names.index(cur) if cur in names else names.index(run.best_model_name),
                          help="Defaults to the CV-selected model.")
bundle = bundle_from_run(run, choice, target_kind=kind, dataset_name=ss.get("dataset_name", ""))
ss.bundles[kind] = bundle
with cc2:
    st.write("")
    if st.button(f"Save to {MODEL_FILES[kind].name}", type="primary"):
        try:
            p = save_bundle(bundle)
            ss[f"saved_{kind}"] = True
            st.success(f"Saved {choice} → models/{p.name}")
        except Exception as e:
            st.error(f"Could not save model: {e}")
with cc3:
    st.write("")
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    st.download_button("Download model (.pkl)", buf.getvalue(), MODEL_FILES[kind].name, "application/octet-stream")
st.caption(f"Active {kind} model for the simulator in this session: **{choice}**. Saving makes it persist after restart.")
