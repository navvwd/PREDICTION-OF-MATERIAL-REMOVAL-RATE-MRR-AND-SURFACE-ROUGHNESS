# Machining ML Lab — Prediction of MRR and Surface Roughness in Turning

**MEDX 55 Advanced Machining Processes · Project Based Learning · Simulation Based Project No. 08**

A Streamlit application that trains and compares five regression algorithms to predict:

* **Surface roughness Ra**, the *measured* target;
* **Material removal rate MRR**, *calculated* as `1000·Vc·f·ap`.

The models are trained on the published Leibniz University Hannover longitudinal-turning dataset.

> Dataset: Denkena, B., Bergmann, B., Buhl, H., Handrup, M. (2026). *Multivariate time series dataset for predicting surface roughness and dimensional error in longitudinal turning.* LUIS, Leibniz University Hannover. https://doi.org/10.25835/9xffqwnc (CC BY 3.0)

## Quick start

```bash
cd machining-ml-project
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. The extracted dataset (`data/dataset.csv`) and pre-trained models (`models/*.pkl`) are included, so every page works immediately.

## Pages

| # | Page | What it does |
|---|---|---|
| 1 | Project Overview | Title, objectives, MRR / Ra theory, dataset source, technologies |
| 2 | Dataset Explorer | Published dataset or CSV upload with column mapping; preview, column dictionary, missing values, duplicates, statistics, correlation heatmap, distributions, design space; CSV export |
| 3 | Model Training | Choose target (MRR / Ra), features, algorithms, grouped or random 80/20 split, CV folds, grid search; progress bar; metrics table and charts; save or download the model |
| 4 | MRR Simulator | Vc, f, ap → predicted MRR, analytical check, sensitivity curve, CSV export |
| 5 | Roughness Simulator | Vc, f, ap, material → predicted Ra, ISO 1302 grade, kinematic f²/(32rε) comparison, nearest real measurements, sensitivity curve |
| 6 | Performance Analysis | Actual vs predicted (MRR and Ra), residuals, model comparison, permutation importance, heatmap, metrics and test-prediction export |
| 7 | Engineering Insights | Effect of Vc, f, ap on MRR and Ra; observed (computed live) vs general principles; algorithm comparison |
| 8 | Report Generator | Downloadable PDF report with the actual metrics and figures; CSV exports |

## Results (default settings: grouped split, seed 42, 5-fold CV)

| Target | Selected model | Test R² | Test MAE | Test RMSE | n test |
|---|---|---|---|---|---|
| Ra (measured) | SVR (RBF) | 0.977 | 0.090 µm | 0.154 µm | 89 |
| MRR (calculated) | SVR (RBF) | 0.995 | 323 mm³/min | 932 mm³/min | 99 |

The full tables are in `results/`. The full academic report is `docs/Project_Report.md`. A high MRR score only shows that a model can reproduce a known equation; it says nothing about experimental accuracy.

## Project structure

```
machining-ml-project/
├── app.py                     Streamlit entry point (st.navigation)
├── requirements.txt
├── .streamlit/config.toml     light engineering theme
├── assets/styles.css          custom CSS
├── data/dataset.csv           453 cuts extracted from the HDF5 file
├── models/                    mrr_model.pkl, roughness_model.pkl (pipeline + metadata + full comparison)
├── results/                   metrics / importance CSVs, summary.json from the default training
├── docs/Project_Report.md     B.Tech project report
├── scripts/
│   ├── build_dataset.py       HDF5 → CSV (optionally downloads the 982 MB file)
│   └── train_default_models.py  retrain and save both default models
├── src/
│   ├── data_loader.py         loading, validation, column mapping, metadata, references
│   ├── preprocessing.py       targets and default features
│   ├── mrr_calculator.py      MRR equation and input validation
│   ├── model_training.py      pipelines, split, GridSearchCV, model selection, permutation importance
│   ├── evaluation.py          R², MAE, MSE, RMSE
│   ├── prediction.py          model persistence and single prediction
│   ├── visualization.py       Matplotlib / Seaborn figures
│   ├── insights.py            data-derived observations
│   ├── report_generator.py    ReportLab PDF
│   ├── sim_inputs.py, ui.py   shared UI helpers
├── views/                     the eight pages (named views/ to avoid Streamlit's legacy pages/ auto-navigation)
└── tests/                     pytest unit tests + Streamlit AppTest UI tests
```

## Rebuilding the dataset from the original source

```bash
python scripts/build_dataset.py --download        # downloads Database.h5 (~982 MB) to data/raw/ and writes data/dataset.csv
python scripts/build_dataset.py path/to/database.h5
python scripts/train_default_models.py            # retrain and save both models and results/
```

## Methodology in brief

* **Features.**
  * MRR: Vc, f, ap.
  * Ra: Vc, f, ap and material (one-hot encoded).
  * Rz is never used as an input, because it is a second measurement of the target and would leak it.
* **Split.** 80/20, seed 42. The default *grouped* split keeps every repeated cut of a parameter setting on one side, so the test score reflects unseen settings.
* **Preprocessing.** Imputation, scaling and encoding all run inside scikit-learn pipelines and are fitted on training folds only, so there is no data leakage.
* **Tuning.** `GridSearchCV` with k-fold cross-validation (GroupKFold for the grouped split).
* **Model selection.** The lowest mean CV RMSE wins. The test set is used once, for reporting.

## Using your own CSV

On **Dataset Explorer**, choose *Upload a CSV file*. Comma, semicolon and tab separators are accepted. Map the columns for cutting speed (m/min), feed (mm/rev), depth of cut (mm) and Ra (µm); common names are detected automatically. MRR is calculated when Vc, f and ap are mapped. The app handles a range of input problems with clear messages and does not crash:

* missing values;
* non-numeric entries;
* wrong column names;
* too few rows (minimum 20);
* missing features;
* pages opened before any model is trained.

## Tests

```bash
python -m pytest -q tests
```

The suite has 24 tests:

* **Equations and data handling:** the MRR equation, CSV parsing and column aliases.
* **Edge cases:** missing values, a dataset that is too small, a missing feature.
* **Modelling:** all five algorithms, prediction and PDF generation.
* **UI:** every page renders without exceptions, and the Train, Predict and Generate PDF buttons work.

## Requirements

Python ≥ 3.10. The packages are streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib and reportlab. h5py and requests are needed only to rebuild the dataset from HDF5.

The saved `.pkl` models were created with scikit-learn 1.7. If loading fails with a different version, the app simply asks you to retrain, which takes about 10 seconds on the Model Training page; alternatively run `scripts/train_default_models.py`.
