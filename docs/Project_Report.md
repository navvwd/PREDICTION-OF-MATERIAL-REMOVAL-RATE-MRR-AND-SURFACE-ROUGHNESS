# Machine-Learning-Based Prediction of Material Removal Rate and Surface Roughness in Longitudinal Turning

**Subject:** Advanced Machining Processes (MEDX 55)
**Assignment:** Project Based Learning — Simulation Based Project No. 08
**Title:** Develop a regression / machine-learning model to predict Material Removal Rate and Surface Roughness using a published machining dataset

*Student name(s): ____________________  Roll no.: ____________  Department: ____________________*

> All numerical results in this report were produced by `scripts/train_default_models.py` (random seed 42, grouped 80/20 split) and are stored in `results/`. Re-running the script reproduces them. A PDF version with the same figures can be generated from the application (page **Report Generator**).

---

## 1. Abstract

This project develops and compares regression models that predict the arithmetic mean surface roughness (Ra) and the material removal rate (MRR) of longitudinal turning from the cutting speed, feed and depth of cut. The models are trained on the published experimental dataset of Denkena, Bergmann, Buhl and Handrup (Leibniz University Hannover, DOI 10.25835/9xffqwnc). It contains 453 cuts on three workpiece materials (20MnCr5, 41Cr4 and EN AW-6082) machined with a DNMG150608 insert. Ra is the experimentally measured target. MRR is not measured in the dataset, so it is **calculated** with MRR = 1000·Vc·f·ap. Five algorithms were tuned by 5-fold cross-validation on 80 % of the data: linear regression, decision tree, random forest, gradient boosting and support-vector regression (SVR). They were then evaluated once on a 20 % hold-out set. The split was grouped by parameter setting, so no setting appears in both sets. For Ra, an RBF-kernel SVR was selected (lowest cross-validated RMSE). It reached test R² = 0.977, MAE = 0.090 µm and RMSE = 0.154 µm on 89 unseen cuts. For the calculated MRR the selected SVR reached R² = 0.995 (RMSE = 932 mm³/min). That score only confirms that the model reproduces a known equation. The data show that feed governs roughness: Pearson r = 0.95 between f and Ra, and measured Ra is on median 1.11 × the kinematic value f²/(32·rε). Cutting speed and depth of cut have negligible linear correlation with Ra (|r| ≤ 0.07). The work is delivered as an interactive Streamlit application with a dataset explorer, training module, simulators, performance analysis and PDF report generation.

## 2. Introduction

Turning produces rotationally symmetric parts by removing material from a rotating workpiece with a single-point tool. Process planning has to balance two competing goals:

* **Productivity**: the material removal rate, i.e. the chip volume removed per unit time.
* **Surface quality**: most often specified by the arithmetic mean roughness Ra (ISO 4287).

Raising feed and depth of cut increases MRR. Feed, however, also increases the height of the feed marks left by the tool nose, and therefore the roughness. Analytical relations describe the ideal kinematic surface, but real surfaces also depend on tool wear, vibration, material side-flow and built-up edge. Regression and machine-learning (ML) models trained on experimental data can capture these effects within the experimental range. This project builds such models on a recently published, openly licensed turning dataset.

## 3. Literature Review

Benardos and Vosniakos (2003) reviewed approaches for predicting surface roughness in machining. They grouped them into approaches based on machining theory, experimental investigation, designed experiments, and artificial-intelligence methods. They noted that no single theoretical model captures all influencing factors, which motivates data-driven models. Özel and Karpat (2005) modelled surface roughness and tool flank wear in hard turning with both regression and neural-network models. They found that the neural networks predicted better than the regression models, which shows the value of non-linear models for roughness.

On the algorithm side, this project uses the following methods:

* **Decision trees.** Classification and regression trees (CART) by Breiman et al. (1984).
* **Random forests.** Bagged ensembles of decorrelated trees by Breiman (2001).
* **Gradient boosting.** Stage-wise additive tree ensembles by Friedman (2001).
* **Support-vector regression.** Drucker et al. (1997).

All are implemented in scikit-learn (Pedregosa et al., 2011). Standard textbooks (Kalpakjian & Schmid, 2014; Shaw, 2005) give the MRR relation for turning and the kinematic roughness of a round-nose tool used here as physical references.

The dataset used here was published in 2026 in the Leibniz University Hannover research-data repository under the Collaborative Research Centre SFB 1153. As its title states, it was created for predicting surface roughness and dimensional error in longitudinal turning. It provides multivariate process time series together with the resulting surface roughness and diameter deviation of each cut.

## 4. Problem Statement

Given the cutting speed Vc, feed f, depth of cut ap and the workpiece material of a longitudinal turning operation, predict:

* the arithmetic mean surface roughness Ra (µm) that will be measured on the machined surface;
* the material removal rate MRR (mm³/min).

The models must use genuine experimental data and must be evaluated on data not used for training. They must also be made available in an interactive simulator.

## 5. Objectives

1. Predict MRR.
2. Predict surface roughness Ra.
3. Compare linear regression, decision tree, random forest, gradient boosting and SVR.
4. Visualise actual versus predicted results and residuals.
5. Evaluate the models with R², MAE, MSE and RMSE.
6. Let users enter machining parameters and obtain predictions.
7. Explain how the machining parameters affect machining performance.

## 6. Dataset Description

**Source.** Denkena, B., Bergmann, B., Buhl, H., Handrup, M. (2026). *Multivariate time series dataset for predicting surface roughness and dimensional error in longitudinal turning* [Data set]. LUIS, Leibniz University Hannover. https://doi.org/10.25835/9xffqwnc. Licence CC BY 3.0.

**Format.** The source is a single HDF5 file, `Database.h5` (982 MB). Its structure was verified by inspection:

* **One group per workpiece.** K0025, K0026 and K0027, with attributes for material, hardness HV, tool and machine description ("Longitudinal turning on NTX 1000").
* **151 sub-groups per workpiece.** Each sub-group is one cut. Its scalar attributes are `vc`, `f`, `ap`, `re`, `Ra`, `Rz`, `d0`, `d1`, `d0_nominal`, `d1_nominal`, `d_error`, `T_error`, `T_length`, `Measurement_ID` and `num_segments`.
* **Time series per cut.** Forces, currents, positions, velocities and following errors from the NC at 1 kHz, plus three acceleration channels at about 20 kHz.

`scripts/build_dataset.py` converts the file into `data/dataset.csv` with one row per cut. Measured values are copied unchanged. The nominal parameters are rounded to 3 decimals to remove float32 artefacts (for example 0.5199999809 → 0.52). Per-cut mean, standard deviation and RMS of selected signals are added as optional columns.

**Unit verification.** The units are not written in the file, so they were checked against physics:

* (d0_nominal − d1_nominal) / (2·ap) = 1.000 for every cut. Therefore ap and the diameters are in mm.
* Measured Ra is on median 1.11 × the kinematic value f²/(32·rε)·1000 with rε = 0.8 mm. Therefore f is in mm/rev and Ra is in µm.
* Vc lies between 100 and 300, which is the usual range for m/min.
* The units of `d_error`, `T_error`, `T_length` and the signal channels are not documented. These columns are therefore kept but not used.

| Variable | Meaning | Unit | Range in data | Role |
|---|---|---|---|---|
| material | Workpiece material (20MnCr5, 41Cr4, EN AW-6082) | — | 3 levels, 151 cuts each | input (Ra) |
| hardness_HV | Workpiece hardness | HV | 102 – 300 | (one value per material) |
| vc | Cutting speed | m/min | 100 – 300 (26 levels) | input |
| f | Feed | mm/rev | 0.10 – 0.30 (28 levels) | input |
| ap | Depth of cut | mm | 0.25 – 1.50 (43 levels) | input |
| re | Corner radius | mm | 0.8 (constant) | constant |
| Ra | Arithmetic mean roughness | µm | 0.288 – 3.944 | **measured target** |
| Rz | Mean roughness depth | µm | 2.334 – 18.11 | not used as input (would leak) |
| MRR_mm3_min | Material removal rate | mm³/min | 5 000 – 90 000 | **calculated target** |

The dataset has 295 unique (material, Vc, f, ap) combinations. Some were cut up to 5 times, and these replicates are genuine repeated cuts. There are no missing values and no exact duplicate rows. The three process parameters are practically uncorrelated with each other in the design (|r| ≤ 0.04), which makes their individual effects separable.

## 7. Mathematical Formulation

**Material removal rate (turning).**

  MRR = 1000 · Vc · f · ap  [mm³/min]

Vc is in m/min (×1000 → mm/min), f in mm/rev and ap in mm. The product f·ap is the nominal uncut chip cross-section. This relation applies to turning only.

**Surface roughness.** Ra is the arithmetic mean of the absolute profile deviations z(x) from the mean line over the evaluation length l:

  Ra = (1/l) ∫₀ˡ |z(x)| dx  [µm]

**Kinematic (ideal) roughness** for a tool with corner radius rε, used for comparison only:

  Ra,th ≈ f² / (32 · rε)  (mm; ×1000 for µm)

**Evaluation metrics** on n test samples with true values yᵢ and predictions ŷᵢ:

  R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
  MAE = (1/n) Σ |yᵢ − ŷᵢ|
  MSE = (1/n) Σ (yᵢ − ŷᵢ)²
  RMSE = √MSE

## 8. Methodology

1. **Data preparation.** Parse the HDF5 file, build one row per cut and calculate MRR. Rows with a missing target are excluded only for that target.
2. **Input features.**
   * MRR model: Vc, f, ap.
   * Ra model: Vc, f, ap and material, one-hot encoded.
   * Rz is excluded because it is a second roughness measurement taken after the cut, and using it would leak the target.
3. **Train/test split.** 80 % / 20 % with random seed 42. The default is a **grouped split**: all repeated cuts of one parameter setting fall on the same side. The test score therefore measures generalisation to *unseen* settings. A plain random split is available in the application for comparison.
4. **Leakage-free preprocessing.** Each algorithm is a scikit-learn `Pipeline` with these steps:
   * median imputation;
   * standardisation, for linear regression and SVR only;
   * one-hot encoding of categorical inputs;
   * the estimator itself. For SVR and linear regression the target is also standardised through `TransformedTargetRegressor`.

   All steps are fitted only on the training folds.
5. **Hyper-parameter tuning.** `GridSearchCV` with 5-fold cross-validation inside the training set. GroupKFold is used for the grouped split. The scoring metric is RMSE.
6. **Model selection.** For each target, the algorithm with the **lowest mean cross-validated RMSE** is selected. The hold-out test set is used only once, to report the final metrics, so the test estimate is not biased by model selection.
7. **Interpretation.** Permutation importance on the test set, and a multiple linear regression on standardised inputs for effect sizes.

## 9. Machine-Learning Algorithms

| Algorithm | Principle | Tuned hyper-parameters (grid) |
|---|---|---|
| Linear Regression | Ordinary least squares, y = β₀ + Σβⱼxⱼ | — |
| Decision Tree | Recursive binary splits minimising squared error (CART) | max_depth {3, 5, 8, None}, min_samples_leaf {1, 3, 5} |
| Random Forest | Average of 300 bootstrap trees with random feature subsets | max_depth {None, 8}, min_samples_leaf {1, 3}, max_features {1.0, 0.6} |
| Gradient Boosting | Trees fitted sequentially to the residuals (squared-error loss) | n_estimators {150, 400}, learning_rate {0.03, 0.1}, max_depth {2, 3} |
| SVR (RBF) | ε-insensitive regression in an RBF kernel space | C {1, 10, 100}, ε {0.01, 0.1}, γ {scale, 0.1} |

## 10. System Architecture

```
HDF5 (982 MB) ──scripts/build_dataset.py──► data/dataset.csv
                                                │
          src/data_loader.py  ◄─────────────────┘   (load, validate, map columns, metadata, references)
          src/mrr_calculator.py                     (MRR equation, input validation)
          src/preprocessing.py                      (targets, default features)
          src/model_training.py                     (pipelines, grouped split, GridSearchCV, selection)
          src/evaluation.py                         (R², MAE, MSE, RMSE, MAPE)
          src/prediction.py                         (model bundles, persistence, single prediction)
          src/visualization.py                      (all Matplotlib/Seaborn figures)
          src/insights.py                           (data-derived observations)
          src/report_generator.py                   (ReportLab PDF)
                                                │
app.py (st.navigation) ─► views/*.py  (8 pages)  ─► models/*.pkl, CSV & PDF downloads
```

## 11. Implementation

The application is written in Python 3 with Streamlit and has eight pages:

1. **Overview.**
2. **Dataset Explorer.** Upload, preview, columns, missing values, duplicates, statistics, correlation heatmap, distributions and design space.
3. **Model Training.** Choose target, features, algorithms, split, number of folds and tuning; view progress, compare models and save or download the model.
4. **MRR Simulator.** Shows the model prediction alongside the analytical value and a sensitivity curve.
5. **Roughness Simulator.** Shows the prediction, ISO 1302 grade, kinematic comparison, nearest real measurements and a sensitivity curve.
6. **Performance Analysis.** Actual vs predicted, residuals, comparison charts, permutation importance, heatmap, metrics and prediction export.
7. **Engineering Insights.** Separates *observed* relationships, computed live from the data, from general *principles*.
8. **Report Generator.** PDF report and CSV exports.

**Robustness.** Inputs are validated against plausible turning ranges. Uploaded CSVs with wrong column names are handled through automatic alias detection plus manual mapping. Missing values are imputed or dropped, datasets with fewer than 20 valid rows are rejected with a message, and missing features raise a clear error. Pages that need a model show an empty-state message with a link instead of failing.

**Testing.** 24 automated tests (`pytest`) cover the equations, the edge cases, every algorithm, persistence and PDF generation. Streamlit `AppTest` runs render every page and press the Train, Predict and Generate-report buttons.

## 12. Results and Discussion

### 12.1 Surface roughness Ra (measured target)

Grouped split: 364 training and 89 test cuts. Features: Vc, f, ap, material.

| Model | CV RMSE (µm) | Train R² | Test R² | Test MAE (µm) | Test MSE (µm²) | Test RMSE (µm) |
|---|---|---|---|---|---|---|
| **SVR (RBF)** ★ | **0.111** ± 0.027 | 0.996 | 0.977 | 0.090 | 0.0236 | 0.154 |
| Gradient Boosting | 0.164 ± 0.063 | 0.994 | 0.977 | 0.087 | 0.0231 | 0.152 |
| Random Forest | 0.166 ± 0.080 | 0.997 | 0.977 | 0.078 | 0.0236 | 0.154 |
| Decision Tree | 0.168 ± 0.066 | 0.999 | 0.957 | 0.088 | 0.0443 | 0.210 |
| Linear Regression | 0.296 ± 0.088 | 0.927 | 0.954 | 0.182 | 0.0467 | 0.216 |

★ selected by the lowest cross-validated RMSE. The SVR's tuned parameters were C = 100, ε = 0.01 and γ = 0.1.

**Discussion.**

* **Selection.** The SVR had clearly the lowest and most stable cross-validation error. On the test set, SVR, gradient boosting and random forest are practically equal (R² 0.977, RMSE 0.152–0.154 µm). With 89 test samples these differences are within statistical noise, so selecting by CV error is the more reliable criterion.
* **Linear regression.** It still reaches R² = 0.954 because the f² dependence is close to linear over 0.1–0.3 mm/rev. Its MAE is twice that of the non-linear models.
* **Decision tree.** It shows the largest gap between train and test (0.999 vs 0.957), which is typical of overfitting by a single tree.
* **Residuals.** They are centred on zero. A few cuts are over-predicted by up to about 0.6 µm.

**Feature importance.** Permutation importance for the selected SVR (mean decrease in test R²):

| Feature | Importance (mean decrease in test R²) |
|---|---|
| f | 1.931 |
| material | 0.065 |
| vc | 0.044 |
| ap | 0.020 |

### 12.2 Material removal rate (calculated target)

Grouped split: 354 training and 99 test cuts. Features: Vc, f, ap.

| Model | CV RMSE (mm³/min) | Train R² | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|---|
| **SVR (RBF)** ★ | **358** | 1.000 | 0.995 | 323 | 932 |
| Gradient Boosting | 3 466 | 0.995 | 0.904 | 3 279 | 3 931 |
| Random Forest | 4 592 | 0.999 | 0.893 | 3 462 | 4 156 |
| Linear Regression | 5 534 | 0.892 | 0.912 | 2 951 | 3 769 |
| Decision Tree | 7 331 | 0.992 | 0.712 | 5 797 | 6 830 |

**Discussion.** MRR is an exact product of the inputs, so these scores do **not** measure experimental predictive accuracy. They show how well each algorithm can represent a smooth multiplicative law.

* **SVR.** It interpolates the smooth surface well (MAPE 1.4 %).
* **Tree-based models.** They approximate the law with piecewise-constant steps and generalise poorly to parameter settings absent from the training data. This is a known limitation of trees, and the grouped split makes it visible.
* **Linear regression.** It cannot represent the product term.

In engineering practice the analytical equation should be used for MRR. The ML model is included because the assignment requires it, and the simulator shows both values side by side.

### 12.3 Engineering observations from the data

* **Feed dominates roughness.** Pearson r(f, Ra) = 0.951 overall, and 0.918, 0.990 and 0.978 for 20MnCr5, 41Cr4 and EN AW-6082 respectively. In a multiple linear regression on standardised inputs (R² = 0.932), one standard deviation of feed raises Ra by 1.03 µm. The same change in Vc lowers Ra by 0.074 µm and in ap by 0.026 µm.
* **Agreement with kinematic theory.** Measured Ra is on median 1.11 × f²/(32·rε), with an interquartile range of 1.01–1.18. The feed-mark geometry therefore explains most of the roughness.
* **Cutting speed and depth of cut.** Both have a negligible linear correlation with Ra (r = −0.065 and −0.062). The small negative Vc effect is consistent with the general principle that higher speed reduces built-up edge, but it is weak in this data.
* **Material.** Mean Ra is 1.62 µm for EN AW-6082, 1.98 µm for 41Cr4 and 2.00 µm for 20MnCr5.
* **MRR.** It spans 5 000–90 000 mm³/min. Because MRR is proportional to Vc, f and ap alike, while Ra depends mainly on f, productivity can be raised with less roughness penalty by increasing Vc (within tool-life limits) and ap rather than f.

## 13. Limitations

* **Single tool, three bars.** All data come from one insert geometry (DNMG150608, rε = 0.8 mm) and one workpiece per material. The material effect is therefore confounded with the individual bar, and the models do not transfer to other tools, nose radii or materials.
* **Training range only.** Predictions are valid within Vc 100–300 m/min, f 0.10–0.30 mm/rev and ap 0.25–1.5 mm. The simulators warn when an input is outside this range.
* **MRR is nominal.** It is calculated from set values, not measured, so it does not include the real engagement or any interruptions.
* **Tool wear not modelled.** The parameter-only models do not represent tool wear progression. The cut sequence index and the process signals are available as optional features but are not used in the simulators.
* **Undocumented units.** The units of some stored quantities (`d_error`, `T_error`, `T_length`, signal channels) are not documented and were not used.
* **Small test set.** 89–99 samples per target means that differences of about 0.001 in test R² between the top models are not significant.

## 14. Conclusion

Five regression algorithms were trained and compared on a genuine, published longitudinal-turning dataset of 453 cuts. Leakage-free pipelines, cross-validated tuning and a grouped hold-out test were used throughout.

* **Ra.** The RBF-kernel SVR was selected by cross-validation and predicted unseen parameter settings with R² = 0.977, MAE = 0.090 µm and RMSE = 0.154 µm. Random forest and gradient boosting performed equally well on the test set.
* **MRR.** It was calculated from the turning equation and reproduced by the SVR with R² = 0.995. For MRR the analytical equation remains the method of choice.
* **Physical behaviour.** The data confirm that feed is the governing parameter for surface roughness, in close agreement with kinematic theory (median ratio 1.11). Cutting speed and depth of cut have only minor effects in the tested range.

The resulting Streamlit application makes the full workflow reproducible and demonstrable, from raw data to trained models, simulators and a PDF report.

## 15. References

1. Denkena, B., Bergmann, B., Buhl, H., Handrup, M. (2026). *Multivariate time series dataset for predicting surface roughness and dimensional error in longitudinal turning* [Data set]. LUIS, Leibniz University Hannover. https://doi.org/10.25835/9xffqwnc
2. Benardos, P.G., Vosniakos, G.-C. (2003). Predicting surface roughness in machining: a review. *International Journal of Machine Tools and Manufacture*, 43(8), 833–844.
3. Özel, T., Karpat, Y. (2005). Predictive modeling of surface roughness and tool wear in hard turning using regression and neural networks. *International Journal of Machine Tools and Manufacture*, 45(4–5), 467–479.
4. Kalpakjian, S., Schmid, S.R. (2014). *Manufacturing Engineering and Technology*, 7th ed. Pearson.
5. Shaw, M.C. (2005). *Metal Cutting Principles*, 2nd ed. Oxford University Press.
6. ISO 4287:1997. Geometrical Product Specifications (GPS) — Surface texture: Profile method — Terms, definitions and surface texture parameters.
7. Breiman, L., Friedman, J.H., Olshen, R.A., Stone, C.J. (1984). *Classification and Regression Trees*. Wadsworth.
8. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
9. Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
10. Drucker, H., Burges, C.J.C., Kaufman, L., Smola, A., Vapnik, V. (1997). Support vector regression machines. *Advances in Neural Information Processing Systems 9*, 155–161.
11. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
12. Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.
13. Waskom, M.L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
