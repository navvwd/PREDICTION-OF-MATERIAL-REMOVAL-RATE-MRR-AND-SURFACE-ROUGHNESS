"""Training of the five regression algorithms with leakage-free pipelines.

Protocol (identical for MRR and Ra):
  1. Hold-out split 80 % train / 20 % test with a fixed random seed.
     Optionally *grouped* by unique parameter combination so that repeated
     cuts with identical settings never appear in both train and test.
  2. Every algorithm is a sklearn Pipeline: imputer -> (scaler) / one-hot -> model.
     All preprocessing is fitted inside the pipeline on training folds only.
  3. Hyper-parameters are tuned by GridSearchCV (k-fold CV on the training set).
  4. The best algorithm is selected by the lowest mean CV RMSE on the training
     set (not by the test score, which would bias the test estimate).
  5. Test metrics are then reported once on the untouched 20 % hold-out set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .evaluation import regression_metrics

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_SAMPLES = 20

ALGORITHMS = {
    "Linear Regression": {
        "estimator": lambda: LinearRegression(),
        "scale": True,
        "grid": {},
    },
    "Decision Tree": {
        "estimator": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
        "scale": False,
        "grid": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 3, 5]},
    },
    "Random Forest": {
        "estimator": lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "scale": False,
        "grid": {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 3], "model__max_features": [1.0, 0.6]},
    },
    "Gradient Boosting": {
        "estimator": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
        "scale": False,
        "grid": {"model__n_estimators": [150, 400], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]},
    },
    "SVR (RBF)": {
        "estimator": lambda: SVR(kernel="rbf"),
        "scale": True,
        "grid": {"model__C": [1, 10, 100], "model__epsilon": [0.01, 0.1], "model__gamma": ["scale", 0.1]},
    },
}


@dataclass
class TrainingRun:
    target: str
    features: list
    numeric_features: list
    categorical_features: list
    split_mode: str
    n_train: int
    n_test: int
    results: dict = field(default_factory=dict)  # name -> dict
    best_model_name: str | None = None
    feature_ranges: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def best(self):
        return self.results.get(self.best_model_name) if self.best_model_name else None


def _build_pipeline(algo: str, numeric: list, categorical: list, scale_target: bool) -> Pipeline:
    spec = ALGORITHMS[algo]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if spec["scale"]:
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    model = spec["estimator"]()
    if scale_target and spec["scale"]:
        # SVR / linear models behave better on a standardised target
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    return Pipeline([("pre", pre), ("model", model)])


def _grid_for(algo: str, scale_target: bool) -> dict:
    grid = ALGORITHMS[algo]["grid"]
    if scale_target and ALGORITHMS[algo]["scale"]:
        grid = {k.replace("model__", "model__regressor__"): v for k, v in grid.items()}
    return grid


def validate_training_data(df: pd.DataFrame, features: list, target: str) -> list[str]:
    errs = []
    if df is None or df.empty:
        return ["No dataset is loaded."]
    if not features:
        errs.append("Select at least one input feature.")
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        errs.append(f"Column(s) not found in dataset: {missing}")
        return errs
    if target in features:
        errs.append("The target column cannot also be used as an input feature.")
    y = pd.to_numeric(df[target], errors="coerce")
    n_valid = int(y.notna().sum())
    if n_valid < MIN_SAMPLES:
        errs.append(f"Only {n_valid} rows have a valid target value; at least {MIN_SAMPLES} are required.")
    return errs


def train_models(df: pd.DataFrame, features: list, target: str, algorithms: list,
                 split_mode: str = "random", cv_folds: int = 5, tune: bool = True,
                 progress_cb=None) -> TrainingRun:
    """Train the selected algorithms. ``split_mode`` is 'random' or 'grouped'."""
    errs = validate_training_data(df, features, target)
    if errs:
        raise ValueError(" ".join(errs))

    data = df[features + [target]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target]).reset_index(drop=True)

    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(data[c])]
    numeric = [c for c in features if c not in categorical]
    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    X, y = data[features], data[target].to_numpy(dtype=float)
    groups = X.astype(str).agg("|".join, axis=1).to_numpy()

    warnings = []
    if split_mode == "grouped" and len(np.unique(groups)) >= 10:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(gss.split(X, y, groups))
    else:
        if split_mode == "grouped":
            warnings.append("Too few unique parameter combinations for a grouped split; a random split was used.")
            split_mode = "random"
        tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y[tr_idx], y[te_idx]
    g_tr = groups[tr_idx]

    k = int(max(2, min(cv_folds, len(X_tr) // 5)))
    if split_mode == "grouped":
        k = int(min(k, len(np.unique(g_tr))))
        cv = GroupKFold(n_splits=k)
        fit_kw = {"groups": g_tr}
    else:
        cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        fit_kw = {}

    run = TrainingRun(target=target, features=list(features), numeric_features=numeric,
                      categorical_features=categorical, split_mode=split_mode,
                      n_train=len(X_tr), n_test=len(X_te), warnings=warnings)
    for c in numeric:
        col = X_tr[c].dropna()
        run.feature_ranges[c] = (float(col.min()), float(col.max()), float(col.median()))
    for c in categorical:
        run.categories[c] = sorted(X_tr[c].dropna().astype(str).unique().tolist())

    for i, algo in enumerate(algorithms):
        if progress_cb:
            progress_cb(i, len(algorithms), algo)
        t0 = time.perf_counter()
        pipe = _build_pipeline(algo, numeric, categorical, scale_target=True)
        grid = _grid_for(algo, True) if tune else {}
        search = GridSearchCV(pipe, grid or {}, cv=cv, scoring="neg_root_mean_squared_error",
                              n_jobs=-1, refit=True, error_score="raise")
        search.fit(X_tr, y_tr, **fit_kw)
        est = search.best_estimator_
        idx = search.best_index_
        cv_mean = float(-search.cv_results_["mean_test_score"][idx])
        cv_std = float(search.cv_results_["std_test_score"][idx])
        pred_tr, pred_te = est.predict(X_tr), est.predict(X_te)
        run.results[algo] = {
            "estimator": est,
            "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
            "cv_rmse_mean": cv_mean,
            "cv_rmse_std": cv_std,
            "cv_folds": k,
            "train_metrics": regression_metrics(y_tr, pred_tr),
            "test_metrics": regression_metrics(y_te, pred_te),
            "y_test": y_te,
            "y_pred_test": pred_te,
            "X_test": X_te.reset_index(drop=True),
            "fit_time": time.perf_counter() - t0,
        }
    if progress_cb:
        progress_cb(len(algorithms), len(algorithms), "done")

    run.best_model_name = min(run.results, key=lambda n: run.results[n]["cv_rmse_mean"])
    return run


def feature_importance(run: TrainingRun, model_name: str, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance on the hold-out test set (model-agnostic, on original features)."""
    from sklearn.inspection import permutation_importance

    r = run.results[model_name]
    X_te, y_te = r["X_test"], r["y_test"]
    if len(X_te) < 3:
        return pd.DataFrame(columns=["Feature", "Importance", "Std"])
    pi = permutation_importance(r["estimator"], X_te, y_te, n_repeats=n_repeats,
                                random_state=RANDOM_STATE, scoring="r2", n_jobs=-1)
    return (pd.DataFrame({"Feature": run.features, "Importance": pi.importances_mean, "Std": pi.importances_std})
            .sort_values("Importance", ascending=False).reset_index(drop=True))
