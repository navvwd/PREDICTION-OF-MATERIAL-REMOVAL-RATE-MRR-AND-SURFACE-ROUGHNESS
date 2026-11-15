"""Model persistence and single-point prediction."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILES = {"MRR": MODEL_DIR / "mrr_model.pkl", "Ra": MODEL_DIR / "roughness_model.pkl"}


def bundle_from_run(run, model_name: str | None = None, target_kind: str = "Ra", dataset_name: str = "") -> dict:
    name = model_name or run.best_model_name
    r = run.results[name]
    return {
        "target_kind": target_kind,
        "target": run.target,
        "model_name": name,
        "estimator": r["estimator"],
        "features": run.features,
        "numeric_features": run.numeric_features,
        "categorical_features": run.categorical_features,
        "feature_ranges": run.feature_ranges,
        "categories": run.categories,
        "test_metrics": r["test_metrics"],
        "cv_rmse_mean": r["cv_rmse_mean"],
        "best_params": r["best_params"],
        "split_mode": run.split_mode,
        "n_train": run.n_train,
        "n_test": run.n_test,
        "dataset": dataset_name,
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run": run,  # full comparison (all algorithms, test predictions) for the analysis pages
    }


def save_bundle(bundle: dict, path: Path | None = None) -> Path:
    path = Path(path or MODEL_FILES[bundle["target_kind"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


LOAD_ERRORS: dict = {}


def load_bundle(target_kind: str) -> dict | None:
    """Load a saved model; on failure the reason is kept in LOAD_ERRORS (e.g. library-version mismatch)."""
    path = MODEL_FILES[target_kind]
    if not path.exists():
        LOAD_ERRORS[target_kind] = f"{path.name} not found"
        return None
    try:
        b = joblib.load(path)
        if isinstance(b, dict) and "estimator" in b:
            LOAD_ERRORS.pop(target_kind, None)
            return b
        LOAD_ERRORS[target_kind] = f"{path.name} is not a model bundle"
    except Exception as e:
        LOAD_ERRORS[target_kind] = f"{path.name}: {type(e).__name__}: {e}"
    return None


def predict_one(bundle: dict, inputs: dict) -> float:
    missing = [f for f in bundle["features"] if f not in inputs]
    if missing:
        raise KeyError(f"Missing input(s) for this model: {missing}")
    row = {}
    for f in bundle["features"]:
        v = inputs[f]
        row[f] = str(v) if f in bundle["categorical_features"] else float(v)
    X = pd.DataFrame([row], columns=bundle["features"])
    y = float(bundle["estimator"].predict(X)[0])
    if not np.isfinite(y):
        raise ValueError("Model returned a non-finite prediction.")
    return y


def extrapolation_warnings(bundle: dict, inputs: dict) -> list[str]:
    out = []
    for f, (lo, hi, _) in bundle.get("feature_ranges", {}).items():
        v = inputs.get(f)
        if v is not None and not (lo <= float(v) <= hi):
            out.append(f"{f} = {v} lies outside the training range [{lo:g}, {hi:g}] — the prediction is an extrapolation and is less reliable.")
    return out
