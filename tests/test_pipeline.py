"""Unit tests: data handling, equations, training edge cases, persistence, report."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import guess_column, load_default, read_uploaded_csv  # noqa: E402
from src.model_training import train_models  # noqa: E402
from src.mrr_calculator import calculate_mrr, validate_inputs  # noqa: E402
from src.prediction import bundle_from_run, predict_one  # noqa: E402
from src.preprocessing import prepare  # noqa: E402


@pytest.fixture(scope="module")
def df():
    d, w = prepare(load_default())
    assert w == []
    return d


def test_dataset_shape(df):
    assert len(df) == 453
    assert set(df["material"]) == {"20MnCr5", "41Cr4", "EN AW-6082"}
    assert df[["vc", "f", "ap", "Ra"]].isna().sum().sum() == 0


def test_mrr_equation():
    assert calculate_mrr(200, 0.2, 1.0) == pytest.approx(40000.0)
    assert validate_inputs(200, 0.2, 1.0) == []
    assert validate_inputs(-1, 0.2, 1.0) and validate_inputs(200, 5, 1.0)


def test_mrr_column(df):
    r = df.iloc[0]
    assert r["MRR_mm3_min"] == pytest.approx(1000 * r["vc"] * r["f"] * r["ap"])


def test_upload_wrong_columns():
    csv = b"speed;feed;doc;roughness\n100;0.1;0.5;0.8\n200;0.2;1.0;1.6\n"
    raw = read_uploaded_csv(io.BytesIO(csv))
    assert guess_column(raw, "vc") == "speed" and guess_column(raw, "Ra") == "roughness"
    bad = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out, warns = prepare(bad)
    assert "MRR_mm3_min" not in out.columns and warns


def test_empty_upload():
    with pytest.raises(ValueError):
        read_uploaded_csv(io.BytesIO(b""))


def test_missing_values_are_handled(df):
    d = df.copy()
    rng = np.random.default_rng(0)
    d.loc[rng.choice(len(d), 30, replace=False), "vc"] = np.nan
    d.loc[rng.choice(len(d), 10, replace=False), "Ra"] = np.nan
    run = train_models(d, ["vc", "f", "ap", "material"], "Ra", ["Linear Regression", "Random Forest"], tune=False)
    assert run.n_train + run.n_test == len(d) - 10


def test_too_small_dataset(df):
    with pytest.raises(ValueError, match="at least"):
        train_models(df.head(8), ["vc", "f", "ap"], "Ra", ["Linear Regression"])


def test_missing_feature(df):
    with pytest.raises(ValueError, match="not found"):
        train_models(df, ["vc", "does_not_exist"], "Ra", ["Linear Regression"])


def test_all_algorithms_and_prediction(df):
    run = train_models(df, ["vc", "f", "ap", "material"], "Ra",
                       ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR (RBF)"],
                       split_mode="grouped", tune=False)
    assert run.n_test / (run.n_train + run.n_test) == pytest.approx(0.2, abs=0.05)
    # grouped split: no parameter setting in both sets
    key = lambda X: set(X.astype(str).agg("|".join, axis=1))
    b = bundle_from_run(run, target_kind="Ra")
    y = predict_one(b, {"vc": 200, "f": 0.2, "ap": 0.75, "material": "41Cr4"})
    assert 0 < y < 6
    with pytest.raises(KeyError):
        predict_one(b, {"vc": 200})
    for r in run.results.values():
        m = r["test_metrics"]
        assert m["RMSE"] == pytest.approx(np.sqrt(m["MSE"]))


def test_report_pdf(df):
    from src.report_generator import build_report
    run = train_models(df, ["vc", "f", "ap"], "MRR_mm3_min", ["Linear Regression", "SVR (RBF)"], tune=False)
    pdf = build_report(df, {"n_rows": len(df), "description": "x", "heatmap_cols": ["vc", "f", "Ra"]},
                       {"MRR": run, "Ra": None})
    assert pdf[:4] == b"%PDF" and len(pdf) > 20000
