"""Regression metrics computed from real test-set predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.nanmean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAPE_%": float(mape),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    """results: {model_name: TrainedModelResult-like dict}. Returns tidy table sorted by CV RMSE."""
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        row.update({f"Test {k}": v for k, v in r["test_metrics"].items()})
        row.update({f"Train {k}": v for k, v in r["train_metrics"].items() if k in ("R2", "RMSE")})
        row["CV RMSE (mean)"] = r.get("cv_rmse_mean", np.nan)
        row["CV RMSE (std)"] = r.get("cv_rmse_std", np.nan)
        row["Fit time (s)"] = r.get("fit_time", np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("CV RMSE (mean)").reset_index(drop=True)
    return df
