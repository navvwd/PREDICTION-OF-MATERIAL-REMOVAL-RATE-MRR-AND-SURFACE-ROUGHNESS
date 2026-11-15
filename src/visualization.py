"""Matplotlib / Seaborn figures in a consistent engineering style.

Every function takes real data (DataFrames or model outputs) and returns a
matplotlib Figure, so the same figure can be shown in Streamlit and embedded
in the PDF report.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NAVY = "#12263f"
BLUE = "#2563eb"
STEEL = "#64748b"
GRID = "#e2e8f0"
PALETTE = ["#2563eb", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#475569"]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#94a3b8",
    "axes.labelcolor": NAVY,
    "axes.titlecolor": NAVY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def correlation_heatmap(df: pd.DataFrame, title: str = "Pearson correlation matrix"):
    num = df.select_dtypes(include="number")
    num = num.loc[:, num.nunique() > 1]
    n = max(len(num.columns), 2)
    fig, ax = plt.subplots(figsize=(min(1.0 + 0.75 * n, 11), min(0.8 + 0.65 * n, 9)))
    if num.shape[1] < 2:
        ax.text(0.5, 0.5, "Need ≥ 2 numeric columns", ha="center", va="center"); ax.axis("off")
        return fig
    corr = num.corr()
    sns.heatmap(corr, annot=n <= 14, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title(title)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return fig


def distributions(df: pd.DataFrame, cols: list, bins: int = 25):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        fig, ax = plt.subplots(figsize=(6, 2)); ax.axis("off"); ax.text(0.5, 0.5, "No columns selected", ha="center")
        return fig
    ncol = min(3, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 2.9 * nrow), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            s = s.dropna()
            nun = s.nunique()
            if nun <= 12:
                vc = s.value_counts().sort_index()
                ax.bar([f"{v:g}" for v in vc.index], vc.values, color=BLUE, edgecolor=NAVY, linewidth=0.5)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            else:
                ax.hist(s, bins=bins, color=BLUE, edgecolor="white", alpha=0.9)
        else:
            vc = s.astype(str).value_counts()
            ax.bar(vc.index, vc.values, color=STEEL, edgecolor=NAVY, linewidth=0.5)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(c, fontsize=9.5)
        ax.set_ylabel("count")
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def actual_vs_predicted(y_true, y_pred, title: str, unit: str, metrics: dict | None = None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(y_true, y_pred, s=26, color=BLUE, alpha=0.75, edgecolor=NAVY, linewidth=0.4, label="test samples")
    lo = float(min(y_true.min(), y_pred.min())); hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.04 * (hi - lo if hi > lo else 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color=NAVY, lw=1, label="ideal (y = x)")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"Actual [{unit}]"); ax.set_ylabel(f"Predicted [{unit}]")
    ax.set_title(title)
    if metrics:
        ax.text(0.03, 0.97, f"R² = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.4g}\nMAE = {metrics['MAE']:.4g}\nn = {len(y_true)}",
                transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def residual_plots(y_true, y_pred, title: str, unit: str):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    res = y_true - y_pred
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    a1.hist(res, bins=min(25, max(5, len(res) // 3)), color=BLUE, edgecolor="white")
    a1.axvline(0, color=NAVY, ls="--", lw=1)
    a1.set_xlabel(f"Residual (actual − predicted) [{unit}]"); a1.set_ylabel("count")
    a1.set_title("Residual distribution")
    a2.scatter(y_pred, res, s=20, color=BLUE, alpha=0.75, edgecolor=NAVY, linewidth=0.3)
    a2.axhline(0, color=NAVY, ls="--", lw=1)
    a2.set_xlabel(f"Predicted [{unit}]"); a2.set_ylabel(f"Residual [{unit}]")
    a2.set_title("Residuals vs predicted")
    fig.suptitle(title, fontsize=11, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig


def model_comparison(table: pd.DataFrame, title: str):
    if table is None or table.empty:
        fig, ax = plt.subplots(figsize=(6, 2)); ax.axis("off"); return fig
    t = table.copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = [BLUE if i == 0 else "#94a3b8" for i in range(len(t))]
    a1.barh(t["Model"], t["Test R2"], color=colors, edgecolor=NAVY, linewidth=0.5)
    a1.invert_yaxis(); a1.set_xlabel("Test R²"); a1.set_title("Coefficient of determination (test)")
    xmin = min(0.0, float(t["Test R2"].min()))
    a1.set_xlim(xmin, 1.0)
    for i, v in enumerate(t["Test R2"]):
        a1.text(max(v, xmin) + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    a2.barh(t["Model"], t["Test RMSE"], color=colors, edgecolor=NAVY, linewidth=0.5,
            xerr=None)
    a2.scatter(t["CV RMSE (mean)"], range(len(t)), marker="D", color="#b45309", zorder=3, s=22, label="CV RMSE (train)")
    a2.invert_yaxis(); a2.set_xlabel("RMSE"); a2.set_title("Error (test bar, CV marker)")
    a2.legend(fontsize=8, loc="lower right")
    a2.set_yticklabels([])
    fig.suptitle(title, fontsize=11, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig


def feature_importance_plot(imp: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(6.2, 0.5 + 0.42 * max(len(imp), 2)))
    if imp is None or imp.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "Not available", ha="center"); return fig
    ax.barh(imp["Feature"], imp["Importance"], xerr=imp["Std"], color=BLUE, edgecolor=NAVY,
            linewidth=0.5, error_kw=dict(ecolor=STEEL, lw=1, capsize=2))
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in test R² when feature is permuted")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def effect_plot(df: pd.DataFrame, x: str, y: str, hue: str | None = None, title: str = "",
                xlabel: str | None = None, ylabel: str | None = None):
    """Mean ± std of *y* for each level of *x* (optionally by *hue*) from the real dataset."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    d = df[[c for c in [x, y, hue] if c]].dropna()
    groups = [(None, d)] if not hue else list(d.groupby(hue))
    for i, (h, g) in enumerate(groups):
        s = g.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(s[x], s["mean"], yerr=s["std"].fillna(0), marker="o", ms=5, capsize=3, lw=1.6,
                    color=PALETTE[i % len(PALETTE)], label=str(h) if h is not None else None)
        ax.scatter(g[x], g[y], s=8, alpha=0.18, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} vs {x}")
    if hue:
        ax.legend(title=hue, fontsize=8, title_fontsize=8)
    fig.tight_layout()
    return fig


def response_curve(xs, ys, x_label: str, y_label: str, point=None, title: str = "", analytic=None):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(xs, ys, color=BLUE, lw=2, label="model prediction")
    if analytic is not None:
        ax.plot(xs, analytic, color=NAVY, ls="--", lw=1.2, label="analytical equation")
    if point is not None:
        ax.scatter([point[0]], [point[1]], color="#b45309", zorder=4, s=50, label="selected point")
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
