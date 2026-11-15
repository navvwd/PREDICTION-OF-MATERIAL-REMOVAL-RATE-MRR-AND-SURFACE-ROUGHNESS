"""Data-derived engineering observations (every statement is computed, never hard-coded)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PARAM_LABELS = {"vc": "cutting speed Vc", "f": "feed f", "ap": "depth of cut ap"}


def _strength(r: float) -> str:
    a = abs(r)
    if a >= 0.8:
        s = "very strong"
    elif a >= 0.6:
        s = "strong"
    elif a >= 0.4:
        s = "moderate"
    elif a >= 0.2:
        s = "weak"
    else:
        s = "negligible"
    return f"{s} {'positive' if r > 0 else 'negative'}" if a >= 0.2 else s


def correlation_table(df: pd.DataFrame, target: str, params=("vc", "f", "ap"), by: str | None = "material") -> pd.DataFrame:
    rows = []
    groups = [("All data", df)]
    if by and by in df.columns:
        groups += [(str(k), g) for k, g in df.groupby(by)]
    for name, g in groups:
        for p in params:
            if p not in g.columns or g[p].nunique() < 2:
                continue
            d = g[[p, target]].dropna()
            rows.append({"Group": name, "Parameter": p, "n": len(d),
                         "Pearson r": d[p].corr(d[target]),
                         "Spearman ρ": d[p].corr(d[target], method="spearman")})
    return pd.DataFrame(rows)


def ols_effects(df: pd.DataFrame, target: str = "Ra", params=("vc", "f", "ap"), cat: str | None = "material") -> pd.DataFrame:
    """Multiple linear regression on standardised inputs -> comparable effect sizes (per 1 SD)."""
    cols = [p for p in params if p in df.columns and df[p].nunique() > 1]
    d = df[cols + [target] + ([cat] if cat and cat in df.columns else [])].dropna()
    if len(d) < len(cols) + 5:
        return pd.DataFrame()
    X = (d[cols] - d[cols].mean()) / d[cols].std(ddof=0)
    if cat and cat in d.columns and d[cat].nunique() > 1:
        X = pd.concat([X, pd.get_dummies(d[cat], prefix=cat, drop_first=True, dtype=float)], axis=1)
    A = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    coef, *_ = np.linalg.lstsq(A, d[target].to_numpy(float), rcond=None)
    resid = d[target].to_numpy(float) - A @ coef
    r2 = 1 - (resid ** 2).sum() / ((d[target] - d[target].mean()) ** 2).sum()
    out = pd.DataFrame({"Term": ["intercept"] + list(X.columns), "Coefficient": coef})
    out.attrs["R2"] = float(r2)
    return out


def kinematic_ra(f_mm: np.ndarray | float, r_eps_mm: float) -> np.ndarray:
    """Theoretical (kinematic) Ra in µm for a round-nose tool: Ra ≈ f² / (32·rε) [mm] → ×1000 µm."""
    return np.asarray(f_mm, float) ** 2 / (32.0 * r_eps_mm) * 1000.0


def ra_observations(df: pd.DataFrame) -> list[str]:
    if "Ra" not in df.columns:
        return []
    out = []
    ct = correlation_table(df, "Ra")
    allc = ct[ct.Group == "All data"].set_index("Parameter")
    for p in ("f", "vc", "ap"):
        if p in allc.index:
            r = allc.loc[p, "Pearson r"]
            out.append(f"Across all {int(allc.loc[p, 'n'])} cuts, Ra shows a {_strength(r)} correlation with "
                       f"{PARAM_LABELS[p]} (Pearson r = {r:.2f}, Spearman ρ = {allc.loc[p, 'Spearman ρ']:.2f}).")
    eff = ols_effects(df, "Ra")
    if not eff.empty:
        e = eff.set_index("Term")["Coefficient"]
        parts = [f"{PARAM_LABELS[p]}: {e[p]:+.3f} µm" for p in ("f", "vc", "ap") if p in e.index]
        out.append("A multiple linear regression on standardised inputs (R² = "
                   f"{eff.attrs['R2']:.3f}) gives the change in Ra per one standard deviation of each parameter — "
                   + "; ".join(parts) + ".")
    if "material" in df.columns:
        m = df.groupby("material")["Ra"].agg(["mean", "median", "count"]).sort_values("mean")
        out.append("Mean measured Ra by material: " + ", ".join(f"{k} {v['mean']:.2f} µm (n = {int(v['count'])})"
                                                               for k, v in m.iterrows()) + ".")
    if "re" in df.columns and df["re"].nunique() == 1 and "f" in df.columns:
        r_eps = float(df["re"].iloc[0])
        ratio = df["Ra"] / kinematic_ra(df["f"], r_eps)
        out.append(f"Measured Ra is on median {ratio.median():.2f}× the kinematic value f²/(32·rε) with rε = {r_eps:g} mm "
                   f"(interquartile range {ratio.quantile(.25):.2f}–{ratio.quantile(.75):.2f}), i.e. the feed-mark geometry "
                   "explains most of the roughness, with additional contributions from non-ideal effects.")
    return out


def mrr_observations(df: pd.DataFrame) -> list[str]:
    if "MRR_mm3_min" not in df.columns:
        return []
    s = df["MRR_mm3_min"].dropna()
    out = [f"Calculated MRR in the dataset ranges from {s.min():,.0f} to {s.max():,.0f} mm³/min "
           f"(median {s.median():,.0f} mm³/min)."]
    ct = correlation_table(df, "MRR_mm3_min", by=None).set_index("Parameter")
    parts = [f"{PARAM_LABELS[p]} r = {ct.loc[p, 'Pearson r']:.2f}" for p in ("vc", "f", "ap") if p in ct.index]
    if parts:
        out.append("Correlation of calculated MRR with each parameter over the experimental design: " + ", ".join(parts) +
                   ". These values reflect both the equation and how the parameters were combined in the experiments; "
                   "the equation itself is exactly proportional to each parameter.")
    return out


def algorithm_observations(runs: dict) -> list[str]:
    out = []
    for kind, run in runs.items():
        if run is None:
            continue
        unit = "mm³/min" if kind == "MRR" else "µm"
        ranked = sorted(run.results.items(), key=lambda kv: kv[1]["cv_rmse_mean"])
        best_name, best = ranked[0]
        worst_name, worst = ranked[-1]
        out.append(f"{kind}: {best_name} had the lowest cross-validated RMSE ({best['cv_rmse_mean']:.4g} {unit}) and reached "
                   f"test R² = {best['test_metrics']['R2']:.3f}, RMSE = {best['test_metrics']['RMSE']:.4g} {unit}; "
                   f"{worst_name} was weakest (CV RMSE {worst['cv_rmse_mean']:.4g} {unit}, test R² = {worst['test_metrics']['R2']:.3f}).")
    return out


def design_summary(df: pd.DataFrame) -> str:
    parts = []
    if "material" in df.columns:
        parts.append(f"{len(df)} cuts on {df['material'].nunique()} workpiece materials ({', '.join(sorted(df['material'].astype(str).unique()))})")
    else:
        parts.append(f"{len(df)} records")
    for p, u in (("vc", "m/min"), ("f", "mm/rev"), ("ap", "mm")):
        if p in df.columns:
            parts.append(f"{PARAM_LABELS[p]} {df[p].min():g}–{df[p].max():g} {u} ({df[p].nunique()} levels)")
    if {"vc", "f", "ap"}.issubset(df.columns):
        key = ["material", "vc", "f", "ap"] if "material" in df.columns else ["vc", "f", "ap"]
        parts.append(f"{df[key].drop_duplicates().shape[0]} unique parameter combinations")
    if "Ra" in df.columns:
        parts.append(f"measured Ra {df['Ra'].min():.3f}–{df['Ra'].max():.3f} µm")
    return "; ".join(parts) + "."
