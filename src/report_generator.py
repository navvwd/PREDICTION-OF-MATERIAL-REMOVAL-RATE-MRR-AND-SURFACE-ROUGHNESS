"""PDF report generation (ReportLab). All numbers come from the trained runs passed in."""
from __future__ import annotations

import datetime as dt
import io

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import visualization as viz
from .data_loader import DATASET_CITATION, DATASET_DOI, REFERENCES
from .evaluation import metrics_table
from .model_training import feature_importance

import matplotlib
from pathlib import Path
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Unicode-capable fonts (DejaVu ships with matplotlib) so that µ, ², ε, Σ, ≈ … render correctly.
_TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
try:
    for name, fn in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
                     ("DejaVuMono-Bold", "DejaVuSansMono-Bold.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(_TTF / fn)))
    addMapping("DejaVu", 0, 0, "DejaVu"); addMapping("DejaVu", 1, 0, "DejaVu-Bold")
    addMapping("DejaVu", 0, 1, "DejaVu-Oblique"); addMapping("DejaVu", 1, 1, "DejaVu-BoldOblique")
    FONT, FONT_B, FONT_MONO = "DejaVu", "DejaVu-Bold", "DejaVuMono-Bold"
except Exception:  # fall back to built-in fonts
    FONT, FONT_B, FONT_MONO = "Helvetica", "Helvetica-Bold", "Courier-Bold"

NAVY = colors.HexColor("#12263f")
BLUE = colors.HexColor("#2563eb")
GREY = colors.HexColor("#e2e8f0")


def _styles():
    ss = getSampleStyleSheet()
    for st_ in ss.byName.values():
        st_.fontName = FONT_B if "Heading" in st_.name or st_.name == "Title" else FONT
    ss.add(ParagraphStyle("TitleX", parent=ss["Title"], textColor=NAVY, fontSize=20, leading=25, alignment=TA_CENTER))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#475569"), fontSize=10.5, leading=14))
    ss.add(ParagraphStyle("H1x", parent=ss["Heading1"], textColor=NAVY, fontSize=14, spaceBefore=12, spaceAfter=6))
    ss.add(ParagraphStyle("H2x", parent=ss["Heading2"], textColor=NAVY, fontSize=11.5, spaceBefore=8, spaceAfter=4))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.8, leading=14, alignment=TA_JUSTIFY, spaceAfter=5))
    ss.add(ParagraphStyle("Eq", parent=ss["Normal"], fontName=FONT_MONO, fontSize=10.5, alignment=TA_CENTER,
                          textColor=NAVY, spaceBefore=4, spaceAfter=6))
    ss.add(ParagraphStyle("Cap", parent=ss["Normal"], fontSize=8.5, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=8))
    ss.add(ParagraphStyle("Ref", parent=ss["Normal"], fontSize=8.8, leading=12, leftIndent=14, firstLineIndent=-14, spaceAfter=3))
    return ss


def _fig(fig, width_cm=15.5):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    return img


def _table(df: pd.DataFrame, col_widths=None, font=7.8):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTNAME", (0, 0), (-1, 0), FONT_B),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    return t


def _fmt_table(run) -> pd.DataFrame:
    t = metrics_table(run.results)
    out = pd.DataFrame({
        "Model": t["Model"],
        "CV RMSE": t["CV RMSE (mean)"].map(lambda v: f"{v:.4g}"),
        "Test R²": t["Test R2"].map(lambda v: f"{v:.4f}"),
        "Test MAE": t["Test MAE"].map(lambda v: f"{v:.4g}"),
        "Test MSE": t["Test MSE"].map(lambda v: f"{v:.4g}"),
        "Test RMSE": t["Test RMSE"].map(lambda v: f"{v:.4g}"),
    })
    out.loc[out["Model"] == run.best_model_name, "Model"] = run.best_model_name + " *"
    return out


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(2 * cm, 1.2 * cm, "MEDX 55 Advanced Machining Processes — Simulation Based Project No. 8")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report(df: pd.DataFrame, dataset_info: dict, runs: dict, authors: str = "", institution: str = "") -> bytes:
    """runs: {'MRR': TrainingRun | None, 'Ra': TrainingRun | None}. Returns PDF bytes."""
    ss = _styles()
    P = lambda t, s="Body": Paragraph(t, ss[s])
    story = []
    mrr_run, ra_run = runs.get("MRR"), runs.get("Ra")

    # ---------- title ----------
    story += [Spacer(1, 2.2 * cm),
              P("Machine-Learning-Based Prediction of Material Removal Rate and Surface Roughness in Longitudinal Turning", "TitleX"),
              Spacer(1, 0.4 * cm),
              P("Advanced Machining Processes (MEDX 55) — Project Based Learning", "Sub"),
              P("Simulation Based Project No. 08", "Sub"), Spacer(1, 0.6 * cm)]
    if authors:
        story.append(P(authors, "Sub"))
    if institution:
        story.append(P(institution, "Sub"))
    story += [P(f"Generated {dt.date.today():%d %B %Y}", "Sub"), Spacer(1, 1.2 * cm)]

    # ---------- abstract ----------
    n = dataset_info.get("n_rows", len(df))
    abs_parts = [f"This report develops and compares regression models that predict the arithmetic mean surface roughness "
                 f"(R<sub>a</sub>) and the material removal rate (MRR) of longitudinal turning from the process parameters "
                 f"cutting speed, feed and depth of cut. The models are trained on the published experimental dataset of "
                 f"Denkena et al. (Leibniz University Hannover, DOI {DATASET_DOI}), from which {n} individual cuts were "
                 f"extracted. R<sub>a</sub> is the experimentally measured target; MRR is not measured in the dataset and is "
                 f"therefore calculated with the turning equation MRR = 1000·V<sub>c</sub>·f·a<sub>p</sub>. Five algorithms "
                 f"(linear regression, decision tree, random forest, gradient boosting and support vector regression) were "
                 f"tuned by cross-validation on an 80 % training split and evaluated on a 20 % hold-out set."]
    for kind, run, unit in (("R<sub>a</sub>", ra_run, "µm"), ("MRR", mrr_run, "mm³/min")):
        if run:
            b = run.best
            abs_parts.append(f"For {kind}, the best model selected by cross-validation was <b>{run.best_model_name}</b> "
                             f"with a test R² of {b['test_metrics']['R2']:.3f} and RMSE of {b['test_metrics']['RMSE']:.4g} {unit}.")
    story += [P("Abstract", "H1x"), P(" ".join(abs_parts))]
    story.append(PageBreak())

    # ---------- 1 introduction ----------
    story += [P("1. Introduction", "H1x"),
              P("Turning is the most widely used process for producing rotationally symmetric parts. Two of its key "
                "performance indicators are productivity, expressed as the material removal rate, and surface quality, most "
                "commonly expressed as the arithmetic mean roughness R<sub>a</sub>. Selecting cutting parameters that satisfy a "
                "roughness specification at the highest possible removal rate is a classical machining optimisation problem. "
                "Analytical models (e.g. the kinematic roughness R<sub>th</sub> ≈ f²/(8r<sub>ε</sub>)) describe only the ideal "
                "geometric profile and ignore tool wear, vibrations, material side flow and built-up edge. Data-driven "
                "regression models learned from experiments can capture such effects within the experimental domain."),
              P("2. Objectives", "H1x"),
              P("(i) Predict MRR and R<sub>a</sub> from machining parameters; (ii) compare five regression algorithms with "
                "R², MAE, MSE and RMSE; (iii) visualise actual vs. predicted results and residuals; (iv) provide an "
                "interactive simulator; (v) interpret the effect of cutting parameters on MRR and R<sub>a</sub>.")]

    # ---------- dataset ----------
    story += [P("3. Dataset Information", "H1x"),
              P(f"<b>Source:</b> {DATASET_CITATION}"),
              P(dataset_info.get("description", "")),
              ]
    if dataset_info.get("columns_table") is not None:
        story += [_table(dataset_info["columns_table"], font=7.5), P("Table 1 — Variables used in this project.", "Cap")]
    if dataset_info.get("design_text"):
        story.append(P(dataset_info["design_text"]))

    # ---------- equations ----------
    story += [P("4. Mathematical Formulation", "H1x"),
              P("Material removal rate in longitudinal turning (cutting speed × uncut chip cross-section):"),
              P("MRR = 1000 × Vc × f × ap   [mm³/min]", "Eq"),
              P("with V<sub>c</sub> in m/min, f in mm/rev and a<sub>p</sub> in mm (factor 1000 converts m to mm). "
                "<b>MRR is a calculated target</b> — the dataset contains no measured MRR. Surface roughness R<sub>a</sub> "
                "(µm) is the arithmetic mean deviation of the measured profile, R<sub>a</sub> = (1/l)∫|z(x)| dx."),
              P("Evaluation metrics on n test samples:"),
              P("R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²", "Eq"),
              P("MAE = (1/n) Σ|yᵢ − ŷᵢ|     MSE = (1/n) Σ(yᵢ − ŷᵢ)²     RMSE = √MSE", "Eq")]

    # ---------- methodology ----------
    split_txt = ra_run.split_mode if ra_run else (mrr_run.split_mode if mrr_run else "random")
    story += [P("5. Methodology", "H1x"),
              P("The raw HDF5 database was parsed and one record per cut was extracted with its nominal process parameters, "
                "workpiece material and measured R<sub>a</sub>. The original measurements were not modified. Rows with a "
                "missing target are excluded from training of that target only. The data were split 80/20 with random seed 42 "
                f"(split mode: <i>{split_txt}</i>; in grouped mode all repetitions of a parameter combination fall on the same "
                "side of the split, so the test score measures generalisation to unseen settings). All preprocessing "
                "(median imputation, standardisation for scale-sensitive models, one-hot encoding of the material) is "
                "embedded in scikit-learn pipelines and fitted only on training folds, preventing data leakage. "
                "Hyper-parameters were tuned with grid search and k-fold cross-validation on the training set, and the "
                "final model for each target was selected by the lowest mean cross-validated RMSE — the hold-out test set "
                "was used only once for reporting."),
              P("6. Machine-Learning Algorithms", "H1x"),
              P("<b>Linear Regression</b> — ordinary least squares, y = β₀ + Σβⱼxⱼ; interpretable baseline. "
                "<b>Decision Tree</b> — recursive binary partitioning minimising squared error. "
                "<b>Random Forest</b> — bootstrap-aggregated ensemble of decorrelated trees (Breiman, 2001). "
                "<b>Gradient Boosting</b> — additive ensemble of shallow trees fitted sequentially to residuals (Friedman, 2001). "
                "<b>Support Vector Regression</b> — ε-insensitive regression with an RBF kernel on standardised inputs.")]

    # ---------- results ----------
    story += [PageBreak(), P("7. Results", "H1x")]
    fig_no = 1
    for kind, run, unit, label in (("Ra", ra_run, "µm", "Surface roughness R<sub>a</sub>"),
                                   ("MRR", mrr_run, "mm³/min", "Material removal rate")):
        if run is None:
            story.append(P(f"<i>{label}: no model was trained in this session, so no results are reported.</i>"))
            continue
        story += [P(f"7.{1 if kind == 'Ra' else 2} {label}", "H2x"),
                  P(f"Features: {', '.join(run.features)}. Training samples: {run.n_train}; test samples: {run.n_test}. "
                    f"Best model (lowest CV RMSE): <b>{run.best_model_name}</b>, parameters {run.best['best_params'] or 'default'}."),
                  _table(_fmt_table(run)), P(f"Table — Model comparison for {label} (* = selected by CV RMSE).", "Cap")]
        b = run.best
        story += [KeepTogether([_fig(viz.actual_vs_predicted(b["y_test"], b["y_pred_test"], f"{kind}: actual vs predicted ({run.best_model_name})", unit, b["test_metrics"]), 10),
                                P(f"Figure {fig_no} — Actual vs predicted {kind} on the hold-out test set.", "Cap")])]
        fig_no += 1
        story += [KeepTogether([_fig(viz.residual_plots(b["y_test"], b["y_pred_test"], f"{kind} residuals", unit), 15),
                                P(f"Figure {fig_no} — Residual analysis for {kind}.", "Cap")])]
        fig_no += 1
        story += [KeepTogether([_fig(viz.model_comparison(metrics_table(run.results), f"{kind}: model comparison"), 15.5),
                                P(f"Figure {fig_no} — Comparison of algorithms for {kind}.", "Cap")])]
        fig_no += 1
        try:
            imp = feature_importance(run, run.best_model_name)
            story += [KeepTogether([_fig(viz.feature_importance_plot(imp, f"{kind}: permutation importance"), 11),
                                    P(f"Figure {fig_no} — Permutation feature importance ({run.best_model_name}).", "Cap")])]
            fig_no += 1
        except Exception:
            pass

    num_cols = [c for c in dataset_info.get("heatmap_cols", []) if c in df.columns]
    if num_cols:
        story += [KeepTogether([_fig(viz.correlation_heatmap(df[num_cols]), 12),
                                P(f"Figure {fig_no} — Correlation matrix of the dataset variables.", "Cap")])]

    # ---------- discussion ----------
    story += [P("8. Discussion", "H1x")]
    for t in dataset_info.get("discussion", []):
        story.append(P(t))
    if mrr_run:
        story.append(P("Because MRR is a deterministic product of the three inputs, a high score is not evidence of "
                       "predictive power on real measurements; it only shows how well each algorithm can represent a "
                       "multiplicative relationship. Linear regression cannot represent the product term exactly. Tree-based "
                       "models approximate it with piecewise-constant steps and cannot interpolate smoothly to parameter "
                       "settings absent from the training set, which matters with a grouped split, whereas the RBF-kernel "
                       "SVR produces a smooth response surface. In practice the analytical equation should be used for MRR; "
                       "the ML model is included as required by the assignment."))
    if ra_run:
        story.append(P("R<sub>a</sub> is a genuinely measured, noisy quantity influenced by effects not captured in the inputs "
                       "(likely tool wear progression, vibrations and measurement scatter), which limits the achievable accuracy. The "
                       "models are valid only inside the experimental parameter ranges and for the materials tested."))
    story += [P("9. Conclusion", "H1x"),
              P(dataset_info.get("conclusion", "Regression models were trained on genuine experimental data and evaluated on a hold-out set."))]

    story += [P("References", "H1x")] + [P(f"[{i}] {r}", "Ref") for i, r in enumerate(REFERENCES, 1)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="ML prediction of MRR and surface roughness", author=authors or "MEDX 55 project")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
