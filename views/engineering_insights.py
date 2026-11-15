import numpy as np
import streamlit as st

from src import insights as ins
from src import ui
from src import visualization as viz

ui.header("Page 7 · Interpretation", "Engineering Insights",
          "How cutting speed, feed and depth of cut influence MRR and surface roughness. Statements marked "
          "<b>Observed</b> are computed live from the loaded dataset; statements marked <b>Principle</b> are general "
          "machining theory from textbooks.")

if ui.need_data():
    st.stop()
df = ui.get_df()
hue = "material" if "material" in df.columns else None


def observed(lines):
    for t in lines:
        st.markdown(f"<span class='tag meas'>Observed</span> {t}", unsafe_allow_html=True)


def principle(text):
    st.markdown(f"<span class='tag grey'>Principle</span> {text}", unsafe_allow_html=True)


# ---------------------------------------------------------------- MRR
st.subheader("1 · Effect of cutting parameters on MRR")
if "MRR_mm3_min" in df.columns:
    principle("MRR = 1000·Vc·f·ap is linear in each parameter: doubling any one of Vc, f or ap doubles MRR when the "
              "others are fixed. Feed and depth of cut also raise the cutting force (≈ kc·ap·f), whereas Vc mainly raises "
              "temperature and tool wear, so tool life — not MRR — usually limits Vc.")
    st.markdown("<span class='tag calc'>Note</span> MRR here is <i>calculated</i> from the recorded parameters, so the "
                "plots show how MRR varied over the experimental design, not an independent measurement.",
                unsafe_allow_html=True)
    c = st.columns(3)
    for col, (p, unit) in zip(c, (("vc", "m/min"), ("f", "mm/rev"), ("ap", "mm"))):
        with col:
            if p in df.columns:
                ui.show_fig(viz.effect_plot(df, p, "MRR_mm3_min", None, f"MRR vs {p}", f"{p} [{unit}]", "MRR [mm³/min]"))
    observed(ins.mrr_observations(df))
    if {"vc", "f", "ap"}.issubset(df.columns):
        med = {p: float(df[p].median()) for p in ("vc", "f", "ap")}
        base = 1000 * med["vc"] * med["f"] * med["ap"]
        st.markdown(f"<span class='tag grey'>Principle</span> At the median setting (Vc {med['vc']:g} m/min, f {med['f']:g} mm/rev, "
                    f"ap {med['ap']:g} mm) MRR = {base:,.0f} mm³/min; +10 % in any single parameter gives "
                    f"{base*1.1:,.0f} mm³/min (+10 %).", unsafe_allow_html=True)
else:
    st.info("MRR is unavailable (Vc, f and ap are required).")

# ---------------------------------------------------------------- Ra
st.subheader("2 · Machining parameters and surface roughness")
if "Ra" in df.columns:
    principle("For a round-nose tool the ideal feed-mark profile gives Ra ≈ f²/(32·rε): roughness grows with the square "
              "of feed and falls with a larger corner radius. Higher cutting speed generally reduces built-up edge and "
              "improves the surface in steels; depth of cut has only a secondary, indirect effect via forces and vibration.")
    c = st.columns(3)
    for col, (p, unit) in zip(c, (("f", "mm/rev"), ("vc", "m/min"), ("ap", "mm"))):
        with col:
            if p in df.columns:
                fig = viz.effect_plot(df, p, "Ra", hue, f"Ra vs {p}", f"{p} [{unit}]", "Ra [µm]")
                if p == "f" and "re" in df.columns and df["re"].nunique() == 1:
                    xs = np.linspace(df["f"].min(), df["f"].max(), 50)
                    fig.axes[0].plot(xs, ins.kinematic_ra(xs, float(df["re"].iloc[0])), "k--", lw=1.1,
                                     label="kinematic f²/(32rε)")
                    fig.axes[0].legend(fontsize=7)
                ui.show_fig(fig)
    observed(ins.ra_observations(df))
    with st.expander("Correlation table by material (Pearson and Spearman)"):
        st.dataframe(ins.correlation_table(df, "Ra").round(3), hide_index=True, width="stretch")
    eff = ins.ols_effects(df, "Ra")
    if not eff.empty:
        with st.expander("Multiple linear regression on standardised inputs"):
            st.dataframe(eff.round(4), hide_index=True, width="stretch")
            st.caption(f"R² = {eff.attrs['R2']:.3f}. Coefficients are µm of Ra per one standard deviation of each parameter; "
                       "material terms are offsets relative to the first material alphabetically.")
    if "cut_index" in df.columns:
        principle("Progressive flank wear during a series of cuts can change roughness over time. The cut sequence index "
                  "is available as an optional feature to examine this.")
else:
    st.info("No Ra column in the loaded dataset.")

# ---------------------------------------------------------------- trade-off
if {"Ra", "MRR_mm3_min"}.issubset(df.columns):
    st.subheader("3 · Productivity vs surface quality")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    groups = df.groupby(hue) if hue else [("all", df)]
    for i, (k, g) in enumerate(groups):
        ax.scatter(g["MRR_mm3_min"], g["Ra"], s=18, alpha=.7, color=viz.PALETTE[i % 6], label=str(k))
    ax.set_xlabel("MRR [mm³/min] (calculated)"); ax.set_ylabel("Ra [µm] (measured)")
    ax.set_title("Each point is one real cut"); ax.legend(fontsize=8)
    fig.tight_layout()
    ui.show_fig(fig)
    r = df["MRR_mm3_min"].corr(df["Ra"])
    observed([f"Pearson correlation between calculated MRR and measured Ra across all cuts is r = {r:.2f}."])
    principle("Because Ra depends mainly on feed while MRR is proportional to Vc, f and ap alike, productivity can be "
              "increased with less penalty on surface quality by raising Vc (within tool-life limits) and ap rather than f.")

# ---------------------------------------------------------------- algorithms
st.subheader("4 · Comparison of prediction algorithms")
runs = {k: ui.get_run(k) for k in ("MRR", "Ra")}
obs = ins.algorithm_observations(runs)
if obs:
    observed(obs)
else:
    st.info("Train models to populate this comparison.")
principle("Linear regression assumes additive linear effects and cannot represent the product Vc·f·ap or the f² "
          "dependence of Ra. Decision trees approximate any shape with steps but are high-variance. Random forests "
          "average many trees (lower variance); gradient boosting fits trees sequentially to residuals. SVR with an RBF "
          "kernel produces smooth non-linear surfaces, which suits smooth physical relationships on small datasets. "
          "Tree models cannot extrapolate beyond the training range (predictions become constant).")
