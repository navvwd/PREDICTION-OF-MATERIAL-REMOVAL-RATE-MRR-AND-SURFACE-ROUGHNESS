"""ML prediction of MRR and surface roughness in longitudinal turning — Streamlit entry point.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Machining ML Lab · MEDX 55 Project 08", page_icon=":material/precision_manufacturing:",
                   layout="wide", initial_sidebar_state="expanded")

from src import ui  # noqa: E402

ui.inject_css()
ui.init_state()

nav = st.navigation({
    "Project": [
        st.Page("views/overview.py", title="Project Overview", icon=":material/dashboard:", default=True),
        st.Page("views/dataset_explorer.py", title="Dataset Explorer", icon=":material/table_view:"),
    ],
    "Machine Learning": [
        st.Page("views/model_training.py", title="Model Training", icon=":material/model_training:"),
        st.Page("views/performance_analysis.py", title="Performance Analysis", icon=":material/query_stats:"),
    ],
    "Simulators": [
        st.Page("views/mrr_simulator.py", title="MRR Simulator", icon=":material/precision_manufacturing:"),
        st.Page("views/roughness_simulator.py", title="Roughness Simulator", icon=":material/texture:"),
    ],
    "Analysis & Output": [
        st.Page("views/engineering_insights.py", title="Engineering Insights", icon=":material/engineering:"),
        st.Page("views/report_generator.py", title="Report Generator", icon=":material/picture_as_pdf:"),
    ],
})
ui.sidebar_status()
nav.run()
