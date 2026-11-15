"""Shared Streamlit UI helpers and session state."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from .data_loader import load_default
from .prediction import load_bundle
from .preprocessing import prepare

ROOT = Path(__file__).resolve().parents[1]


def inject_css():
    css = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def init_state():
    ss = st.session_state
    if "df" not in ss:
        raw = load_default()
        if raw is not None:
            df, warns = prepare(raw)
            ss.df, ss.dataset_name, ss.dataset_warnings = df, "Hannover longitudinal turning dataset (DOI 10.25835/9xffqwnc)", warns
            ss.is_default = True
        else:
            ss.df, ss.dataset_name, ss.dataset_warnings, ss.is_default = None, None, [], False
    if "bundles" not in ss:
        ss.bundles = {}
        ss.runs = {}
        for kind in ("MRR", "Ra"):
            b = load_bundle(kind)
            # only reuse saved models that were trained on the dataset currently loaded
            if b is not None and (not ss.is_default or "Hannover" in str(b.get("dataset", ""))):
                ss.bundles[kind] = b
                ss.runs[kind] = b.get("run")
    ss.setdefault("runs", {})


def get_df():
    return st.session_state.get("df")


def get_bundle(kind: str):
    return st.session_state.get("bundles", {}).get(kind)


def get_run(kind: str):
    return st.session_state.get("runs", {}).get(kind)


def header(eyebrow: str, title: str, text: str = ""):
    st.markdown(f"""<div class="page-header"><div class="eyebrow">{eyebrow}</div>
<h1>{title}</h1><p>{text}</p></div>""", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = ""):
    st.markdown(f"""<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>
<div class="sub">{sub}</div></div>""", unsafe_allow_html=True)


def card(title: str, body_html: str, accent: bool = False):
    st.markdown(f"""<div class="card{' accent' if accent else ''}"><h4>{title}</h4>{body_html}</div>""",
                unsafe_allow_html=True)


def note(text: str, warn: bool = False):
    st.markdown(f"""<div class="note{' warn' if warn else ''}">{text}</div>""", unsafe_allow_html=True)


def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def need_data() -> bool:
    """Render an empty state and return True if no dataset is loaded."""
    if get_df() is None or get_df().empty:
        st.info("No dataset is loaded. Open **Dataset Explorer** to load the published dataset or upload a CSV file.")
        st.page_link("views/dataset_explorer.py", label="Go to Dataset Explorer", icon=":material/table_view:")
        return True
    return False


def need_model(kind: str) -> bool:
    if get_bundle(kind) is None:
        name = "MRR" if kind == "MRR" else "surface roughness (Ra)"
        st.warning(f"No trained {name} model is available. Train one on the **Model Training** page "
                   f"(select target “{kind}”), then return here.")
        from .prediction import LOAD_ERRORS
        if kind in LOAD_ERRORS:
            st.caption(f"Saved model could not be loaded — {LOAD_ERRORS[kind]}")
        st.page_link("views/model_training.py", label="Go to Model Training", icon=":material/model_training:")
        return True
    return False


def sidebar_status():
    ss = st.session_state
    df = ss.get("df")
    with st.sidebar:
        st.markdown("""<div class="sb-brand"><div class="t2">MEDX 55 · Project 08</div>
<div class="t1">Machining ML Lab</div></div>""", unsafe_allow_html=True)
        lines = []
        dot = lambda on: f'<span class="dot {"on" if on else "off"}"></span>'
        lines.append(f"{dot(df is not None)}Dataset: {'%d rows' % len(df) if df is not None else 'not loaded'}")
        for kind in ("MRR", "Ra"):
            b = ss.get("bundles", {}).get(kind)
            lines.append(f"{dot(b is not None)}{kind} model: {b['model_name'] if b else 'not trained'}")
        st.markdown(f'<div class="sb-status">{"<br>".join(lines)}</div>', unsafe_allow_html=True)
        st.divider()
