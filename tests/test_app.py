"""Headless UI tests: every page renders, buttons work (Streamlit AppTest)."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
PAGES = ["views/overview.py", "views/dataset_explorer.py", "views/model_training.py", "views/performance_analysis.py",
         "views/mrr_simulator.py", "views/roughness_simulator.py", "views/engineering_insights.py",
         "views/report_generator.py"]


def _app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _app()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.markdown) > 0


def _button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"button {label!r} not found: {[b.label for b in at.button]}")


def test_train_ra_and_save_button():
    at = _app()
    at.switch_page("views/model_training.py").run()
    at.radio[0].set_value("Ra").run()
    at.multiselect(key="algos").set_value(["Linear Regression", "Decision Tree"]).run()
    _button(at, "Train models").click().run()
    assert not at.exception, [e.value for e in at.exception]
    run = at.session_state["runs"]["Ra"]
    assert set(run.results) == {"Linear Regression", "Decision Tree"}


def test_mrr_prediction_form():
    at = _app()
    at.switch_page("views/mrr_simulator.py").run()
    at.number_input(key="mrr_vc").set_value(200.0)
    at.number_input(key="mrr_f").set_value(0.2)
    at.number_input(key="mrr_ap").set_value(1.0)
    _button(at, "Predict MRR").click().run()
    assert not at.exception, [e.value for e in at.exception]
    vals, y = at.session_state["mrr_pred"]
    assert abs(y - 40000) / 40000 < 0.15
    assert any("40,000.0 mm³/min" in m.value for m in at.markdown)


def test_mrr_invalid_input():
    at = _app()
    at.switch_page("views/mrr_simulator.py").run()
    at.number_input(key="mrr_f").set_value(0.0)
    _button(at, "Predict MRR").click().run()
    assert not at.exception
    assert len(at.error) >= 1


def test_ra_prediction_form():
    at = _app()
    at.switch_page("views/roughness_simulator.py").run()
    at.selectbox(key="ra_material").set_value("41Cr4")
    at.number_input(key="ra_f").set_value(0.25)
    _button(at, "Predict surface roughness").click().run()
    assert not at.exception, [e.value for e in at.exception]
    _, y = at.session_state["ra_pred"]
    assert 1.0 < y < 5.0


def test_report_generation():
    at = _app()
    at.switch_page("views/report_generator.py").run()
    _button(at, "Generate PDF report").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["report_pdf"][:4] == b"%PDF"


def test_no_model_empty_state():
    at = _app()
    at.session_state["bundles"] = {}
    at.session_state["runs"] = {}
    for p in ("views/mrr_simulator.py", "views/roughness_simulator.py", "views/performance_analysis.py"):
        at.switch_page(p).run()
        assert not at.exception
        assert len(at.warning) >= 1
