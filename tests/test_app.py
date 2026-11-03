"""Behavioural tests for the Streamlit app, run headlessly with AppTest."""

import pytest
from streamlit.testing.v1 import AppTest

from heart_disease.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")

HIGH_RISK = {
    "sex": "Male",
    "chest_pain": "Asymptomatic",
    "exercise_angina": "Yes",
    "st_slope": "Flat",
    "major_vessels": "3",
    "thal": "Reversible defect",
    "max_heart_rate": 105,
    "st_depression": 3.0,
}
LOW_RISK = {
    "sex": "Female",
    "chest_pain": "Non-anginal pain",
    "exercise_angina": "No",
    "st_slope": "Upsloping",
    "major_vessels": "0",
    "thal": "Normal",
    "max_heart_rate": 175,
    "st_depression": 0.0,
}


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    return app


def set_inputs(app: AppTest, values: dict) -> None:
    for key, value in values.items():
        widget = next(w for w in [*app.radio, *app.selectbox, *app.number_input] if w.key == key)
        widget.set_value(value)
    app.run()


def verdict(app: AppTest) -> str:
    return next(m.value for m in app.markdown if m.value.startswith("**"))


@pytest.mark.parametrize("page", ["predict.py", "explore.py", "performance.py", "about.py"])
def test_page_renders_without_errors(page):
    app = run_app()
    app.switch_page(page)
    app.run()
    assert not app.exception


def test_high_risk_profile_is_flagged():
    app = run_app()
    set_inputs(app, HIGH_RISK)
    assert verdict(app).startswith("**Flagged for further assessment.**")


def test_low_risk_profile_is_below_the_threshold():
    app = run_app()
    set_inputs(app, LOW_RISK)
    assert verdict(app).startswith("**Below the decision threshold.**")


def test_unmeasured_imaging_results_are_accepted():
    app = run_app()
    set_inputs(app, {"major_vessels": "Not measured", "thal": "Not measured"})
    assert not app.exception
