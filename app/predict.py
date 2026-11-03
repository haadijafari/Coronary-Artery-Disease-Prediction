"""Risk estimate: enter a patient's measurements and see the model's estimate, explained."""

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from common import DISCLAIMER, load_data, load_metrics, load_model, style_chart

from heart_disease.plots import BLUE, ORANGE
from heart_disease.schema import (
    CATEGORICAL_FEATURES,
    CLASS_LABELS,
    FEATURE_NAMES,
    FEATURES_BY_NAME,
    NUMERIC_FEATURES,
    TARGET,
)

NOT_MEASURED = "Not measured"
OPTIONAL = {"major_vessels", "thal"}  # missing in a few training rows; the pipeline imputes them

model = load_model()
data = load_data()
metrics = load_metrics()


def typical_values() -> dict:
    """Median of each numeric feature and most common category, as widget values."""
    values = {}
    for name in NUMERIC_FEATURES:
        median = data[name].median()
        values[name] = round(float(median), 1) if name == "st_depression" else round(median)
    for name in CATEGORICAL_FEATURES:
        values[name] = str(data[name].mode().iloc[0])
    values["major_vessels"] = str(values["major_vessels"])
    return values


def set_inputs(values: dict) -> None:
    for name, value in values.items():
        st.session_state[name] = value


def load_random_patient() -> None:
    row = data.sample(1).iloc[0]
    values = {}
    for name in FEATURE_NAMES:
        value = row[name]
        if pd.isna(value):
            values[name] = NOT_MEASURED
        elif name == "major_vessels":
            values[name] = str(int(value))
        elif name == "st_depression":
            values[name] = float(value)
        elif name in NUMERIC_FEATURES:
            values[name] = int(value)
        else:
            values[name] = str(value)
    set_inputs(values)
    st.session_state["actual_diagnosis"] = int(row[TARGET])


def reset_inputs() -> None:
    set_inputs(typical_values())
    st.session_state.pop("actual_diagnosis", None)


def forget_actual_diagnosis() -> None:
    st.session_state.pop("actual_diagnosis", None)


def patient_frame() -> pd.DataFrame:
    """The current inputs as a one-row frame with the same dtypes as the training data."""
    row = {}
    for name in FEATURE_NAMES:
        value = st.session_state[name]
        feature = FEATURES_BY_NAME[name]
        if value == NOT_MEASURED:
            value = None
        if feature.kind == "categorical":
            row[name] = pd.Categorical([value], categories=feature.categories)
        else:
            row[name] = [np.nan if value is None else float(value)]
    return pd.DataFrame(row)


def choice(name: str, options: list[str], horizontal: bool = True, **kwargs) -> None:
    feature = FEATURES_BY_NAME[name]
    common = {"key": name, "help": feature.description, "on_change": forget_actual_diagnosis}
    if horizontal:
        st.radio(feature.label, options, horizontal=True, **common, **kwargs)
    else:
        st.selectbox(feature.label, options, **common, **kwargs)


def number(name: str, step: float = 1, fmt: str = "%d") -> None:
    feature = FEATURES_BY_NAME[name]
    low, high = feature.valid_range
    cast = float if isinstance(step, float) else int
    st.number_input(
        f"{feature.label} ({feature.unit})" if feature.unit else feature.label,
        min_value=cast(low),
        max_value=cast(high),
        step=step,
        format=fmt,
        key=name,
        help=feature.description,
        on_change=forget_actual_diagnosis,
    )


if "age" not in st.session_state:
    reset_inputs()

st.title("Coronary artery disease risk")
st.caption(DISCLAIMER)

inputs, result = st.columns([5, 7], gap="large")

with inputs:
    sample, reset = st.columns(2)
    sample.button(
        "Load a random patient", on_click=load_random_patient, icon=":material/shuffle:",
        width="stretch",
    )  # fmt: skip
    reset.button(
        "Reset to typical values", on_click=reset_inputs, icon=":material/restart_alt:",
        width="stretch",
    )  # fmt: skip

    st.subheader("Patient")
    st.slider("Age (years)", 18, 100, key="age", on_change=forget_actual_diagnosis)
    choice("sex", FEATURES_BY_NAME["sex"].categories)
    choice("chest_pain", FEATURES_BY_NAME["chest_pain"].categories, horizontal=False)

    st.subheader("At rest")
    left, right = st.columns(2)
    with left:
        number("resting_bp")
    with right:
        number("cholesterol")
    choice("high_fasting_sugar", FEATURES_BY_NAME["high_fasting_sugar"].categories)
    choice("rest_ecg", FEATURES_BY_NAME["rest_ecg"].categories, horizontal=False)

    st.subheader("Exercise stress test")
    left, right = st.columns(2)
    with left:
        number("max_heart_rate")
    with right:
        number("st_depression", step=0.1, fmt="%.1f")
    choice("exercise_angina", FEATURES_BY_NAME["exercise_angina"].categories)
    choice("st_slope", FEATURES_BY_NAME["st_slope"].categories)

    st.subheader("Imaging")
    choice("major_vessels", ["0", "1", "2", "3", NOT_MEASURED])
    choice("thal", [*FEATURES_BY_NAME["thal"].categories, NOT_MEASURED], horizontal=False)

patient = patient_frame()
probability = float(model.predict_proba(patient)[0])
flagged = probability >= model.threshold
colour = ORANGE if flagged else BLUE
test = {row["metric"]: row for row in metrics["test"]["metrics"]}

with result:
    threshold_pct = model.threshold * 100
    marker_pct = min(max(probability * 100, 1.5), 98.5)
    st.html(
        f"""
        <div class="readout">
          <div class="readout-value" style="color: {colour}">{probability:.0%}</div>
          <div class="readout-caption">estimated probability of coronary artery disease</div>
          <div class="scale" role="img"
               aria-label="Probability {probability:.0%} against a decision threshold of
               {model.threshold:.0%}">
            <div class="scale-track">
              <div class="scale-flagged" style="left: {threshold_pct:.1f}%"></div>
              <div class="scale-threshold" style="left: {threshold_pct:.1f}%"></div>
              <div class="scale-marker" style="left: {marker_pct:.1f}%; background: {colour}">
              </div>
            </div>
            <div class="scale-labels">
              <span style="left: 0">0%</span>
              <span style="left: {threshold_pct:.1f}%; transform: translateX(-50%)">
                threshold {model.threshold:.0%}</span>
              <span style="right: 0">100%</span>
            </div>
          </div>
        </div>
        """
    )

    sensitivity = test["sensitivity"]["estimate"]
    false_alarms = 1 - test["specificity"]["estimate"]
    if flagged:
        st.markdown(
            f"**Flagged for further assessment.** The estimate is at or above the "
            f"{model.threshold:.0%} threshold. On unseen patients that threshold caught "
            f"{sensitivity:.0%} of disease cases, at the cost of flagging {false_alarms:.0%} of "
            "healthy patients too."
        )
    else:
        st.markdown(
            f"**Below the decision threshold.** The estimate is under {model.threshold:.0%}. "
            f"On unseen patients, {test['npv']['estimate']:.0%} of the people below the threshold "
            "were truly free of disease."
        )

    if "actual_diagnosis" in st.session_state:
        truth = CLASS_LABELS[st.session_state["actual_diagnosis"]].lower()
        st.info(
            f"This is a real patient from the dataset. Their angiogram showed **{truth}**. "
            "The model saw this patient during training, so this is an illustration, not a test.",
            icon=":material/clinical_notes:",
        )

    st.subheader("What moves this estimate")
    contributions = model.explain(patient).iloc[0]
    explanation = pd.DataFrame(
        {
            "feature": [FEATURES_BY_NAME[name].label for name in contributions.index],
            "value": [
                NOT_MEASURED if pd.isna(patient[name].iloc[0]) else f"{patient[name].iloc[0]:g}"
                if name in NUMERIC_FEATURES else str(patient[name].iloc[0])
                for name in contributions.index
            ],
            "contribution": contributions.to_numpy(),
        }
    )  # fmt: skip
    explanation["direction"] = np.where(
        explanation["contribution"] > 0, "Towards disease", "Towards no disease"
    )
    explanation["odds"] = [f"× {math.exp(c):.2f}" for c in explanation["contribution"]]
    explanation["row"] = explanation["feature"] + " = " + explanation["value"]
    explanation = explanation.reindex(
        explanation["contribution"].abs().sort_values(ascending=False).index
    )

    bars = (
        alt.Chart(explanation)
        .mark_bar(cornerRadiusEnd=3, height=16)
        .encode(
            x=alt.X("contribution:Q", title="Push on the log-odds of disease"),
            y=alt.Y(
                "row:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Towards no disease", "Towards disease"], range=[BLUE, ORANGE]
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip("contribution:Q", title="Log-odds", format="+.2f"),
                alt.Tooltip("odds:N", title="Odds multiplier"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3c2b7").encode(x="x:Q")
    st.altair_chart(style_chart(zero + bars, height=alt.Step(26)), width="stretch", theme=None)
    st.caption(
        "Each bar is how much one measurement moves this patient's log-odds compared with an "
        "average patient in the training data. Because the model is a logistic regression, the "
        "bars add up exactly to the estimate above. Hover a bar for its odds multiplier."
    )
