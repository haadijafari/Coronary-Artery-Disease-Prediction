"""Explore the data: distributions, disease rates and relationships, interactively."""

import altair as alt
import pandas as pd
import streamlit as st
from common import DIAGNOSIS, DIAGNOSIS_SCALE, for_charts, label, load_data, style_chart

from heart_disease.interpretation import disease_rate_by_category
from heart_disease.plots import ORANGE
from heart_disease.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, FEATURES_BY_NAME, TARGET

CONTINUOUS = ["age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression"]
DISCRETE = [*CATEGORICAL_FEATURES, "major_vessels"]

data = load_data()
chart_data = for_charts(data)
prevalence = data[TARGET].mean()

st.title("Explore the data")
st.markdown(
    f"302 patients referred for coronary angiography at the Cleveland Clinic. "
    f"**{data[TARGET].sum()} ({prevalence:.0%})** had coronary artery disease. Pick a "
    "measurement to see how it differs between patients with and without disease."
)

feature = st.selectbox(
    "Measurement", [n for n in FEATURE_NAMES if n in CONTINUOUS or n in DISCRETE],
    format_func=lambda n: FEATURES_BY_NAME[n].label,
)  # fmt: skip
st.caption(FEATURES_BY_NAME[feature].description)

if feature in CONTINUOUS:
    density = (
        alt.Chart(chart_data)
        .transform_density(
            feature, groupby=[DIAGNOSIS], as_=[feature, "density"], extent=[
                float(data[feature].min()), float(data[feature].max())
            ],
        )
        .mark_area(opacity=0.18, line={"strokeWidth": 2}, interpolate="monotone")
        .encode(
            x=alt.X(f"{feature}:Q", title=label(feature), axis=alt.Axis(grid=False, tickCount=8)),
            # Overlay the two densities; Vega-Lite would otherwise stack them.
            y=alt.Y("density:Q", title=None, stack=None, axis=alt.Axis(labels=False, grid=False)),
            color=alt.Color(f"{DIAGNOSIS}:N", scale=DIAGNOSIS_SCALE),
            tooltip=[
                alt.Tooltip(f"{DIAGNOSIS}:N"),
                alt.Tooltip(f"{feature}:Q", title=FEATURES_BY_NAME[feature].label, format=".1f"),
            ],
        )
    )  # fmt: skip
    st.altair_chart(style_chart(density), width="stretch", theme=None)
    summary = (
        data.groupby(TARGET)[feature]
        .describe()[["count", "mean", "25%", "50%", "75%", "min", "max"]]
        .rename(index={0: "No disease", 1: "Disease"})
    )
    with st.expander("Table view"):
        st.dataframe(summary.round(1), width="stretch")
else:
    rates = disease_rate_by_category(data, feature)
    rates["category"] = rates["category"].map(
        lambda c: str(int(c)) if isinstance(c, float) else str(c)
    )
    order = rates["category"].tolist()
    base = alt.Chart(rates).encode(
        y=alt.Y("category:N", sort=order, title=None, axis=alt.Axis(labelLimit=240))
    )
    interval = base.mark_rule(strokeWidth=5, color=ORANGE, opacity=0.35).encode(
        x=alt.X(
            "lower:Q", title="Share of patients with disease", scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        x2="upper:Q",
    )  # fmt: skip
    point = base.mark_circle(size=140, color=ORANGE, opacity=1, stroke="#fcfcfb").encode(
        x="rate:Q",
        tooltip=[
            alt.Tooltip("category:N", title=FEATURES_BY_NAME[feature].label),
            alt.Tooltip("n:Q", title="Patients"),
            alt.Tooltip("cases:Q", title="With disease"),
            alt.Tooltip("rate:Q", title="Disease rate", format=".0%"),
            alt.Tooltip("lower:Q", title="95% CI low", format=".0%"),
            alt.Tooltip("upper:Q", title="95% CI high", format=".0%"),
        ],
    )
    overall = (
        alt.Chart(pd.DataFrame({"x": [prevalence]}))
        .mark_rule(color="#9aa3ad", strokeWidth=1)
        .encode(x="x:Q")
    )
    st.altair_chart(
        style_chart(overall + interval + point, height=max(140, 56 * len(rates))),
        width="stretch",
        theme=None,
    )
    st.caption(
        f"Dot = share with disease, bar = 95% Wilson interval, grey line = overall rate "
        f"({prevalence:.0%}). Wide bars mean few patients and an uncertain rate."
    )
    with st.expander("Table view"):
        table = rates.set_index("category")[["n", "cases", "rate", "lower", "upper"]]
        st.dataframe(table.style.format(dict.fromkeys(("rate", "lower", "upper"), "{:.0%}")))

st.subheader("Two measurements together")
left, right = st.columns(2)
x_feature = left.selectbox(
    "Horizontal axis", CONTINUOUS, index=0, format_func=lambda n: FEATURES_BY_NAME[n].label
)
y_feature = right.selectbox(
    "Vertical axis", CONTINUOUS, index=3, format_func=lambda n: FEATURES_BY_NAME[n].label
)
scatter = (
    alt.Chart(chart_data)
    .mark_circle(size=70, opacity=0.75, stroke="#fcfcfb", strokeWidth=1)
    .encode(
        x=alt.X(
            f"{x_feature}:Q",
            title=label(x_feature),
            scale=alt.Scale(zero=False),
            axis=alt.Axis(tickCount=8),
        ),
        y=alt.Y(f"{y_feature}:Q", title=label(y_feature), scale=alt.Scale(zero=False)),
        color=alt.Color(f"{DIAGNOSIS}:N", scale=DIAGNOSIS_SCALE),
        tooltip=[
            alt.Tooltip(f"{DIAGNOSIS}:N"),
            alt.Tooltip("age:Q", title="Age"),
            alt.Tooltip("sex:N", title="Sex"),
            alt.Tooltip("chest_pain:N", title="Chest pain"),
            alt.Tooltip(f"{x_feature}:Q", title=FEATURES_BY_NAME[x_feature].label),
            alt.Tooltip(f"{y_feature}:Q", title=FEATURES_BY_NAME[y_feature].label),
        ],
    )
)
st.altair_chart(style_chart(scatter, height=380), width="stretch", theme=None)
