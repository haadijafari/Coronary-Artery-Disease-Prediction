"""Shared resources, styling and helpers for the Streamlit app."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from heart_disease.artifact import ModelArtifact
from heart_disease.config import METRICS_PATH, MODEL_PATH
from heart_disease.data import load_dataset
from heart_disease.plots import BLUE, ORANGE
from heart_disease.schema import CLASS_LABELS, FEATURES_BY_NAME, TARGET

INK = "#1d2733"
INK_SECONDARY = "#4f5b67"
GRID = "#e3e6ea"
DIAGNOSIS = "Diagnosis"
DIAGNOSIS_SCALE = alt.Scale(domain=[CLASS_LABELS[0], CLASS_LABELS[1]], range=[BLUE, ORANGE])

DISCLAIMER = (
    "Educational project, not medical advice. The model was trained on 302 patients referred "
    "for angiography in the 1980s and must not be used to make decisions about anyone's health."
)

REPOSITORY_URL = "https://github.com/haadijafari/Heart-Attack"


@st.cache_resource
def load_model() -> ModelArtifact:
    return ModelArtifact.load(MODEL_PATH)


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def for_charts(data: pd.DataFrame) -> pd.DataFrame:
    """A copy with readable diagnosis labels and plain string categories for Altair."""
    frame = data.copy()
    frame[DIAGNOSIS] = frame[TARGET].map(CLASS_LABELS)
    for column in frame.select_dtypes("category"):
        frame[column] = frame[column].astype("string")
    return frame


def label(name: str) -> str:
    """Readable label for a feature, with its unit if it has one."""
    feature = FEATURES_BY_NAME[name]
    return f"{feature.label} ({feature.unit})" if feature.unit else feature.label


def style_chart(chart: alt.TopLevelMixin, height: int | alt.Step = 320) -> alt.TopLevelMixin:
    """Apply the project chart style: quiet axes, hairline grid, Lato text."""
    return (
        chart.properties(height=height, background="transparent")
        .configure(font="Lato")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=INK_SECONDARY,
            titleColor=INK_SECONDARY,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            gridColor=GRID,
            domainColor="#c3c2b7",
            tickSize=0,
            labelPadding=6,
        )
        .configure_legend(
            labelColor=INK, labelFontSize=12, titleFontSize=12, orient="top", title=None
        )
    )


def apply_style() -> None:
    """Global CSS for the pieces Streamlit's theme cannot express."""
    st.html(
        f"""
        <style>
          .block-container {{ max-width: 1180px; padding-top: 3rem; }}
          .stMarkdown p, .stMarkdown li {{ max-width: 72ch; line-height: 1.6; }}

          .readout {{ margin: 0.25rem 0 1.5rem; }}
          .readout-value {{
            font-family: 'Source Serif 4', serif; font-weight: 600;
            font-size: 4rem; line-height: 1; letter-spacing: -0.02em;
          }}
          .readout-caption {{ color: {INK_SECONDARY}; margin-top: 0.4rem; }}

          .scale {{ margin-top: 1.4rem; }}
          .scale-track {{
            position: relative; height: 12px; border-radius: 6px; background: #e8f0fb;
          }}
          .scale-flagged {{
            position: absolute; top: 0; bottom: 0; right: 0; border-radius: 0 6px 6px 0;
            background: #fbe1d6;
          }}
          .scale-threshold {{
            position: absolute; top: -6px; bottom: -6px; width: 2px; background: {INK};
          }}
          .scale-marker {{
            position: absolute; top: 50%; width: 22px; height: 22px; border-radius: 50%;
            transform: translate(-50%, -50%); border: 3px solid #fcfcfb;
            box-shadow: 0 0 0 1px rgba(29, 39, 51, 0.25);
          }}
          .scale-labels {{
            position: relative; height: 2.6rem; margin-top: 0.5rem; font-size: 0.85rem;
            color: {INK_SECONDARY};
          }}
          .scale-labels span {{ position: absolute; white-space: nowrap; }}
        </style>
        """
    )
