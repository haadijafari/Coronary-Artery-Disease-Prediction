"""Model performance: how the model was chosen and how it did on unseen patients."""

import pandas as pd
import streamlit as st
from common import load_metrics, load_model

from heart_disease.config import FIGURES_DIR
from heart_disease.evaluation import METRIC_LABELS

metrics = load_metrics()
model = load_model()
test = metrics["test"]

st.title("Model performance")
selected = metrics["selected_model"].lower()
st.markdown(
    f"Five models were compared with nested cross-validation on 241 training patients. The "
    f"simplest model within one standard error of the best, **{selected}**, "
    f"was kept. Its decision threshold was then set to catch at least "
    f"**{metrics['target_recall']:.0%} of disease cases**. The {test['n_patients']} patients below "
    "were held out from every step and used once, at the end."
)


def show_figure(name: str, caption: str | None = None) -> None:
    path = FIGURES_DIR / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Figure `{name}.png` is missing. Run the analysis notebook to create it.")


st.subheader(f"Held-out test set: {test['n_patients']} patients, {test['n_disease']} with disease")
rows = [
    {
        "Metric": METRIC_LABELS[row["metric"]],
        "Estimate": f"{row['estimate']:.3f}",
        "95% bootstrap interval": f"{row['lower']:.3f} to {row['upper']:.3f}",
    }
    for row in test["metrics"]
]
st.dataframe(pd.DataFrame(rows).set_index("Metric"), width="stretch")
st.caption(
    f"Threshold-dependent metrics use the threshold of {test['threshold']:.2f} chosen on the "
    "training set. With 61 patients the intervals are wide; the cross-validated ROC AUC below is "
    "the more stable estimate."
)
show_figure("test_performance")

st.subheader("Choosing the model")
summary = pd.DataFrame(metrics["cross_validation"]["summary"]).set_index("model")
summary.columns = ["Mean ROC AUC", "Standard deviation", "Standard error"]
st.dataframe(summary.style.format("{:.3f}"), width="stretch")
show_figure("model_comparison")

st.subheader("Choosing the threshold")
show_figure("threshold_tradeoff")

st.subheader("The model in use")
st.markdown(
    f"After evaluation, the same tuning and threshold procedure was repeated on all "
    f"{model.n_patients} patients to produce the model behind the risk estimate page "
    f"(regularisation strength C = {model.params.get('C', 'n/a')}, threshold "
    f"{model.threshold:.2f})."
)
