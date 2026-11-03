"""About the project: the data, what the model predicts, and its limits."""

import streamlit as st
from common import DISCLAIMER, REPOSITORY_URL

st.title("About the project")

st.markdown(
    f"""
This app is the interactive end of a data science project on the
[UCI Cleveland heart disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).
The full analysis, code and tests are on [GitHub]({REPOSITORY_URL}).

### What the model predicts

Every patient in the data was referred to the Cleveland Clinic for coronary angiography between
1981 and 1984. The model estimates the probability that the angiogram shows **coronary artery
disease**: more than 50% narrowing of at least one major coronary artery. That is a diagnosis made
at the visit. It is not the risk of a future heart attack.

### What was wrong with the data, and how it was fixed

The widely shared Kaggle copy of this dataset looks clean but is not:

- **The categories were re-coded.** Chest pain, resting ECG, ST slope and the thallium test use
  different codes from the UCI documentation. The correct mapping was recovered by matching every
  row against the original UCI file.
- **The target is inverted.** In the Kaggle file, `output = 1` means *no* disease.
- **Missing values hide behind codes.** Five patients have a vessel count of "4" and two have a
  thallium result of "0". Both mean "not recorded".
- **One patient appears twice**, which can leak into evaluation, so the duplicate is removed.

No patient was deleted for having unusual measurements: high blood pressure or cholesterol is a
risk factor, not an error.

### Limits

- 302 patients from one hospital in the early 1980s.
- Everyone was already suspected of heart disease, so 46% had it, far more than in the general
  population. The probabilities only apply to similar patients.
- Several inputs (the thallium scan, fluoroscopy and the exercise test) come from specialist tests,
  so this is a decision aid within a cardiac work-up, not a screening tool.

{DISCLAIMER}
"""
)
