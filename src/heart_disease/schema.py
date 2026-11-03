"""Single source of truth for the dataset's columns.

The Kaggle file ``heart.csv`` is the UCI Cleveland heart disease dataset with
every categorical variable re-coded, the target inverted and missing values
replaced by out-of-range codes. The mappings below were verified row by row
against the original UCI ``processed.cleveland.data`` file: after decoding,
all 302 unique Kaggle rows match a UCI patient exactly.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Feature:
    """One predictor column: its raw Kaggle name, readable name and meaning."""

    name: str
    raw_name: str
    label: str
    description: str
    kind: Literal["numeric", "categorical"]
    unit: str | None = None
    codes: dict[int, str] = field(default_factory=dict)
    """Raw integer code -> readable category, listed in clinical order."""
    missing_codes: tuple[int, ...] = ()
    """Raw codes that stand for a missing measurement."""
    valid_range: tuple[float, float] | None = None
    """Physiologically plausible bounds; values outside are data errors."""

    @property
    def categories(self) -> list[str]:
        """Category labels in clinical order (categorical features only)."""
        return list(self.codes.values())


FEATURES: tuple[Feature, ...] = (
    Feature(
        name="age",
        raw_name="age",
        label="Age",
        description="Age of the patient.",
        kind="numeric",
        unit="years",
        valid_range=(18, 100),
    ),
    Feature(
        name="sex",
        raw_name="sex",
        label="Sex",
        description="Sex of the patient.",
        kind="categorical",
        codes={0: "Female", 1: "Male"},
    ),
    Feature(
        name="chest_pain",
        raw_name="cp",
        label="Chest pain type",
        description=(
            "Typical angina is chest pain brought on by exertion and relieved by rest; "
            "atypical angina meets only some of those criteria; non-anginal pain is "
            "unlikely to be cardiac; asymptomatic patients report no chest pain."
        ),
        kind="categorical",
        codes={3: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 0: "Asymptomatic"},
    ),
    Feature(
        name="resting_bp",
        raw_name="trtbps",
        label="Resting blood pressure",
        description="Systolic blood pressure at rest, measured on admission.",
        kind="numeric",
        unit="mm Hg",
        valid_range=(70, 250),
    ),
    Feature(
        name="cholesterol",
        raw_name="chol",
        label="Serum cholesterol",
        description="Total serum cholesterol.",
        kind="numeric",
        unit="mg/dl",
        valid_range=(100, 700),
    ),
    Feature(
        name="high_fasting_sugar",
        raw_name="fbs",
        label="Fasting blood sugar > 120 mg/dl",
        description="Whether fasting blood sugar exceeds 120 mg/dl.",
        kind="categorical",
        codes={0: "No", 1: "Yes"},
    ),
    Feature(
        name="rest_ecg",
        raw_name="restecg",
        label="Resting ECG",
        description=(
            "Resting electrocardiogram: normal, ST-T wave abnormality (T wave inversion "
            "and/or ST elevation or depression > 0.05 mV), or probable/definite left "
            "ventricular hypertrophy by Estes' criteria."
        ),
        kind="categorical",
        codes={1: "Normal", 2: "ST-T abnormality", 0: "LV hypertrophy"},
    ),
    Feature(
        name="max_heart_rate",
        raw_name="thalachh",
        label="Maximum heart rate",
        description="Highest heart rate reached during the exercise stress test.",
        kind="numeric",
        unit="bpm",
        valid_range=(60, 220),
    ),
    Feature(
        name="exercise_angina",
        raw_name="exng",
        label="Exercise-induced angina",
        description="Whether the exercise test brought on angina.",
        kind="categorical",
        codes={0: "No", 1: "Yes"},
    ),
    Feature(
        name="st_depression",
        raw_name="oldpeak",
        label="ST depression",
        description="ST segment depression induced by exercise relative to rest.",
        kind="numeric",
        unit="mm",
        valid_range=(0, 7),
    ),
    Feature(
        name="st_slope",
        raw_name="slp",
        label="ST slope",
        description="Slope of the ST segment at peak exercise.",
        kind="categorical",
        codes={2: "Upsloping", 1: "Flat", 0: "Downsloping"},
    ),
    Feature(
        name="major_vessels",
        raw_name="caa",
        label="Major vessels coloured",
        description="Number of major vessels (0-3) coloured by fluoroscopy.",
        kind="numeric",
        missing_codes=(4,),
        valid_range=(0, 3),
    ),
    Feature(
        name="thal",
        raw_name="thall",
        label="Thallium stress test",
        description=(
            "Result of the thallium myocardial perfusion scan: normal, a fixed defect "
            "(scar from a past infarct) or a reversible defect (reduced blood flow under "
            "stress). Often mislabelled 'thalassemia' in Kaggle notebooks."
        ),
        kind="categorical",
        codes={2: "Normal", 1: "Fixed defect", 3: "Reversible defect"},
        missing_codes=(0,),
    ),
)

TARGET = "has_disease"
RAW_TARGET = "output"
TARGET_LABEL = "Coronary artery disease"
TARGET_DESCRIPTION = (
    "Angiographic coronary artery disease: more than 50% narrowing of at least one major "
    "coronary artery. In the raw Kaggle file the target is inverted (output = 1 means no "
    "disease); here 1 means disease."
)
CLASS_LABELS = {0: "No disease", 1: "Disease"}

FEATURES_BY_NAME = {feature.name: feature for feature in FEATURES}
FEATURE_NAMES = [feature.name for feature in FEATURES]
NUMERIC_FEATURES = [feature.name for feature in FEATURES if feature.kind == "numeric"]
CATEGORICAL_FEATURES = [feature.name for feature in FEATURES if feature.kind == "categorical"]
