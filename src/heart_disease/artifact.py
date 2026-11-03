"""The fitted model as it is saved, loaded and served."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from heart_disease.features import term_origin
from heart_disease.schema import FEATURE_NAMES


@dataclass
class ModelArtifact:
    """A fitted pipeline together with its decision threshold and provenance."""

    pipeline: Pipeline
    threshold: float
    """Patients with a predicted probability at or above this are flagged."""
    model_name: str
    params: dict[str, Any]
    feature_means: pd.Series
    """Mean of each transformed feature over the training rows (explanation reference)."""
    n_patients: int
    target_recall: float
    versions: dict[str, str] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of coronary artery disease for each row."""
        return self.pipeline.predict_proba(X[FEATURE_NAMES])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 for patients at or above the decision threshold, else 0."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    @property
    def _linear_model(self):
        model = self.pipeline[-1]
        return model if hasattr(model, "coef_") else None

    @property
    def baseline_log_odds(self) -> float | None:
        """Log-odds of disease for a patient with average (transformed) features."""
        model = self._linear_model
        if model is None:
            return None
        return float(model.intercept_[0] + model.coef_[0] @ self.feature_means.to_numpy())

    def explain(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Each feature's contribution to the log-odds, relative to an average patient.

        For a linear model the log-odds decompose exactly:
        ``log-odds = baseline_log_odds + sum(contributions)``. One-hot columns
        are summed back into the feature they came from. Returns ``None`` for
        models without coefficients.
        """
        model = self._linear_model
        if model is None:
            return None
        transformed = self.pipeline[:-1].transform(X[FEATURE_NAMES])
        contributions = (transformed - self.feature_means) * model.coef_[0]
        grouped = contributions.T.groupby(lambda term: term_origin(term)[0]).sum().T
        return grouped[FEATURE_NAMES]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        trained_with = artifact.versions.get("scikit-learn")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Model was trained with scikit-learn {trained_with} but "
                f"{sklearn.__version__} is installed; retrain with `uv run heart-train`.",
                stacklevel=2,
            )
        return artifact
