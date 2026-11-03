"""Candidate models and their hyperparameter grids."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from heart_disease.config import RANDOM_STATE
from heart_disease.features import build_preprocessor


@dataclass(frozen=True)
class Candidate:
    """A model to compare, with the grid searched for it."""

    name: str
    classifier: ClassifierMixin
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    complexity: int = 0
    """Rank used by the one-standard-error rule: lower means simpler."""

    def pipeline(self) -> Pipeline:
        """Preprocessing followed by the classifier, as one estimator."""
        return Pipeline([("preprocess", build_preprocessor()), ("model", self.classifier)])

    def search(
        self, n_splits: int = 5, random_state: int = RANDOM_STATE, n_jobs: int | None = None
    ):
        """The pipeline wrapped in a cross-validated grid search, if it has a grid."""
        if not self.param_grid:
            return self.pipeline()
        return GridSearchCV(
            self.pipeline(),
            {f"model__{key}": values for key, values in self.param_grid.items()},
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits, shuffle=True, random_state=random_state),
            n_jobs=n_jobs,
        )


def candidate_models(random_state: int = RANDOM_STATE) -> list[Candidate]:
    """Models from simplest to most flexible, plus a no-skill baseline."""
    return [
        Candidate(
            name="Baseline",
            classifier=DummyClassifier(strategy="prior"),
            complexity=0,
        ),
        Candidate(
            name="Logistic regression",
            classifier=LogisticRegression(max_iter=5_000),
            param_grid={"C": np.logspace(-3, 2, 11).tolist()},
            complexity=1,
        ),
        Candidate(
            name="SVM (RBF kernel)",
            classifier=CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False, cv=5),
            param_grid={"estimator__C": [0.1, 1.0, 10.0], "estimator__gamma": ["scale", 0.01]},
            complexity=2,
        ),
        Candidate(
            name="Random forest",
            classifier=RandomForestClassifier(n_estimators=200, random_state=random_state),
            param_grid={"min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", 0.5]},
            complexity=3,
        ),
        Candidate(
            name="Gradient boosting",
            classifier=HistGradientBoostingClassifier(max_iter=150, random_state=random_state),
            param_grid={"learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
            complexity=4,
        ),
    ]
