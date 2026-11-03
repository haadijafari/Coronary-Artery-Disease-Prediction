"""Tests for the fitted model artifact and the end-to-end training run."""

import json

import numpy as np
import pytest

from heart_disease.artifact import ModelArtifact
from heart_disease.models import candidate_models
from heart_disease.schema import FEATURE_NAMES, TARGET
from heart_disease.train import fit_procedure, main


def candidate(name):
    return next(c for c in candidate_models() if c.name == name)


@pytest.fixture(scope="module")
def xy(dataset):
    return dataset.drop(columns=TARGET), dataset[TARGET]


@pytest.fixture(scope="module")
def logistic(xy):
    return fit_procedure(candidate("Logistic regression"), *xy)


class TestArtifact:
    def test_explanation_adds_up_to_the_model_log_odds(self, logistic, xy):
        X = xy[0].head(10)
        proba = logistic.predict_proba(X)
        explanation = logistic.explain(X)
        log_odds = logistic.baseline_log_odds + explanation.sum(axis=1)
        np.testing.assert_allclose(log_odds, np.log(proba / (1 - proba)))

    def test_explanation_has_one_column_per_original_feature(self, logistic, xy):
        assert list(logistic.explain(xy[0].head(1)).columns) == FEATURE_NAMES

    def test_models_without_coefficients_cannot_be_explained(self, xy):
        baseline = fit_procedure(candidate("Baseline"), *xy)
        assert baseline.explain(xy[0].head(1)) is None

    def test_patient_exactly_at_the_threshold_is_flagged(self, logistic, xy):
        X = xy[0].head(5)
        proba = logistic.predict_proba(X)
        at_threshold = ModelArtifact(**{**vars(logistic), "threshold": float(proba[2])})
        assert at_threshold.predict(X)[2] == 1

    def test_save_and_load_round_trip(self, logistic, xy, tmp_path):
        path = tmp_path / "model.joblib"
        logistic.save(path)
        np.testing.assert_array_equal(
            ModelArtifact.load(path).predict_proba(xy[0]), logistic.predict_proba(xy[0])
        )


def test_training_cli_writes_a_loadable_model_and_metrics(tmp_path):
    main(["--output-dir", str(tmp_path), "--n-repeats", "1", "--fast"])

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    artifact = ModelArtifact.load(tmp_path / "model.joblib")

    assert metrics["selected_model"] == artifact.model_name == "Logistic regression"
    assert metrics["test"]["n_patients"] == 61
    assert artifact.n_patients == 302
    assert 0 < artifact.threshold < 1
    assert {row["metric"] for row in metrics["test"]["metrics"]} >= {"roc_auc", "sensitivity"}
