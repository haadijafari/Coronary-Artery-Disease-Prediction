"""Tests for metrics, threshold selection, model selection and bootstrapping."""

import numpy as np
import pandas as pd
import pytest

from heart_disease.evaluation import (
    bootstrap_metrics,
    choose_threshold,
    compare_models,
    metrics_at_threshold,
    select_model,
)
from heart_disease.models import candidate_models
from heart_disease.schema import TARGET


class TestChooseThreshold:
    y_true = np.array([1, 1, 1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.4, 0.2, 0.3, 0.1])

    def test_picks_highest_threshold_that_reaches_the_recall_target(self):
        # Catching 3 of the 4 cases requires flagging everyone at or above 0.4.
        assert choose_threshold(self.y_true, self.proba, min_recall=0.75) == pytest.approx(0.4)

    def test_full_recall_requires_the_lowest_positive_score(self):
        assert choose_threshold(self.y_true, self.proba, min_recall=1.0) == pytest.approx(0.2)


class TestMetricsAtThreshold:
    def test_confusion_based_metrics(self):
        # Predictions at 0.5 -> [1, 0, 1, 0]: one TP, one FN, one FP, one TN.
        metrics = metrics_at_threshold(
            np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1]), threshold=0.5
        )
        assert metrics["sensitivity"] == pytest.approx(0.5)
        assert metrics["specificity"] == pytest.approx(0.5)
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["npv"] == pytest.approx(0.5)
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_score_is_compared_inclusively_with_the_threshold(self):
        metrics = metrics_at_threshold(np.array([1, 0]), np.array([0.5, 0.2]), threshold=0.5)
        assert metrics["sensitivity"] == pytest.approx(1.0)

    def test_ranking_metrics_ignore_the_threshold(self):
        y_true, proba = np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.6, 0.1])
        low = metrics_at_threshold(y_true, proba, threshold=0.05)
        high = metrics_at_threshold(y_true, proba, threshold=0.95)
        # 3 of the 4 positive/negative pairs are ranked correctly.
        assert low["roc_auc"] == high["roc_auc"] == pytest.approx(0.75)


class TestSelectModel:
    def summary(self, simple_mean):
        return pd.DataFrame(
            {
                "model": ["Simple", "Complex"],
                "complexity": [1, 3],
                "mean": [simple_mean, 0.905],
                "se": [0.01, 0.01],
            }
        )

    def test_prefers_the_simpler_model_within_one_standard_error(self):
        assert select_model(self.summary(simple_mean=0.90)) == "Simple"

    def test_picks_the_best_model_when_the_simpler_one_is_clearly_worse(self):
        assert select_model(self.summary(simple_mean=0.85)) == "Complex"


class TestBootstrap:
    def test_perfect_predictions_have_degenerate_intervals(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        intervals = bootstrap_metrics(y_true, proba, threshold=0.5, n_resamples=200)
        row = intervals.set_index("metric").loc["sensitivity"]
        assert (row["estimate"], row["lower"], row["upper"]) == (1.0, 1.0, 1.0)

    def test_resampling_keeps_both_classes_in_tiny_samples(self):
        y_true = np.array([0] * 20 + [1] * 2)
        proba = np.linspace(0, 1, 22)
        intervals = bootstrap_metrics(y_true, proba, threshold=0.5, n_resamples=200)
        assert intervals["lower"].notna().all()

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        proba = np.clip(y_true * 0.3 + rng.uniform(0, 0.7, 200), 0, 1)
        intervals = bootstrap_metrics(y_true, proba, threshold=0.5, n_resamples=300)
        assert (intervals["lower"] <= intervals["estimate"]).all()
        assert (intervals["estimate"] <= intervals["upper"]).all()
        assert (intervals["lower"] < intervals["upper"]).all()


def test_compare_models_scores_every_model_on_every_fold(dataset):
    candidates = [c for c in candidate_models() if c.name in {"Baseline", "Logistic regression"}]
    scores = compare_models(
        candidates, dataset.drop(columns=TARGET), dataset[TARGET], n_splits=3, n_repeats=1
    )
    auc = scores[scores["metric"] == "roc_auc"].groupby("model")["value"]
    assert auc.size().to_dict() == {"Baseline": 3, "Logistic regression": 3}
    assert auc.mean()["Baseline"] == pytest.approx(0.5)
    assert auc.mean()["Logistic regression"] > 0.85
