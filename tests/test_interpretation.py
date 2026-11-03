"""Tests for the statistics behind the exploratory and interpretation charts."""

import numpy as np
import pandas as pd
import pytest

from heart_disease.interpretation import disease_rate_by_category, odds_ratios, wilson_interval
from heart_disease.models import candidate_models
from heart_disease.schema import TARGET


class TestWilsonInterval:
    def test_half_of_ten(self):
        # Hand-computed Wilson score interval for 5/10 at 95%.
        assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)

    def test_zero_successes_has_a_lower_bound_of_zero_and_a_useful_upper_bound(self):
        assert wilson_interval(0, 4) == pytest.approx((0.0, 0.4899), abs=1e-4)


class TestDiseaseRateByCategory:
    frame = pd.DataFrame(
        {
            "colour": pd.Categorical(["A", "A", "B", None], categories=["B", "A", "Unused"]),
            TARGET: [1, 0, 1, 1],
        }
    )

    def test_rates_follow_the_category_order_and_skip_empty_and_missing(self):
        rates = disease_rate_by_category(self.frame, "colour")
        assert rates["category"].tolist() == ["B", "A"]
        assert rates["n"].tolist() == [1, 2]
        assert rates["rate"].tolist() == [1.0, 0.5]

    def test_each_rate_has_a_wilson_interval(self):
        rates = disease_rate_by_category(self.frame, "colour").set_index("category")
        assert (rates.loc["A", "lower"], rates.loc["A", "upper"]) == pytest.approx(
            wilson_interval(1, 2)
        )


@pytest.fixture(scope="module")
def fitted(dataset):
    X, y = dataset.drop(columns=TARGET), dataset[TARGET]
    logistic = next(c for c in candidate_models() if c.name == "Logistic regression")
    pipeline = logistic.pipeline().set_params(model__C=1.0).fit(X, y)
    return pipeline, odds_ratios(pipeline, X, y, n_resamples=50)


class TestOddsRatios:
    def test_point_estimate_is_the_exponentiated_coefficient_of_the_fitted_model(self, fitted):
        pipeline, table = fitted
        expected = np.exp(pipeline[-1].coef_[0])
        np.testing.assert_allclose(table["odds_ratio"], expected)

    def test_intervals_bracket_the_point_estimate(self, fitted):
        _, table = fitted
        assert (table["lower"] <= table["odds_ratio"]).all()
        assert (table["odds_ratio"] <= table["upper"]).all()

    def test_terms_are_labelled_with_their_original_feature(self, fitted):
        _, table = fitted
        row = table.set_index("term").loc["chest_pain_Asymptomatic"]
        assert row["feature"] == "chest_pain"
        assert row["level"] == "Asymptomatic"
