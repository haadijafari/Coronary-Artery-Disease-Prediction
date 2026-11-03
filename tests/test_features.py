"""Tests for the preprocessing pipeline."""

import numpy as np
import pytest

from heart_disease.features import build_preprocessor
from heart_disease.schema import TARGET


@pytest.fixture(scope="module")
def features(dataset):
    return dataset.drop(columns=TARGET)


def test_missing_values_are_imputed(features):
    transformed = build_preprocessor().fit_transform(features)
    assert features["major_vessels"].isna().any()
    assert not np.isnan(transformed.to_numpy()).any()


def test_reference_category_of_each_feature_is_dropped(features):
    columns = build_preprocessor().fit(features).get_feature_names_out()
    assert "chest_pain_Typical angina" not in columns
    assert "chest_pain_Asymptomatic" in columns
    assert "sex_Female" not in columns
    assert "sex_Male" in columns


def test_rare_category_gets_a_column_even_when_absent_from_training_rows(features):
    without_st_t = features[features["rest_ecg"] != "ST-T abnormality"]
    columns = build_preprocessor().fit(without_st_t).get_feature_names_out()
    assert "rest_ecg_ST-T abnormality" in columns
    assert len(columns) == 18


def test_numeric_features_are_standardised_with_training_statistics(features):
    transformed = build_preprocessor().fit_transform(features)
    assert transformed["age"].mean() == pytest.approx(0, abs=1e-9)
    assert transformed["age"].std(ddof=0) == pytest.approx(1)
