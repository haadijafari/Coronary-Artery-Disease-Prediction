"""Tests for loading, cleaning and validating the heart disease data."""

import numpy as np
import pandas as pd
import pytest

from heart_disease.data import DataValidationError, clean

# First row of data/raw/heart.csv, in the raw Kaggle encoding.
BASE_ROW = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trtbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalachh": 150,
    "exng": 0,
    "oldpeak": 2.3,
    "slp": 0,
    "caa": 0,
    "thall": 1,
    "output": 1,
}


def make_raw(*overrides: dict) -> pd.DataFrame:
    """Build a raw frame with one row per override dict, each based on BASE_ROW."""
    return pd.DataFrame([{**BASE_ROW, **override} for override in overrides or ({},)])


class TestDecoding:
    def test_chest_pain_uses_kaggle_codes_not_uci_codes(self):
        raw = make_raw({"cp": 0}, {"cp": 1}, {"cp": 2}, {"cp": 3})
        assert clean(raw)["chest_pain"].tolist() == [
            "Asymptomatic",
            "Atypical angina",
            "Non-anginal pain",
            "Typical angina",
        ]

    def test_rest_ecg_uses_kaggle_codes(self):
        raw = make_raw({"restecg": 0}, {"restecg": 1}, {"restecg": 2})
        assert clean(raw)["rest_ecg"].tolist() == ["LV hypertrophy", "Normal", "ST-T abnormality"]

    def test_st_slope_uses_kaggle_codes(self):
        raw = make_raw({"slp": 0}, {"slp": 1}, {"slp": 2})
        assert clean(raw)["st_slope"].tolist() == ["Downsloping", "Flat", "Upsloping"]

    def test_thal_uses_kaggle_codes(self):
        raw = make_raw({"thall": 1}, {"thall": 2}, {"thall": 3})
        assert clean(raw)["thal"].tolist() == ["Fixed defect", "Normal", "Reversible defect"]

    def test_binary_flags_are_decoded(self):
        cleaned = clean(make_raw({"sex": 0, "fbs": 0, "exng": 1}, {"sex": 1, "fbs": 1, "exng": 0}))
        assert cleaned["sex"].tolist() == ["Female", "Male"]
        assert cleaned["high_fasting_sugar"].tolist() == ["No", "Yes"]
        assert cleaned["exercise_angina"].tolist() == ["Yes", "No"]

    def test_categories_keep_every_level_even_when_absent_from_the_data(self):
        cleaned = clean(make_raw({"restecg": 1}))
        assert set(cleaned["rest_ecg"].cat.categories) == {
            "Normal",
            "ST-T abnormality",
            "LV hypertrophy",
        }


class TestMissingValues:
    def test_caa_code_4_is_missing(self):
        cleaned = clean(make_raw({"caa": 3}, {"caa": 4}))
        assert cleaned["major_vessels"].iloc[0] == 3
        assert np.isnan(cleaned["major_vessels"].iloc[1])

    def test_thall_code_0_is_missing(self):
        cleaned = clean(make_raw({"thall": 2}, {"thall": 0}))
        assert cleaned["thal"].iloc[0] == "Normal"
        assert pd.isna(cleaned["thal"].iloc[1])


class TestRows:
    def test_target_is_flipped_so_one_means_disease(self):
        cleaned = clean(make_raw({"output": 1}, {"output": 0}))
        assert cleaned["has_disease"].tolist() == [0, 1]

    def test_exact_duplicates_are_removed(self):
        cleaned = clean(make_raw({}, {}, {"age": 40}))
        assert len(cleaned) == 2

    def test_extreme_but_plausible_values_are_kept(self):
        # These rows were deleted by the old IQR outlier filter.
        cleaned = clean(make_raw({"chol": 564}, {"trtbps": 200}, {"thalachh": 71}))
        assert len(cleaned) == 3

    def test_input_frame_is_not_modified(self):
        raw = make_raw({"caa": 4})
        before = raw.copy()
        clean(raw)
        pd.testing.assert_frame_equal(raw, before)

    def test_columns_are_renamed_to_readable_names(self):
        assert list(clean(make_raw()).columns) == [
            "age",
            "sex",
            "chest_pain",
            "resting_bp",
            "cholesterol",
            "high_fasting_sugar",
            "rest_ecg",
            "max_heart_rate",
            "exercise_angina",
            "st_depression",
            "st_slope",
            "major_vessels",
            "thal",
            "has_disease",
        ]


class TestValidation:
    def test_unknown_category_code_is_rejected(self):
        with pytest.raises(DataValidationError, match="cp"):
            clean(make_raw({"cp": 7}))

    @pytest.mark.parametrize(
        ("column", "value"),
        [("age", 250), ("chol", 0), ("trtbps", 20), ("thalachh", 400), ("oldpeak", -1.0)],
    )
    def test_physiologically_impossible_value_is_rejected(self, column, value):
        with pytest.raises(DataValidationError, match=column):
            clean(make_raw({column: value}))

    def test_missing_column_is_rejected(self):
        with pytest.raises(DataValidationError, match="chol"):
            clean(make_raw().drop(columns="chol"))

    def test_target_outside_zero_and_one_is_rejected(self):
        with pytest.raises(DataValidationError, match="output"):
            clean(make_raw({"output": 2}))


class TestRealDataset:
    """Hand-checked facts about data/raw/heart.csv (see the analysis notebook)."""

    def test_one_duplicate_patient_is_dropped(self, dataset):
        assert len(dataset) == 302

    def test_class_counts(self, dataset):
        assert dataset["has_disease"].value_counts().to_dict() == {0: 164, 1: 138}

    def test_hidden_missing_values_are_exposed(self, dataset):
        assert dataset["major_vessels"].isna().sum() == 4
        assert dataset["thal"].isna().sum() == 2
        assert dataset.drop(columns=["major_vessels", "thal"]).notna().all().all()
