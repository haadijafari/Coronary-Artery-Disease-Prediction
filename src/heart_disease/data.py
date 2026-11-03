"""Load, clean and validate the heart disease data.

Cleaning is deliberately conservative. It only fixes what is provably wrong
with the file (duplicate rows, missing values hidden behind codes, opaque
category codes and an inverted target) and never deletes a patient for having
an unusual but plausible measurement. Anything physiologically impossible is
treated as a data error and rejected loudly instead of being silently dropped.
"""

from pathlib import Path

import pandas as pd

from heart_disease.config import RAW_DATA_PATH
from heart_disease.schema import FEATURES, RAW_TARGET, TARGET


class DataValidationError(ValueError):
    """Raised when the raw data violates the schema."""


def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Kaggle CSV without any changes."""
    return pd.read_csv(path)


def validate_raw(raw: pd.DataFrame) -> None:
    """Check the raw frame against the schema, reporting every problem at once."""
    required = [feature.raw_name for feature in FEATURES] + [RAW_TARGET]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    problems = []
    for feature in FEATURES:
        values = raw[feature.raw_name]
        values = values[values.notna() & ~values.isin(feature.missing_codes)]
        if feature.kind == "categorical":
            unknown = sorted(set(values) - set(feature.codes))
            if unknown:
                problems.append(f"{feature.raw_name}: unknown codes {unknown}")
        elif feature.valid_range is not None:
            low, high = feature.valid_range
            outside = sorted(values[(values < low) | (values > high)].unique())
            if outside:
                problems.append(
                    f"{feature.raw_name}: values outside plausible range [{low}, {high}]: {outside}"
                )

    bad_target = sorted(set(raw[RAW_TARGET].dropna()) - {0, 1})
    if bad_target or raw[RAW_TARGET].isna().any():
        problems.append(f"{RAW_TARGET}: target must be 0 or 1, found {bad_target or 'NaN'}")

    if problems:
        raise DataValidationError("; ".join(problems))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw Kaggle frame into an analysis-ready frame.

    - drops exact duplicate rows;
    - replaces sentinel codes (``caa = 4``, ``thall = 0``) with missing values;
    - decodes categorical codes into readable, ordered categories;
    - renames columns to readable names;
    - inverts the target so that ``has_disease = 1`` means disease.

    The input frame is never modified.
    """
    validate_raw(raw)
    deduplicated = raw.drop_duplicates().reset_index(drop=True)

    columns = {}
    for feature in FEATURES:
        values = deduplicated[feature.raw_name]
        if feature.missing_codes:
            values = values.mask(values.isin(feature.missing_codes))
        if feature.kind == "categorical":
            values = pd.Categorical(values.map(feature.codes), categories=feature.categories)
        columns[feature.name] = values
    columns[TARGET] = 1 - deduplicated[RAW_TARGET]

    return pd.DataFrame(columns)


def load_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return the cleaned frame."""
    return clean(load_raw(path))
