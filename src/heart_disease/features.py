"""Preprocessing shared by every model.

Everything that learns from data (imputation values, scaling statistics) lives
inside this transformer, so it is refit on the training folds only and can
never leak information from validation or test rows.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease.schema import CATEGORICAL_FEATURES, FEATURES_BY_NAME, NUMERIC_FEATURES

_TERMS = {name: (name, None) for name in NUMERIC_FEATURES} | {
    f"{name}_{category}": (name, category)
    for name in CATEGORICAL_FEATURES
    for category in FEATURES_BY_NAME[name].categories
}


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale numeric features and one-hot encode categorical ones.

    Categorical features are encoded against the full list of categories from
    the schema, so rare levels (only 4 patients have an ST-T abnormality) keep
    their column even when a training fold happens to contain none of them.
    The first category of each feature, the clinically normal one where there
    is one, is dropped and acts as the reference level.
    """
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(
            categories=[FEATURES_BY_NAME[name].categories for name in CATEGORICAL_FEATURES],
            drop="first",
            sparse_output=False,
        ),
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def term_origin(term: str) -> tuple[str, str | None]:
    """Split a transformed column into its feature and category.

    ``"chest_pain_Asymptomatic"`` gives ``("chest_pain", "Asymptomatic")`` and a
    numeric column such as ``"age"`` gives ``("age", None)``.
    """
    try:
        return _TERMS[term]
    except KeyError:
        raise KeyError(f"{term!r} is not a column produced by the preprocessor") from None
