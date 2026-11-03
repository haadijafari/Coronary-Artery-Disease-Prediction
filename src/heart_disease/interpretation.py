"""Statistics behind the exploratory and interpretation charts."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from heart_disease.config import RANDOM_STATE
from heart_disease.features import term_origin
from heart_disease.schema import TARGET


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Unlike the textbook normal approximation it stays inside [0, 1] and is
    honest for small groups, such as the four patients with an ST-T abnormality.
    """
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def disease_rate_by_category(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Share of patients with disease in each level of ``feature``, with Wilson intervals.

    Levels follow the categorical order (or sorted values for numeric
    features); empty levels and missing values are left out.
    """
    grouped = (
        frame.dropna(subset=[feature])
        .groupby(feature, observed=True, sort=True)[TARGET]
        .agg(n="size", cases="sum")
        .reset_index()
        .rename(columns={feature: "category"})
    )
    grouped["rate"] = grouped["cases"] / grouped["n"]
    intervals = [wilson_interval(c, n) for c, n in zip(grouped["cases"], grouped["n"], strict=True)]
    grouped[["lower", "upper"]] = intervals
    return grouped


def odds_ratios(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Odds ratios of a fitted logistic pipeline with percentile bootstrap intervals.

    Numeric terms are per one standard deviation increase; categorical terms
    are relative to the feature's reference (first) category. Intervals come
    from refitting the same pipeline, hyperparameters included, on bootstrap
    resamples of ``X``. They describe the stability of the regularised fit
    rather than classical Wald intervals.
    """
    rng = np.random.default_rng(random_state)
    terms = pipeline[:-1].get_feature_names_out()
    resampled = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(X), len(X))
        refit = clone(pipeline).fit(X.iloc[index], y.iloc[index])
        resampled.append(refit[-1].coef_[0])
    resampled = np.exp(np.array(resampled))

    tail = (1 - confidence) / 2 * 100
    origins = [term_origin(term) for term in terms]
    return pd.DataFrame(
        {
            "term": terms,
            "feature": [feature for feature, _ in origins],
            "level": [level for _, level in origins],
            "odds_ratio": np.exp(pipeline[-1].coef_[0]),
            "lower": np.percentile(resampled, tail, axis=0),
            "upper": np.percentile(resampled, 100 - tail, axis=0),
        }
    )
