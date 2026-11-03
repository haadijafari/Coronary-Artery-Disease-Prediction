"""Model comparison, threshold selection and uncertainty estimates.

Disease (``has_disease = 1``) is the positive class throughout, so
"sensitivity" is the share of disease cases the model catches.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    make_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from heart_disease.config import RANDOM_STATE
from heart_disease.models import Candidate

METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
    "brier": "Brier score",
    "sensitivity": "Sensitivity (recall)",
    "specificity": "Specificity",
    "precision": "Precision (PPV)",
    "npv": "Negative predictive value",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1 score",
}

_CV_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "brier": "neg_brier_score",
    "sensitivity": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
    "balanced_accuracy": "balanced_accuracy",
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def metrics_at_threshold(y_true, proba, threshold: float) -> dict[str, float]:
    """Ranking metrics plus confusion-matrix metrics at ``proba >= threshold``."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    predicted = proba >= threshold
    actual = y_true == 1

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)

    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": _safe_ratio(tp, tp + fp),
        "npv": _safe_ratio(tn, tn + fn),
        "accuracy": (tp + tn) / len(y_true),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def choose_threshold(y_true, proba, min_recall: float) -> float:
    """Highest threshold that still flags at least ``min_recall`` of the disease cases.

    A higher threshold means fewer false alarms, so this is the most specific
    operating point that meets the sensitivity requirement.
    """
    y_true = np.asarray(y_true)
    positive_scores = np.sort(np.asarray(proba)[y_true == 1])[::-1]
    # Subtract a tiny epsilon so that e.g. 0.9 * 110 = 99.00000000000001 needs 99 cases, not 100.
    needed = max(1, math.ceil(min_recall * len(positive_scores) - 1e-9))
    return float(positive_scores[needed - 1])


def bootstrap_metrics(
    y_true,
    proba,
    threshold: float,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Percentile bootstrap intervals for every metric in ``metrics_at_threshold``.

    Resampling is stratified by class so every resample contains both
    classes, which keeps ROC AUC and sensitivity defined even for small sets.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(random_state)
    positives, negatives = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)

    resampled = []
    for _ in range(n_resamples):
        index = np.concatenate(
            [rng.choice(positives, positives.size), rng.choice(negatives, negatives.size)]
        )
        resampled.append(metrics_at_threshold(y_true[index], proba[index], threshold))
    resampled = pd.DataFrame(resampled)

    tail = (1 - confidence) / 2 * 100
    estimate = metrics_at_threshold(y_true, proba, threshold)
    return pd.DataFrame(
        {
            "metric": list(estimate),
            "estimate": list(estimate.values()),
            "lower": [np.nanpercentile(resampled[m], tail) for m in estimate],
            "upper": [np.nanpercentile(resampled[m], 100 - tail) for m in estimate],
        }
    )


def compare_models(
    candidates: Sequence[Candidate],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Nested cross-validation: tune each candidate inside every outer training fold.

    Returns one row per model, outer fold and metric. Threshold-dependent
    metrics use the default 0.5 cut-off here; the operating threshold is
    chosen later, for the selected model only.
    """
    outer = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    rows = []
    for candidate in candidates:
        results = cross_validate(
            candidate.search(random_state=random_state, n_jobs=1),
            X,
            y,
            cv=outer,
            scoring=_CV_SCORING,
            n_jobs=n_jobs,
        )
        for metric in _CV_SCORING:
            values = results[f"test_{metric}"]
            if metric == "brier":
                values = -values
            rows.extend(
                {"model": candidate.name, "fold": fold, "metric": metric, "value": value}
                for fold, value in enumerate(values)
            )
    return pd.DataFrame(rows)


def summarize_comparison(
    scores: pd.DataFrame, candidates: Sequence[Candidate], metric: str = "roc_auc"
) -> pd.DataFrame:
    """Mean, standard deviation and standard error of one metric per model.

    The standard error treats the repeated folds as independent, which they
    are not, so it is optimistic; it is used only as the tolerance of the
    one-standard-error rule, not as a confidence interval.
    """
    complexity = {candidate.name: candidate.complexity for candidate in candidates}
    summary = (
        scores[scores["metric"] == metric]
        .groupby("model", sort=False)["value"]
        .agg(mean="mean", std="std", folds="size")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["folds"])
    summary["complexity"] = summary["model"].map(complexity)
    return summary.sort_values("mean", ascending=False, ignore_index=True)


def select_model(summary: pd.DataFrame) -> str:
    """One-standard-error rule: the simplest model within one SE of the best mean."""
    best = summary.loc[summary["mean"].idxmax()]
    eligible = summary[summary["mean"] >= best["mean"] - best["se"]]
    return str(eligible.sort_values("complexity").iloc[0]["model"])


def out_of_fold_proba(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Disease probabilities for each row from a model that never saw that row."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
