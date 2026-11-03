"""End-to-end training: compare models, evaluate the chosen one, save it.

The procedure is evaluated honestly and then deployed:

1. Hold out a stratified test set that nothing below looks at.
2. Compare candidates with nested cross-validation on the training set.
3. Pick the simplest model within one standard error of the best ROC AUC.
4. Tune it on the training set and choose the decision threshold from
   out-of-fold predictions, so that it catches ``TARGET_RECALL`` of cases.
5. Score that model once on the held-out test set, with bootstrap intervals.
6. Re-run step 4 on all patients to produce the deployed model.

Run with ``uv run heart-train``.
"""

import argparse
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, train_test_split

from heart_disease import __version__
from heart_disease.artifact import ModelArtifact
from heart_disease.config import (
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_RECALL,
    TEST_SIZE,
)
from heart_disease.data import load_dataset
from heart_disease.evaluation import (
    bootstrap_metrics,
    choose_threshold,
    compare_models,
    out_of_fold_proba,
    select_model,
    summarize_comparison,
)
from heart_disease.models import Candidate, candidate_models
from heart_disease.schema import TARGET

FAST_CANDIDATES = {"Baseline", "Logistic regression"}


def split(
    dataset: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split of the cleaned dataset."""
    X, y = dataset.drop(columns=TARGET), dataset[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def fit_procedure(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    target_recall: float = TARGET_RECALL,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> ModelArtifact:
    """Tune a candidate, pick its threshold out-of-fold, and fit it on ``X``."""
    search = candidate.search(random_state=random_state, n_jobs=n_jobs).fit(X, y)
    if isinstance(search, GridSearchCV):
        pipeline = search.best_estimator_
        params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    else:
        pipeline, params = search, {}

    oof = out_of_fold_proba(clone(pipeline), X, y, random_state=random_state)
    return ModelArtifact(
        pipeline=pipeline,
        threshold=choose_threshold(y, oof, target_recall),
        model_name=candidate.name,
        params=params,
        feature_means=pipeline[:-1].transform(X).mean(),
        n_patients=len(X),
        target_recall=target_recall,
        versions={
            "heart-disease": __version__,
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
    )


@dataclass
class TrainingResult:
    scores: pd.DataFrame
    summary: pd.DataFrame
    evaluated: ModelArtifact
    test_proba: np.ndarray
    y_test: pd.Series
    test_metrics: pd.DataFrame
    deployed: ModelArtifact
    n_splits: int
    n_repeats: int


def run(
    data_path: str | Path = RAW_DATA_PATH,
    candidates: Sequence[Candidate] | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    n_jobs: int | None = -1,
) -> TrainingResult:
    """Execute the full procedure described in the module docstring."""
    candidates = list(candidates or candidate_models())
    dataset = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split(dataset)

    scores = compare_models(
        candidates, X_train, y_train, n_splits=n_splits, n_repeats=n_repeats, n_jobs=n_jobs
    )
    summary = summarize_comparison(scores, candidates)
    chosen = next(c for c in candidates if c.name == select_model(summary))

    evaluated = fit_procedure(chosen, X_train, y_train, n_jobs=n_jobs)
    test_proba = evaluated.predict_proba(X_test)
    test_metrics = bootstrap_metrics(y_test, test_proba, evaluated.threshold)

    X_all, y_all = dataset.drop(columns=TARGET), dataset[TARGET]
    deployed = fit_procedure(chosen, X_all, y_all, n_jobs=n_jobs)

    return TrainingResult(
        scores, summary, evaluated, test_proba, y_test, test_metrics, deployed, n_splits, n_repeats
    )


def _rounded(value):
    if isinstance(value, float | np.floating):
        return round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: _rounded(v) for k, v in row.items()} for row in frame.to_dict("records")]


def metrics_report(result: TrainingResult) -> dict:
    """A JSON-serialisable summary of the run, read by the README and the app."""
    deployed = result.deployed
    return {
        "selected_model": deployed.model_name,
        "target_recall": deployed.target_recall,
        "cross_validation": {
            "scheme": f"nested, {result.n_repeats}x repeated stratified {result.n_splits}-fold",
            "metric": "roc_auc",
            "summary": _records(result.summary[["model", "mean", "std", "se"]]),
        },
        "test": {
            "n_patients": len(result.y_test),
            "n_disease": int(result.y_test.sum()),
            "threshold": _rounded(result.evaluated.threshold),
            "params": {k: _rounded(v) for k, v in result.evaluated.params.items()},
            "metrics": _records(result.test_metrics),
        },
        "deployed": {
            "n_patients": deployed.n_patients,
            "threshold": _rounded(deployed.threshold),
            "params": {k: _rounded(v) for k, v in deployed.params.items()},
        },
        "versions": deployed.versions,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the heart disease model.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="raw heart.csv path")
    parser.add_argument(
        "--output-dir", type=Path, default=MODEL_PATH.parent, help="where to write outputs"
    )
    parser.add_argument("--n-repeats", type=int, default=5, help="repeats of the outer CV")
    parser.add_argument(
        "--fast", action="store_true", help="compare only the baseline and logistic regression"
    )
    args = parser.parse_args(argv)

    candidates = [c for c in candidate_models() if not args.fast or c.name in FAST_CANDIDATES]
    result = run(args.data, candidates=candidates, n_repeats=args.n_repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.deployed.save(args.output_dir / MODEL_PATH.name)
    report = metrics_report(result)
    (args.output_dir / METRICS_PATH.name).write_text(json.dumps(report, indent=2) + "\n")

    print(result.summary.to_string(index=False, float_format="{:.3f}".format))
    print(f"\nSelected: {report['selected_model']}  (threshold {report['deployed']['threshold']})")
    print(pd.DataFrame(report["test"]["metrics"]).to_string(index=False))


if __name__ == "__main__":
    main()
