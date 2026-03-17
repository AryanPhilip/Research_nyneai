"""Baseline matching models for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from nyne_er_lab.eval.metrics import MetricSummary, optimize_threshold, summarize_predictions
from nyne_er_lab.features.dataset import PairExample
from nyne_er_lab.features.extractor import PairFeatureExtractor


@dataclass(frozen=True)
class BaselineRun:
    name: str
    threshold: float
    metrics: MetricSummary
    scores: list[float]


LEXICAL_FEATURES = (
    "name_similarity",
    "alias_similarity",
    "headline_similarity",
    "bio_similarity",
    "same_source_type",
)


def _labels(examples: list[PairExample]) -> list[int]:
    return [example.label for example in examples]


def _feature_rows(examples: list[PairExample]) -> list[dict[str, float]]:
    return [example.features for example in examples]


def run_name_baseline(val_examples: list[PairExample], test_examples: list[PairExample]) -> BaselineRun:
    val_scores = [example.features["name_similarity"] for example in val_examples]
    threshold = optimize_threshold(_labels(val_examples), val_scores)
    test_scores = [example.features["name_similarity"] for example in test_examples]
    metrics = summarize_predictions(_labels(test_examples), test_scores, threshold)
    return BaselineRun(name="fuzzy_name", threshold=threshold, metrics=metrics, scores=test_scores)


def run_embedding_baseline(val_examples: list[PairExample], test_examples: list[PairExample]) -> BaselineRun:
    val_scores = [example.features["embedding_cosine"] for example in val_examples]
    threshold = optimize_threshold(_labels(val_examples), val_scores)
    test_scores = [example.features["embedding_cosine"] for example in test_examples]
    metrics = summarize_predictions(_labels(test_examples), test_scores, threshold)
    return BaselineRun(name="embedding_only", threshold=threshold, metrics=metrics, scores=test_scores)


def run_lexical_baseline(
    train_examples: list[PairExample],
    val_examples: list[PairExample],
    test_examples: list[PairExample],
    extractor: PairFeatureExtractor,
) -> BaselineRun:
    train_rows = [{key: row[key] for key in LEXICAL_FEATURES} for row in _feature_rows(train_examples)]
    val_rows = [{key: row[key] for key in LEXICAL_FEATURES} for row in _feature_rows(val_examples)]
    test_rows = [{key: row[key] for key in LEXICAL_FEATURES} for row in _feature_rows(test_examples)]

    train_matrix = extractor.vectorize_features(train_rows, include_embedding=False)
    val_matrix = extractor.vectorize_features(val_rows, include_embedding=False)
    test_matrix = extractor.vectorize_features(test_rows, include_embedding=False)

    model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    model.fit(train_matrix, _labels(train_examples))

    val_scores = model.predict_proba(val_matrix)[:, 1].tolist()
    threshold = optimize_threshold(_labels(val_examples), val_scores)
    test_scores = model.predict_proba(test_matrix)[:, 1].tolist()
    metrics = summarize_predictions(_labels(test_examples), test_scores, threshold)
    return BaselineRun(name="lexical_logistic", threshold=threshold, metrics=metrics, scores=test_scores)
