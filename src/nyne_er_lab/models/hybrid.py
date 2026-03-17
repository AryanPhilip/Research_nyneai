"""Hybrid matcher with calibration, abstention, and feature ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from nyne_er_lab.eval.metrics import MetricSummary, optimize_threshold, summarize_predictions
from nyne_er_lab.features.dataset import PairExample
from nyne_er_lab.features.extractor import PairFeatureExtractor


LEXICAL_FEATURES = (
    "name_similarity",
    "alias_similarity",
    "headline_similarity",
    "bio_similarity",
    "same_source_type",
)
STRUCTURED_FEATURES = (
    "shared_domain_count",
    "org_overlap_count",
    "org_jaccard",
    "topic_overlap_count",
    "topic_jaccard",
    "location_overlap",
    "temporal_overlap_count",
    "temporal_distance",
    "location_conflict",
)
FULL_FEATURES = LEXICAL_FEATURES + STRUCTURED_FEATURES + ("embedding_cosine",)


class LLMAdjudicator(Protocol):
    """Optional adjudication interface for uncertain examples."""

    def adjudicate(self, example: PairExample, score: float) -> float | None:
        """Return an adjusted score or None to leave the model output unchanged."""


class NullLLMAdjudicator:
    """Default no-op adjudicator used in tests and offline runs."""

    def adjudicate(self, example: PairExample, score: float) -> float | None:
        return None


@dataclass(frozen=True)
class HybridRun:
    """Full hybrid-model benchmark result."""

    name: str
    threshold: float
    match_threshold: float
    non_match_threshold: float
    raw_metrics: MetricSummary
    calibrated_metrics: MetricSummary
    raw_brier: float
    calibrated_brier: float
    accepted_precision: float
    accepted_recall: float
    abstain_rate: float
    raw_scores: list[float]
    calibrated_scores: list[float]
    decisions: list[str]


@dataclass
class TrainedHybridMatcher:
    """Trained pairwise matcher used by clustering and demo layers."""

    model: LogisticRegression
    calibrate: Callable[[list[float]], list[float]]
    feature_names: tuple[str, ...]
    threshold: float
    match_threshold: float
    non_match_threshold: float

    def score_examples(
        self,
        examples: list[PairExample],
        extractor: PairFeatureExtractor,
        *,
        adjudicator: LLMAdjudicator | None = None,
    ) -> tuple[list[float], list[str]]:
        adjudicator = adjudicator or NullLLMAdjudicator()
        matrix = extractor.vectorize_features(_feature_rows(examples), feature_names=self.feature_names)
        raw_scores = self.model.predict_proba(matrix)[:, 1].tolist()
        calibrated_scores = self.calibrate(raw_scores)

        final_scores: list[float] = []
        decisions: list[str] = []
        for example, score in zip(examples, calibrated_scores):
            adjusted = adjudicator.adjudicate(example, score)
            score = adjusted if adjusted is not None else score
            final_scores.append(float(score))

            if _contradiction_veto(example, score, self.match_threshold):
                decisions.append("abstain")
            elif score >= self.match_threshold:
                decisions.append("match")
            elif score <= self.non_match_threshold:
                decisions.append("non_match")
            else:
                decisions.append("abstain")
        return final_scores, decisions


@dataclass(frozen=True)
class AblationResult:
    """Named ablation outcome for comparison against the full hybrid model."""

    name: str
    metrics: MetricSummary


def _labels(examples: list[PairExample]) -> list[int]:
    return [example.label for example in examples]


def _feature_rows(examples: list[PairExample]) -> list[dict[str, float]]:
    return [example.features for example in examples]


def _matrix(
    extractor: PairFeatureExtractor,
    examples: list[PairExample],
    feature_names: tuple[str, ...],
) -> list[list[float]]:
    return extractor.vectorize_features(_feature_rows(examples), feature_names=feature_names).tolist()


def _contradiction_veto(example: PairExample, score: float, match_threshold: float) -> bool:
    features = example.features
    return (
        features["location_conflict"] >= 1.0
        and features["org_overlap_count"] == 0.0
        and features["shared_domain_count"] == 0.0
        and features["topic_overlap_count"] <= 1.0
        and (
            score >= match_threshold
            or features["name_similarity"] >= 0.98
        )
    )


def _abstain_stats(labels: list[int], decisions: list[str]) -> tuple[float, float, float]:
    match_indices = [index for index, decision in enumerate(decisions) if decision == "match"]
    abstain_count = sum(1 for decision in decisions if decision == "abstain")
    if not match_indices:
        return 0.0, 0.0, abstain_count / max(len(decisions), 1)

    true_positives = sum(1 for index in match_indices if labels[index] == 1)
    accepted_precision = true_positives / len(match_indices)
    accepted_recall = true_positives / max(sum(labels), 1)
    abstain_rate = abstain_count / max(len(decisions), 1)
    return accepted_precision, accepted_recall, abstain_rate


def _optimize_abstain_band(
    examples: list[PairExample],
    calibrated_scores: list[float],
    match_floor: float,
    non_match_floor: float,
) -> tuple[float, float]:
    labels = _labels(examples)
    best: tuple[float, float, float, float] | None = None

    lower_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    upper_grid = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    for lower in lower_grid:
        for upper in upper_grid:
            if lower >= upper:
                continue
            decisions = [
                "match" if score >= upper else "non_match" if score <= lower else "abstain"
                for score in calibrated_scores
            ]
            if "match" not in decisions:
                continue
            accepted_precision, accepted_recall, abstain_rate = _abstain_stats(labels, decisions)
            objective = accepted_precision + (0.25 * accepted_recall) - (0.1 * abstain_rate)
            candidate = (objective, accepted_precision, accepted_recall, abstain_rate)
            if best is None or candidate > best:
                best = candidate
                best_thresholds = (lower, upper)

    if best is None:
        return non_match_floor, match_floor
    return best_thresholds


def _fit_best_calibrator(
    raw_val_scores: list[float],
    labels: list[int],
    random_state: int,
) -> tuple[list[float], Callable[[list[float]], list[float]]]:
    """Choose the best-behaved calibration mapping on the validation set."""

    candidates: list[tuple[float, list[float], callable]] = []

    raw_brier = float(brier_score_loss(labels, raw_val_scores))
    candidates.append((raw_brier, raw_val_scores, lambda scores: scores))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_val_scores, labels)
    iso_scores = isotonic.transform(raw_val_scores).tolist()
    iso_brier = float(brier_score_loss(labels, iso_scores))
    candidates.append((iso_brier, iso_scores, lambda scores: isotonic.transform(scores).tolist()))

    sigmoid = LogisticRegression(random_state=random_state)
    sigmoid.fit([[score] for score in raw_val_scores], labels)
    sigmoid_scores = sigmoid.predict_proba([[score] for score in raw_val_scores])[:, 1].tolist()
    sigmoid_brier = float(brier_score_loss(labels, sigmoid_scores))
    candidates.append(
        (
            sigmoid_brier,
            sigmoid_scores,
            lambda scores: sigmoid.predict_proba([[score] for score in scores])[:, 1].tolist(),
        )
    )

    _, best_scores, transform = min(candidates, key=lambda item: item[0])
    return best_scores, transform


def run_hybrid_matcher(
    train_examples: list[PairExample],
    val_examples: list[PairExample],
    test_examples: list[PairExample],
    extractor: PairFeatureExtractor,
    *,
    feature_names: tuple[str, ...] = FULL_FEATURES,
    random_state: int = 42,
    adjudicator: LLMAdjudicator | None = None,
) -> HybridRun:
    """Train, calibrate, and evaluate the full hybrid matcher."""

    train_matrix = extractor.vectorize_features(_feature_rows(train_examples), feature_names=feature_names)
    val_matrix = extractor.vectorize_features(_feature_rows(val_examples), feature_names=feature_names)
    test_matrix = extractor.vectorize_features(_feature_rows(test_examples), feature_names=feature_names)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    model.fit(train_matrix, _labels(train_examples))

    raw_val_scores = model.predict_proba(val_matrix)[:, 1].tolist()
    raw_test_scores = model.predict_proba(test_matrix)[:, 1].tolist()
    raw_threshold = optimize_threshold(_labels(val_examples), raw_val_scores)
    raw_metrics = summarize_predictions(_labels(test_examples), raw_test_scores, raw_threshold)

    calibrated_val_scores, calibrate = _fit_best_calibrator(raw_val_scores, _labels(val_examples), random_state)
    calibrated_test_scores = calibrate(raw_test_scores)
    calibrated_threshold = optimize_threshold(_labels(val_examples), calibrated_val_scores)
    calibrated_metrics = summarize_predictions(_labels(test_examples), calibrated_test_scores, calibrated_threshold)

    non_match_threshold, match_threshold = _optimize_abstain_band(
        val_examples,
        calibrated_val_scores,
        match_floor=max(calibrated_threshold, 0.6),
        non_match_floor=min(calibrated_threshold, 0.4),
    )

    trained_matcher = TrainedHybridMatcher(
        model=model,
        calibrate=calibrate,
        feature_names=feature_names,
        threshold=calibrated_threshold,
        match_threshold=match_threshold,
        non_match_threshold=non_match_threshold,
    )
    final_scores, decisions = trained_matcher.score_examples(test_examples, extractor, adjudicator=adjudicator)

    labels = _labels(test_examples)
    accepted_precision, accepted_recall, abstain_rate = _abstain_stats(labels, decisions)

    return HybridRun(
        name="hybrid_gradient_boosting",
        threshold=calibrated_threshold,
        match_threshold=match_threshold,
        non_match_threshold=non_match_threshold,
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        raw_brier=float(brier_score_loss(labels, raw_test_scores)),
        calibrated_brier=float(brier_score_loss(labels, calibrated_test_scores)),
        accepted_precision=accepted_precision,
        accepted_recall=accepted_recall,
        abstain_rate=abstain_rate,
        raw_scores=raw_test_scores,
        calibrated_scores=final_scores,
        decisions=decisions,
    )


def run_feature_ablations(
    train_examples: list[PairExample],
    val_examples: list[PairExample],
    test_examples: list[PairExample],
    extractor: PairFeatureExtractor,
) -> list[AblationResult]:
    """Evaluate major feature-group ablations against the hybrid model."""

    runs = [
        ("full", FULL_FEATURES),
        ("no_embedding", LEXICAL_FEATURES + STRUCTURED_FEATURES),
        ("no_structured", LEXICAL_FEATURES + ("embedding_cosine",)),
    ]
    results: list[AblationResult] = []
    for name, features in runs:
        run = run_hybrid_matcher(
            train_examples,
            val_examples,
            test_examples,
            extractor,
            feature_names=features,
        )
        results.append(AblationResult(name=name, metrics=run.calibrated_metrics))
    return results


def train_hybrid_matcher(
    train_examples: list[PairExample],
    val_examples: list[PairExample],
    extractor: PairFeatureExtractor,
    *,
    feature_names: tuple[str, ...] = FULL_FEATURES,
    random_state: int = 42,
) -> TrainedHybridMatcher:
    """Train a reusable matcher for downstream clustering and demo rendering."""

    train_matrix = extractor.vectorize_features(_feature_rows(train_examples), feature_names=feature_names)
    val_matrix = extractor.vectorize_features(_feature_rows(val_examples), feature_names=feature_names)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    model.fit(train_matrix, _labels(train_examples))

    raw_val_scores = model.predict_proba(val_matrix)[:, 1].tolist()
    calibrated_val_scores, calibrate = _fit_best_calibrator(raw_val_scores, _labels(val_examples), random_state)
    calibrated_threshold = optimize_threshold(_labels(val_examples), calibrated_val_scores)
    non_match_threshold, match_threshold = _optimize_abstain_band(
        val_examples,
        calibrated_val_scores,
        match_floor=max(calibrated_threshold, 0.6),
        non_match_floor=min(calibrated_threshold, 0.4),
    )
    return TrainedHybridMatcher(
        model=model,
        calibrate=calibrate,
        feature_names=feature_names,
        threshold=calibrated_threshold,
        match_threshold=match_threshold,
        non_match_threshold=non_match_threshold,
    )
