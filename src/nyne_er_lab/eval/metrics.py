"""Evaluation helpers for pairwise entity resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


@dataclass(frozen=True)
class MetricSummary:
    precision: float
    recall: float
    f1: float
    average_precision: float


def summarize_predictions(labels: list[int], scores: list[float], threshold: float) -> MetricSummary:
    """Compute pairwise metrics for a thresholded score output."""

    predictions = [int(score >= threshold) for score in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    avg_precision = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.0
    return MetricSummary(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        average_precision=float(avg_precision),
    )


def threshold_sweep(labels: list[int], scores: list[float], n_points: int = 50) -> list[dict]:
    """Return precision/recall/F1 at evenly spaced thresholds across the score range."""

    if not scores or not labels:
        return []

    lo, hi = min(scores), max(scores)
    if lo == hi:
        lo, hi = 0.0, 1.0
    thresholds = np.linspace(lo, hi, n_points)
    results: list[dict] = []
    for threshold in thresholds:
        summary = summarize_predictions(labels, scores, float(threshold))
        results.append({
            "threshold": round(float(threshold), 4),
            "precision": summary.precision,
            "recall": summary.recall,
            "f1": summary.f1,
        })
    return results


def optimize_threshold(labels: list[int], scores: list[float]) -> float:
    """Choose the F1-maximizing threshold on a validation set."""

    if not scores:
        return 0.5

    candidate_thresholds = sorted(set(np.clip(scores, 0.0, 1.0)))
    candidate_thresholds.extend([0.35, 0.5, 0.65, 0.8])

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        summary = summarize_predictions(labels, scores, threshold)
        if summary.f1 > best_f1:
            best_f1 = summary.f1
            best_threshold = float(threshold)
    return best_threshold
