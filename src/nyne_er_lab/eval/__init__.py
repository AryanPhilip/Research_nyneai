"""Evaluation primitives."""

from .clusters import bcubed_f1
from .metrics import (
    MetricSummary,
    expected_calibration_error,
    optimize_threshold,
    summarize_predictions,
    threshold_sweep,
)
from .splits import assert_person_disjoint, examples_by_split, summarize_split_assignments

__all__ = [
    "MetricSummary",
    "assert_person_disjoint",
    "bcubed_f1",
    "examples_by_split",
    "expected_calibration_error",
    "optimize_threshold",
    "summarize_predictions",
    "summarize_split_assignments",
    "threshold_sweep",
]
