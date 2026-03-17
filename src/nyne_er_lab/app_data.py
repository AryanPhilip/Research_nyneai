"""Data loading and caching layer for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from nyne_er_lab.blocking.blocker import blocking_rule_stats
from nyne_er_lab.cluster.resolver import ResolvedPair
from nyne_er_lab.demo.builder import _benchmark_context
from nyne_er_lab.features.dataset import PairExample
from nyne_er_lab.features.extractor import PairFeatureExtractor
from nyne_er_lab.models.hybrid import HybridRun, TrainedHybridMatcher
from nyne_er_lab.schemas import CanonicalIdentity, ProfileRecord


@dataclass
class DashboardData:
    profiles: list[ProfileRecord]
    profile_lookup: dict[str, ProfileRecord]
    example_lookup: dict[tuple[str, str], PairExample]
    extractor: PairFeatureExtractor
    matcher: TrainedHybridMatcher
    hybrid_run: HybridRun
    metrics: list[dict]
    cluster_f1: float
    ablations: list[dict]
    confusion_slices: dict
    identities: list[CanonicalIdentity]
    resolved_pairs: list[ResolvedPair]
    train_examples: list[PairExample]
    val_examples: list[PairExample]
    test_examples: list[PairExample]
    feature_importance: list[tuple[str, float]]
    blocking_stats: dict[str, dict]
    test_labels: list[int]
    test_scores: list[float]


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


@st.cache_resource(show_spinner="Running entity resolution pipeline...")
def load_dashboard_data() -> DashboardData:
    benchmark = _benchmark_context()

    profiles: list[ProfileRecord] = benchmark["profiles"]
    profile_lookup = {p.profile_id: p for p in profiles}

    train_examples: list[PairExample] = benchmark["train_examples"]
    val_examples: list[PairExample] = benchmark["val_examples"]
    test_examples: list[PairExample] = benchmark["test_examples"]

    all_examples = train_examples + val_examples + test_examples
    example_lookup: dict[tuple[str, str], PairExample] = {}
    for ex in all_examples:
        key = _pair_key(ex.left_profile_id, ex.right_profile_id)
        example_lookup[key] = ex

    matcher: TrainedHybridMatcher = benchmark["matcher"]
    hybrid_run: HybridRun = benchmark["hybrid"]

    # Feature importance from logistic regression coefficients
    coefs = matcher.model.coef_[0].tolist()
    feature_importance = sorted(
        zip(matcher.feature_names, coefs),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    # Blocking rule stats from all examples
    b_stats = blocking_rule_stats(all_examples)

    # Test labels and scores for interactive threshold
    test_labels = [ex.label for ex in test_examples]
    test_scores = hybrid_run.calibrated_scores

    return DashboardData(
        profiles=profiles,
        profile_lookup=profile_lookup,
        example_lookup=example_lookup,
        extractor=benchmark["extractor"],
        matcher=matcher,
        hybrid_run=hybrid_run,
        metrics=benchmark["metrics"],
        cluster_f1=benchmark["cluster_f1"],
        ablations=benchmark["ablations"],
        confusion_slices=benchmark["confusion_slices"],
        identities=benchmark["identities"],
        resolved_pairs=benchmark["resolved_pairs"],
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        feature_importance=feature_importance,
        blocking_stats=b_stats,
        test_labels=test_labels,
        test_scores=test_scores,
    )
