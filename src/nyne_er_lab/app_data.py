"""Data loading and caching layer for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import streamlit as st

from nyne_er_lab.blocking.blocker import blocking_rule_stats
from nyne_er_lab.cluster.resolver import ResolvedPair
from nyne_er_lab.eval.benchmark import BenchmarkReport, run_benchmark
from nyne_er_lab.features.dataset import PairExample
from nyne_er_lab.features.extractor import PairFeatureExtractor
from nyne_er_lab.models.hybrid import HybridRun, TrainedHybridMatcher
from nyne_er_lab.schemas import CanonicalIdentity, ProfileRecord


@dataclass
class NetworkGraph:
    """Pre-computed graph layout for identity visualization."""

    node_ids: list[str]
    node_names: list[str]
    node_types: list[str]  # source_type per profile
    node_cluster: list[str]  # canonical_name per profile
    node_x: list[float]
    node_y: list[float]
    edge_left: list[str]
    edge_right: list[str]
    edge_decision: list[str]
    edge_score: list[float]


@dataclass
class DashboardData:
    report: BenchmarkReport
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
    headline_metrics: dict
    stress_metrics: dict
    failure_gallery: list[dict]
    leakage_checks: list[dict]
    thresholds: list[dict]
    search_trace: dict
    dataset_summary: dict
    network: NetworkGraph = field(default_factory=lambda: NetworkGraph([], [], [], [], [], [], [], [], [], []))
    score_by_decision: dict[str, list[float]] = field(default_factory=dict)
    score_by_label: dict[int, list[float]] = field(default_factory=dict)
    baseline_scores: dict[str, list[float]] = field(default_factory=dict)


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def _build_network(
    profiles: list[ProfileRecord],
    resolved_pairs: list[ResolvedPair],
    identities: list[CanonicalIdentity],
) -> NetworkGraph:
    """Build a force-directed layout from resolved pairs."""
    G = nx.Graph()

    # Build identity membership lookup
    profile_to_identity = {}
    for identity in identities:
        for pid in identity.member_profile_ids:
            profile_to_identity[pid] = identity.canonical_name

    # Add nodes
    profile_lookup = {p.profile_id: p for p in profiles}
    for p in profiles:
        G.add_node(p.profile_id)

    # Add edges (matches and abstains get edges, non-matches are lighter)
    for rp in resolved_pairs:
        if rp.decision == "match":
            G.add_edge(rp.left_profile_id, rp.right_profile_id, weight=rp.score * 3)
        elif rp.decision == "abstain":
            G.add_edge(rp.left_profile_id, rp.right_profile_id, weight=rp.score * 0.5)

    # Spring layout
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    node_ids = list(G.nodes())
    node_names = [profile_lookup[nid].display_name if nid in profile_lookup else nid for nid in node_ids]
    node_types = [profile_lookup[nid].source_type if nid in profile_lookup else "unknown" for nid in node_ids]
    node_cluster = [profile_to_identity.get(nid, "singleton") for nid in node_ids]
    node_x = [float(pos[nid][0]) for nid in node_ids]
    node_y = [float(pos[nid][1]) for nid in node_ids]

    edge_left, edge_right, edge_decision, edge_score = [], [], [], []
    for rp in resolved_pairs:
        if rp.left_profile_id in pos and rp.right_profile_id in pos:
            edge_left.append(rp.left_profile_id)
            edge_right.append(rp.right_profile_id)
            edge_decision.append(rp.decision)
            edge_score.append(rp.score)

    return NetworkGraph(
        node_ids=node_ids,
        node_names=node_names,
        node_types=node_types,
        node_cluster=node_cluster,
        node_x=node_x,
        node_y=node_y,
        edge_left=edge_left,
        edge_right=edge_right,
        edge_decision=edge_decision,
        edge_score=edge_score,
    )


@st.cache_resource(show_spinner="Running entity resolution pipeline...")
def load_dashboard_data() -> DashboardData:
    report = run_benchmark("real_curated_core", protocol="grouped_cv", seeds=[7, 11, 17])

    profiles: list[ProfileRecord] = report.profiles
    profile_lookup = {p.profile_id: p for p in profiles}

    train_examples: list[PairExample] = report.train_examples
    val_examples: list[PairExample] = report.val_examples
    test_examples: list[PairExample] = report.test_examples

    all_examples = train_examples + val_examples + test_examples
    example_lookup: dict[tuple[str, str], PairExample] = {}
    for ex in all_examples:
        key = _pair_key(ex.left_profile_id, ex.right_profile_id)
        example_lookup[key] = ex

    matcher: TrainedHybridMatcher = report.matcher
    hybrid_run: HybridRun = report.hybrid_run

    hybrid_metrics = next(metric for metric in report.model_metrics if metric["name"] == "hybrid")

    coefs = matcher.model.coef_[0].tolist()
    feature_importance = sorted(
        zip(matcher.feature_names, coefs),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    b_stats = report.blocking_stats or blocking_rule_stats(all_examples)

    test_labels = [ex.label for ex in test_examples]
    test_scores = report.matcher.score_examples(test_examples, report.extractor)[0]

    identities: list[CanonicalIdentity] = report.identities
    resolved_pairs: list[ResolvedPair] = report.resolved_pairs

    # Network graph
    network = _build_network(profiles, resolved_pairs, identities)

    # Score distributions
    score_by_decision: dict[str, list[float]] = {"match": [], "non_match": [], "abstain": []}
    score_by_label: dict[int, list[float]] = {0: [], 1: []}
    for rp in resolved_pairs:
        if rp.decision in score_by_decision:
            score_by_decision[rp.decision].append(rp.score)
    for ex, score in zip(test_examples, test_scores):
        score_by_label[ex.label].append(score)

    return DashboardData(
        report=report,
        profiles=profiles,
        profile_lookup=profile_lookup,
        example_lookup=example_lookup,
        extractor=report.extractor,
        matcher=matcher,
        hybrid_run=hybrid_run,
        metrics=report.model_metrics,
        cluster_f1=float(report.cluster_metrics["bcubed_f1"]),
        ablations=report.ablations,
        confusion_slices={
            "test_examples": len(test_examples),
            "forced_precision": float(hybrid_metrics["precision"]),
            "accepted_precision": float(report.headline_metrics["accepted_precision"]),
            "accepted_recall": float(report.headline_metrics["accepted_recall"]),
            "abstain_rate": float(report.headline_metrics["abstain_rate"]),
        },
        identities=identities,
        resolved_pairs=resolved_pairs,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        feature_importance=feature_importance,
        blocking_stats=b_stats,
        test_labels=test_labels,
        test_scores=test_scores,
        headline_metrics=report.headline_metrics,
        stress_metrics=report.stress_metrics,
        failure_gallery=report.top_errors,
        leakage_checks=report.leakage_checks,
        thresholds=report.thresholds,
        search_trace=report.open_world_retrieval,
        dataset_summary=report.dataset_summary,
        network=network,
        score_by_decision=score_by_decision,
        score_by_label=score_by_label,
        baseline_scores=report.baseline_scores,
    )
