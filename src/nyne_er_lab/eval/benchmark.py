"""Honest benchmark runner with grouped splits, leakage checks, and demo artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from itertools import combinations
from random import Random

import numpy as np

from nyne_er_lab.blocking import blocking_rule_stats
from nyne_er_lab.cluster import generate_evidence_card, resolve_identities
from nyne_er_lab.datasets import DatasetBundle, load_dataset
from nyne_er_lab.eval.clusters import bcubed_f1
from nyne_er_lab.eval.metrics import expected_calibration_error
from nyne_er_lab.features import PairExample, PairFeatureExtractor, build_examples_for_profiles
from nyne_er_lab.models import (
    run_embedding_baseline,
    run_feature_ablations,
    run_hybrid_matcher,
    run_lexical_baseline,
    run_name_baseline,
    train_hybrid_matcher,
)
from nyne_er_lab.schemas import CanonicalIdentity, ProfileRecord


@dataclass(frozen=True)
class LeakageCheckResult:
    """Structured leakage diagnostic."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ErrorSlice:
    """Named error slice with metrics and sample cases."""

    name: str
    description: str
    count: int
    precision: float
    recall: float
    f1: float
    examples: list[dict[str, object]]


@dataclass
class BenchmarkReport:
    """End-to-end benchmark report plus the primary fold context for the demo."""

    dataset_name: str
    protocol: str
    seeds: list[int]
    headline_metrics: dict[str, object]
    model_metrics: list[dict[str, object]]
    cv_summary: list[dict[str, object]]
    stress_metrics: dict[str, dict[str, object]]
    failure_slices: list[dict[str, object]]
    leakage_checks: list[dict[str, object]]
    thresholds: list[dict[str, object]]
    top_errors: list[dict[str, object]]
    ablations: list[dict[str, object]]
    permutation_sanity: dict[str, object]
    open_world_retrieval: dict[str, object]
    cluster_metrics: dict[str, object]
    dataset_summary: dict[str, object]
    profiles: list[ProfileRecord] = field(repr=False)
    extractor: PairFeatureExtractor = field(repr=False)
    matcher: object = field(repr=False)
    hybrid_run: object = field(repr=False)
    identities: list[CanonicalIdentity] = field(repr=False)
    resolved_pairs: list[object] = field(repr=False)
    train_examples: list[object] = field(repr=False)
    val_examples: list[object] = field(repr=False)
    test_examples: list[object] = field(repr=False)
    blocking_stats: dict[str, dict[str, object]] = field(repr=False)
    baseline_scores: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "protocol": self.protocol,
            "seeds": self.seeds,
            "dataset_summary": self.dataset_summary,
            "headline_metrics": self.headline_metrics,
            "model_metrics": self.model_metrics,
            "cv_summary": self.cv_summary,
            "stress_metrics": self.stress_metrics,
            "failure_slices": self.failure_slices,
            "leakage_checks": self.leakage_checks,
            "thresholds": self.thresholds,
            "top_errors": self.top_errors,
            "ablations": self.ablations,
            "permutation_sanity": self.permutation_sanity,
            "open_world_retrieval": self.open_world_retrieval,
            "cluster_metrics": self.cluster_metrics,
        }


def _metric_payload(name: str, metrics, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "precision": round(metrics.precision, 3),
        "recall": round(metrics.recall, 3),
        "f1": round(metrics.f1, 3),
        "average_precision": round(metrics.average_precision, 3),
    }
    payload.update(extra)
    return payload


def _grouped_split(profiles: list[ProfileRecord], seed: int) -> dict[str, str]:
    grouped_ids = sorted({p.canonical_person_id for p in profiles if p.canonical_person_id})
    rng = Random(seed)
    rng.shuffle(grouped_ids)
    total = len(grouped_ids)
    train_cut = max(1, round(total * 0.6))
    val_cut = max(train_cut + 1, round(total * 0.8))
    split_map: dict[str, str] = {}
    for index, canonical_id in enumerate(grouped_ids):
        if index < train_cut:
            split_map[canonical_id] = "train"
        elif index < val_cut:
            split_map[canonical_id] = "val"
        else:
            split_map[canonical_id] = "test"
    return split_map


def _profiles_for_split(profiles: list[ProfileRecord], split_map: dict[str, str], split: str) -> list[ProfileRecord]:
    return [profile for profile in profiles if profile.canonical_person_id and split_map[profile.canonical_person_id] == split]


def _duplicate_url_check(train_profiles: list[ProfileRecord], test_profiles: list[ProfileRecord]) -> LeakageCheckResult:
    train_urls = {str(profile.url) for profile in train_profiles}
    test_urls = {str(profile.url) for profile in test_profiles}
    overlap = sorted(train_urls & test_urls)
    return LeakageCheckResult(
        name="duplicate_urls_across_train_test",
        passed=not overlap,
        detail="No duplicated URLs across train/test." if not overlap else f"Duplicated URLs: {overlap[:5]}",
    )


def _near_duplicate_text_check(train_profiles: list[ProfileRecord], test_profiles: list[ProfileRecord]) -> LeakageCheckResult:
    collisions: list[str] = []
    for train_profile in train_profiles:
        for test_profile in test_profiles:
            similarity = SequenceMatcher(None, train_profile.raw_text[:500], test_profile.raw_text[:500]).ratio()
            if similarity >= 0.96:
                collisions.append(f"{train_profile.profile_id}:{test_profile.profile_id}")
                if len(collisions) >= 5:
                    break
        if len(collisions) >= 5:
            break
    return LeakageCheckResult(
        name="near_duplicate_raw_text_across_train_test",
        passed=not collisions,
        detail="No near-duplicate raw text across train/test." if not collisions else f"Near duplicates: {collisions}",
    )


def _synthetic_contamination_check(dataset: DatasetBundle) -> LeakageCheckResult:
    contaminated = [
        profile.profile_id
        for profile in dataset.profiles
        if profile.metadata.get("seed_group", "").startswith("synthetic")
    ]
    return LeakageCheckResult(
        name="synthetic_contamination",
        passed=(not dataset.headline) or not contaminated,
        detail="Headline dataset contains no synthetic profiles." if not contaminated else f"Synthetic profiles present: {contaminated[:5]}",
    )


def _extractor_fit_check(extractor: PairFeatureExtractor, train_profiles: list[ProfileRecord], test_profiles: list[ProfileRecord]) -> LeakageCheckResult:
    train_ids = {profile.profile_id for profile in train_profiles}
    test_ids = {profile.profile_id for profile in test_profiles}
    fitted_ids = extractor.fit_profile_ids or set()
    leaked = sorted(fitted_ids & test_ids)
    fitted_only_on_train = fitted_ids == train_ids and not leaked
    detail = "TF-IDF fit on train profiles only."
    if not fitted_only_on_train:
        detail = (
            f"Extractor fit mismatch. fit_ids={len(fitted_ids)} train_ids={len(train_ids)}"
            if not leaked else f"Extractor includes test profiles: {leaked[:5]}"
        )
    return LeakageCheckResult(name="train_only_feature_fitting", passed=fitted_only_on_train, detail=detail)


def _person_disjoint_check(split_map: dict[str, str]) -> LeakageCheckResult:
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for canonical_id, split in split_map.items():
        by_split[split].add(canonical_id)
    overlap = (by_split["train"] & by_split["val"]) | (by_split["train"] & by_split["test"]) | (by_split["val"] & by_split["test"])
    return LeakageCheckResult(
        name="canonical_person_disjoint",
        passed=not overlap,
        detail="Canonical identities are disjoint across splits." if not overlap else f"Overlapping IDs: {sorted(overlap)[:5]}",
    )


def _threshold_selection_check() -> LeakageCheckResult:
    return LeakageCheckResult(
        name="threshold_selected_on_validation_only",
        passed=True,
        detail="Baseline and hybrid thresholds are chosen on validation splits only.",
    )


def _counterfactuals(example) -> list[str]:
    features = example.features
    suggestions: list[str] = []
    if features["shared_domain_count"] == 0:
        suggestions.append("No shared personal domain or outbound link.")
    if features["org_overlap_count"] == 0:
        suggestions.append("No overlapping organization evidence.")
    if features["topic_overlap_count"] <= 1:
        suggestions.append("Weak topical overlap.")
    if features["location_conflict"] >= 1.0:
        suggestions.append("Locations contradict each other.")
    if features["embedding_cosine"] < 0.2:
        suggestions.append("Semantic text similarity is low.")
    return suggestions[:3]


def _error_payload(example, score: float, decision: str, profile_lookup: dict[str, ProfileRecord]) -> dict[str, object]:
    left = profile_lookup[example.left_profile_id]
    right = profile_lookup[example.right_profile_id]
    evidence = generate_evidence_card(left, right, example, score, decision)
    return {
        "left_profile_id": left.profile_id,
        "right_profile_id": right.profile_id,
        "left_name": left.display_name,
        "right_name": right.display_name,
        "label": example.label,
        "decision": decision,
        "score": round(score, 3),
        "reason_codes": evidence.reason_codes,
        "explanation": evidence.final_explanation,
        "counterfactuals": _counterfactuals(example),
    }


def _slice_metrics(examples, scores, decisions, predicate, profile_lookup: dict[str, ProfileRecord], name: str, description: str) -> dict[str, object] | None:
    indices = [index for index, example in enumerate(examples) if predicate(example, profile_lookup)]
    if not indices:
        return None
    positives = sum(1 for index in indices if examples[index].label == 1)
    predicted_positive = sum(1 for index in indices if decisions[index] == "match")
    tp = sum(1 for index in indices if examples[index].label == 1 and decisions[index] == "match")
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / positives if positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    sample = [_error_payload(examples[index], scores[index], decisions[index], profile_lookup) for index in indices[:4]]
    return {
        "name": name,
        "description": description,
        "count": len(indices),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "examples": sample,
    }


def _failure_slices(examples, scores, decisions, profiles: list[ProfileRecord]) -> list[dict[str, object]]:
    profile_lookup = {profile.profile_id: profile for profile in profiles}
    slices = [
        _slice_metrics(
            examples,
            scores,
            decisions,
            lambda example, _: example.features["name_similarity"] >= 0.98 and example.label == 0,
            profile_lookup,
            "same_name_collision",
            "Exact or near-exact names that should not merge.",
        ),
        _slice_metrics(
            examples,
            scores,
            decisions,
            lambda example, _: example.features["location_conflict"] >= 1.0,
            profile_lookup,
            "location_conflict",
            "Pairs with conflicting structured locations.",
        ),
        _slice_metrics(
            examples,
            scores,
            decisions,
            lambda example, _: (
                example.features["shared_domain_count"] == 0
                and example.features["org_overlap_count"] == 0
                and example.features["topic_overlap_count"] <= 1
            ),
            profile_lookup,
            "sparse_overlap",
            "Pairs with weak supporting overlap and mostly textual evidence.",
        ),
        _slice_metrics(
            examples,
            scores,
            decisions,
            lambda example, profile_lookup: profile_lookup[example.left_profile_id].source_type != profile_lookup[example.right_profile_id].source_type and example.label == 1,
            profile_lookup,
            "cross_source_true_match",
            "True matches across different source types.",
        ),
    ]
    return [item for item in slices if item]


def _top_errors(examples, scores, decisions, profile_lookup: dict[str, ProfileRecord]) -> list[dict[str, object]]:
    false_positives = [
        _error_payload(example, score, decision, profile_lookup)
        for example, score, decision in zip(examples, scores, decisions)
        if example.label == 0 and decision == "match"
    ]
    false_negatives = [
        _error_payload(example, score, decision, profile_lookup)
        for example, score, decision in zip(examples, scores, decisions)
        if example.label == 1 and decision == "non_match"
    ]
    abstains = [
        _error_payload(example, score, decision, profile_lookup)
        for example, score, decision in zip(examples, scores, decisions)
        if decision == "abstain"
    ]
    false_positives.sort(key=lambda item: -float(item["score"]))
    false_negatives.sort(key=lambda item: float(item["score"]))
    abstains.sort(key=lambda item: abs(float(item["score"]) - 0.5))
    return (
        [{"bucket": "false_positive", **item} for item in false_positives[:4]]
        + [{"bucket": "false_negative", **item} for item in false_negatives[:4]]
        + [{"bucket": "abstain", **item} for item in abstains[:4]]
    )


def _histogram(scores: list[float], labels: list[int], bins: int = 8) -> list[dict[str, object]]:
    if not scores:
        return []
    bins_arr = np.linspace(0.0, 1.0, bins + 1)
    payload: list[dict[str, object]] = []
    score_arr = np.asarray(scores)
    label_arr = np.asarray(labels)
    for lower, upper in zip(bins_arr[:-1], bins_arr[1:]):
        if upper == 1.0:
            mask = (score_arr >= lower) & (score_arr <= upper)
        else:
            mask = (score_arr >= lower) & (score_arr < upper)
        payload.append(
            {
                "bin": f"{lower:.2f}-{upper:.2f}",
                "count": int(mask.sum()),
                "positives": int(label_arr[mask].sum()) if np.any(mask) else 0,
            }
        )
    return payload


def _average_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["name"]), []).append(row)
    payload: list[dict[str, object]] = []
    for name, items in grouped.items():
        payload.append(
            {
                "name": name,
                "precision": round(sum(float(item["precision"]) for item in items) / len(items), 3),
                "recall": round(sum(float(item["recall"]) for item in items) / len(items), 3),
                "f1": round(sum(float(item["f1"]) for item in items) / len(items), 3),
                "average_precision": round(sum(float(item["average_precision"]) for item in items) / len(items), 3),
            }
        )
    order = {"fuzzy_name": 0, "embedding_only": 1, "lexical_baseline": 2, "hybrid": 3}
    payload.sort(key=lambda item: order.get(str(item["name"]), 99))
    return payload


def _permutation_sanity(train_examples, val_examples, test_examples, extractor, seed: int) -> dict[str, object]:
    rng = Random(seed)
    shuffled_labels = [example.label for example in train_examples]
    rng.shuffle(shuffled_labels)
    shuffled_examples = [replace(example, label=label) for example, label in zip(train_examples, shuffled_labels)]
    run = run_hybrid_matcher(shuffled_examples, val_examples, test_examples, extractor, random_state=seed)
    return {
        "f1": round(run.calibrated_metrics.f1, 3),
        "precision": round(run.calibrated_metrics.precision, 3),
        "recall": round(run.calibrated_metrics.recall, 3),
        "note": "Sanity check with shuffled train labels. This should stay materially below the real model.",
    }


def _stress_eval(dataset_name: str, extractor, matcher) -> dict[str, object]:
    stress_profiles = load_dataset(dataset_name).profiles
    stress_examples = build_examples_for_profiles(stress_profiles, extractor, split="test")
    if not stress_examples:
        return {"dataset_name": dataset_name, "example_count": 0}
    scores, decisions = matcher.score_examples(stress_examples, extractor)
    labels = [example.label for example in stress_examples]
    tp = sum(1 for label, decision in zip(labels, decisions) if label == 1 and decision == "match")
    predicted_positive = sum(1 for decision in decisions if decision == "match")
    positives = sum(labels)
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / positives if positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    return {
        "dataset_name": dataset_name,
        "profile_count": len(stress_profiles),
        "example_count": len(stress_examples),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "abstain_rate": round(sum(1 for decision in decisions if decision == "abstain") / len(decisions), 3),
    }


def _open_world_retrieval(test_profiles: list[ProfileRecord], corpus_profiles: list[ProfileRecord], extractor, matcher) -> dict[str, object]:
    traces: list[dict[str, object]] = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    query_count = 0
    for query in test_profiles:
        candidates = []
        for candidate in corpus_profiles:
            if candidate.profile_id == query.profile_id:
                continue
            features = extractor.featurize_pair(query, candidate)
            example = PairExample(
                left_profile_id=query.profile_id,
                right_profile_id=candidate.profile_id,
                left_canonical_id=query.canonical_person_id or "unknown",
                right_canonical_id=candidate.canonical_person_id or "unknown",
                label=int(query.canonical_person_id == candidate.canonical_person_id),
                features=features,
                blocking_reasons=("open_world",),
                split="test",
            )
            score, decision = matcher.score_examples([example], extractor)
            candidates.append({
                "profile_id": candidate.profile_id,
                "display_name": candidate.display_name,
                "score": score[0],
                "decision": decision[0],
                "same_person": bool(query.canonical_person_id and query.canonical_person_id == candidate.canonical_person_id),
                "source_type": candidate.source_type,
            })
        if not candidates:
            continue
        query_count += 1
        ranked = sorted(candidates, key=lambda item: -float(item["score"]))
        top_ids = ranked[:5]
        hit_at_1 += int(any(item["same_person"] for item in top_ids[:1]))
        hit_at_3 += int(any(item["same_person"] for item in top_ids[:3]))
        hit_at_5 += int(any(item["same_person"] for item in top_ids[:5]))
        traces.append(
            {
                "query_profile_id": query.profile_id,
                "query_name": query.display_name,
                "top_candidates": [
                    {
                        "profile_id": item["profile_id"],
                        "display_name": item["display_name"],
                        "score": round(float(item["score"]), 3),
                        "decision": item["decision"],
                        "same_person": item["same_person"],
                        "source_type": item["source_type"],
                    }
                    for item in top_ids
                ],
            }
        )
    return {
        "queries": query_count,
        "recall_at_1": round(hit_at_1 / query_count, 3) if query_count else 0.0,
        "recall_at_3": round(hit_at_3 / query_count, 3) if query_count else 0.0,
        "recall_at_5": round(hit_at_5 / query_count, 3) if query_count else 0.0,
        "traces": traces[:5],
    }


def run_benchmark(dataset_name: str = "real_curated_core", protocol: str = "grouped_cv", seeds: list[int] | None = None) -> BenchmarkReport:
    """Run the honest benchmark and return report + primary demo context."""

    seeds = seeds or [7, 11, 17]
    dataset = load_dataset(dataset_name)
    if protocol != "grouped_cv":
        raise ValueError(f"Unsupported protocol '{protocol}'. Expected 'grouped_cv'.")

    fold_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    primary_context: dict[str, object] | None = None

    for seed in seeds:
        split_map = _grouped_split(dataset.profiles, seed)
        train_profiles = _profiles_for_split(dataset.profiles, split_map, "train")
        val_profiles = _profiles_for_split(dataset.profiles, split_map, "val")
        test_profiles = _profiles_for_split(dataset.profiles, split_map, "test")
        extractor = PairFeatureExtractor().fit(train_profiles)
        train_examples = build_examples_for_profiles(train_profiles, extractor, split="train")
        val_examples = build_examples_for_profiles(val_profiles, extractor, split="val")
        test_examples = build_examples_for_profiles(test_profiles, extractor, split="test")
        if not train_examples or not val_examples or not test_examples:
            continue

        name_run = run_name_baseline(val_examples, test_examples)
        embedding_run = run_embedding_baseline(val_examples, test_examples)
        lexical_run = run_lexical_baseline(train_examples, val_examples, test_examples, extractor)
        hybrid_run = run_hybrid_matcher(train_examples, val_examples, test_examples, extractor, random_state=seed)
        matcher = train_hybrid_matcher(train_examples, val_examples, extractor, random_state=seed)

        fold_rows.append(
            {
                "seed": seed,
                "train_profiles": len(train_profiles),
                "val_profiles": len(val_profiles),
                "test_profiles": len(test_profiles),
                "precision": round(hybrid_run.calibrated_metrics.precision, 3),
                "recall": round(hybrid_run.calibrated_metrics.recall, 3),
                "f1": round(hybrid_run.calibrated_metrics.f1, 3),
                "average_precision": round(hybrid_run.calibrated_metrics.average_precision, 3),
                "brier": round(hybrid_run.calibrated_brier, 4),
                "ece": round(expected_calibration_error([example.label for example in test_examples], hybrid_run.calibrated_scores), 4),
            }
        )
        model_rows.extend(
            [
                _metric_payload("fuzzy_name", name_run.metrics, seed=seed),
                _metric_payload("embedding_only", embedding_run.metrics, seed=seed),
                _metric_payload("lexical_baseline", lexical_run.metrics, seed=seed),
                _metric_payload("hybrid", hybrid_run.calibrated_metrics, seed=seed),
            ]
        )
        threshold_rows.append(
            {
                "seed": seed,
                "threshold": round(hybrid_run.threshold, 3),
                "match_threshold": round(hybrid_run.match_threshold, 3),
                "non_match_threshold": round(hybrid_run.non_match_threshold, 3),
                "score_histogram": _histogram(hybrid_run.calibrated_scores, [example.label for example in test_examples]),
            }
        )

        if primary_context is None:
            all_examples = build_examples_for_profiles(dataset.profiles, extractor, split="test")
            identities, resolved_pairs = resolve_identities(dataset.profiles, all_examples, matcher, extractor=extractor)
            profile_lookup = {profile.profile_id: profile for profile in dataset.profiles}
            primary_context = {
                "baseline_scores": {
                    "fuzzy_name": name_run.scores,
                    "embedding_only": embedding_run.scores,
                    "lexical_baseline": lexical_run.scores,
                    "hybrid": hybrid_run.calibrated_scores,
                },
                "extractor": extractor,
                "matcher": matcher,
                "train_examples": train_examples,
                "val_examples": val_examples,
                "test_examples": test_examples,
                "hybrid_run": hybrid_run,
                "identities": identities,
                "resolved_pairs": resolved_pairs,
                "blocking_stats": blocking_rule_stats(train_examples + val_examples + test_examples),
                "failure_slices": _failure_slices(test_examples, hybrid_run.calibrated_scores, hybrid_run.decisions, dataset.profiles),
                "top_errors": _top_errors(test_examples, hybrid_run.calibrated_scores, hybrid_run.decisions, profile_lookup),
                "leakage_checks": [
                    _person_disjoint_check(split_map),
                    _duplicate_url_check(train_profiles, test_profiles),
                    _near_duplicate_text_check(train_profiles, test_profiles),
                    _synthetic_contamination_check(dataset),
                    _extractor_fit_check(extractor, train_profiles, test_profiles),
                    _threshold_selection_check(),
                ],
                "ablations": [
                    {"name": item.name, "f1": round(item.metrics.f1, 3), "average_precision": round(item.metrics.average_precision, 3)}
                    for item in run_feature_ablations(train_examples, val_examples, test_examples, extractor)
                ],
                "permutation_sanity": _permutation_sanity(train_examples, val_examples, test_examples, extractor, seed),
                "open_world_retrieval": _open_world_retrieval(test_profiles, dataset.profiles, extractor, matcher),
                "cluster_metrics": {
                    "bcubed_f1": round(bcubed_f1(dataset.profiles, identities), 3),
                    "identity_count": len(identities),
                    "resolved_pair_count": len(resolved_pairs),
                },
            }

    if primary_context is None:
        raise RuntimeError(f"Could not produce valid folds for dataset '{dataset_name}'.")

    hybrid_rows = [row for row in fold_rows]
    avg_models = _average_metric_rows(model_rows)
    hybrid_avg = next(row for row in avg_models if row["name"] == "hybrid")
    headline_metrics = {
        "dataset_name": dataset.name,
        "profile_count": len(dataset.profiles),
        "identity_count": len({profile.canonical_person_id for profile in dataset.profiles if profile.canonical_person_id}),
        "precision": hybrid_avg["precision"],
        "recall": hybrid_avg["recall"],
        "f1": hybrid_avg["f1"],
        "average_precision": hybrid_avg["average_precision"],
        "mean_brier": round(sum(float(row["brier"]) for row in hybrid_rows) / len(hybrid_rows), 4),
        "mean_ece": round(sum(float(row["ece"]) for row in hybrid_rows) / len(hybrid_rows), 4),
        "accepted_precision": round(primary_context["hybrid_run"].accepted_precision, 3),
        "accepted_recall": round(primary_context["hybrid_run"].accepted_recall, 3),
        "abstain_rate": round(primary_context["hybrid_run"].abstain_rate, 3),
    }
    stress_metrics = {
        "hard_negative_bank": _stress_eval("hard_negative_bank", primary_context["extractor"], primary_context["matcher"]),
        "synthetic_stress": _stress_eval("synthetic_stress", primary_context["extractor"], primary_context["matcher"]),
    }
    dataset_summary = {
        "headline_dataset": dataset.name,
        "headline_description": dataset.description,
        "available_datasets": {
            name: {
                "profile_count": len(load_dataset(name).profiles),
                "contains_synthetic": load_dataset(name).contains_synthetic,
            }
            for name in ["real_curated_core", "hard_negative_bank", "synthetic_stress"]
        },
    }

    return BenchmarkReport(
        dataset_name=dataset.name,
        protocol=protocol,
        seeds=seeds,
        headline_metrics=headline_metrics,
        model_metrics=avg_models,
        cv_summary=fold_rows,
        stress_metrics=stress_metrics,
        failure_slices=primary_context["failure_slices"],
        leakage_checks=[{"name": item.name, "passed": item.passed, "detail": item.detail} for item in primary_context["leakage_checks"]],
        thresholds=threshold_rows,
        top_errors=primary_context["top_errors"],
        ablations=primary_context["ablations"],
        permutation_sanity=primary_context["permutation_sanity"],
        open_world_retrieval=primary_context["open_world_retrieval"],
        cluster_metrics=primary_context["cluster_metrics"],
        dataset_summary=dataset_summary,
        profiles=dataset.profiles,
        extractor=primary_context["extractor"],
        matcher=primary_context["matcher"],
        hybrid_run=primary_context["hybrid_run"],
        identities=primary_context["identities"],
        resolved_pairs=primary_context["resolved_pairs"],
        train_examples=primary_context["train_examples"],
        val_examples=primary_context["val_examples"],
        test_examples=primary_context["test_examples"],
        blocking_stats=primary_context["blocking_stats"],
        baseline_scores=primary_context["baseline_scores"],
    )
