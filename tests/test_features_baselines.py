from __future__ import annotations

from nyne_er_lab.datasets import load_benchmark_profiles
from nyne_er_lab.eval import assert_person_disjoint, summarize_split_assignments
from nyne_er_lab.features import (
    PairFeatureExtractor,
    assign_profile_splits,
    build_pair_examples,
    build_split_candidates,
    profiles_for_split,
)
from nyne_er_lab.models import run_embedding_baseline, run_lexical_baseline, run_name_baseline


def _benchmark_inputs():
    profiles = load_benchmark_profiles()
    split_map = assign_profile_splits(profiles)
    assert_person_disjoint(profiles, split_map)

    extractor = PairFeatureExtractor().fit(profiles_for_split(profiles, split_map, "train"))

    train_examples = build_pair_examples(
        profiles,
        build_split_candidates(profiles, split_map, "train"),
        extractor,
        split_map,
        "train",
    )
    val_examples = build_pair_examples(
        profiles,
        build_split_candidates(profiles, split_map, "val"),
        extractor,
        split_map,
        "val",
    )
    test_examples = build_pair_examples(
        profiles,
        build_split_candidates(profiles, split_map, "test"),
        extractor,
        split_map,
        "test",
    )
    return profiles, split_map, extractor, train_examples, val_examples, test_examples


def test_split_assignments_are_person_disjoint() -> None:
    profiles, split_map, _, train_examples, val_examples, test_examples = _benchmark_inputs()

    counts = summarize_split_assignments(split_map)
    assert counts["train"] >= 4
    assert counts["val"] >= 1
    assert counts["test"] >= 1

    train_ids = {profile.canonical_person_id for profile in profiles if split_map[profile.canonical_person_id] == "train"}
    val_ids = {profile.canonical_person_id for profile in profiles if split_map[profile.canonical_person_id] == "val"}
    test_ids = {profile.canonical_person_id for profile in profiles if split_map[profile.canonical_person_id] == "test"}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_examples and val_examples and test_examples


def test_feature_extractor_returns_interpretable_feature_set() -> None:
    _, _, extractor, train_examples, _, _ = _benchmark_inputs()
    feature_row = train_examples[0].features

    assert "name_similarity" in feature_row
    assert "org_overlap_count" in feature_row
    assert "embedding_cosine" in feature_row
    matrix = extractor.vectorize_features([feature_row])
    assert matrix.shape[1] >= 10


def test_all_baselines_run_and_return_metrics() -> None:
    _, _, extractor, train_examples, val_examples, test_examples = _benchmark_inputs()

    name_run = run_name_baseline(val_examples, test_examples)
    embedding_run = run_embedding_baseline(val_examples, test_examples)
    lexical_run = run_lexical_baseline(train_examples, val_examples, test_examples, extractor)

    assert 0.0 <= name_run.metrics.f1 <= 1.0
    assert 0.0 <= embedding_run.metrics.average_precision <= 1.0
    assert 0.0 <= lexical_run.metrics.precision <= 1.0
    assert len(lexical_run.scores) == len(test_examples)


def test_lexical_baseline_remains_competitive_with_name_baseline() -> None:
    _, _, extractor, train_examples, val_examples, test_examples = _benchmark_inputs()

    name_run = run_name_baseline(val_examples, test_examples)
    lexical_run = run_lexical_baseline(train_examples, val_examples, test_examples, extractor)

    assert lexical_run.metrics.f1 >= 0.65
    assert lexical_run.metrics.average_precision >= 0.7
    assert lexical_run.metrics.f1 >= name_run.metrics.f1 - 0.15
