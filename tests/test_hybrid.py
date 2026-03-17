from __future__ import annotations

from nyne_er_lab.datasets import load_benchmark_profiles
from nyne_er_lab.features import (
    PairFeatureExtractor,
    assign_profile_splits,
    build_pair_examples,
    build_split_candidates,
    profiles_for_split,
)
from nyne_er_lab.eval.benchmark import run_benchmark
from nyne_er_lab.models import run_feature_ablations, run_hybrid_matcher, run_lexical_baseline, run_name_baseline


def _hybrid_inputs():
    profiles = load_benchmark_profiles()
    split_map = assign_profile_splits(profiles)
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
    return extractor, train_examples, val_examples, test_examples


def test_hybrid_training_is_reproducible() -> None:
    extractor, train_examples, val_examples, test_examples = _hybrid_inputs()

    first = run_hybrid_matcher(train_examples, val_examples, test_examples, extractor, random_state=7)
    second = run_hybrid_matcher(train_examples, val_examples, test_examples, extractor, random_state=7)

    assert first.raw_scores == second.raw_scores
    assert first.calibrated_scores == second.calibrated_scores
    assert first.decisions == second.decisions


def test_calibration_improves_brier_score() -> None:
    extractor, train_examples, val_examples, test_examples = _hybrid_inputs()

    run = run_hybrid_matcher(train_examples, val_examples, test_examples, extractor)
    assert run.calibrated_brier <= run.raw_brier


def test_hybrid_beats_baselines_on_pairwise_f1() -> None:
    report = run_benchmark("real_curated_core", seeds=[7, 11, 17])
    metrics = {item["name"]: item for item in report.model_metrics}

    assert metrics["hybrid"]["f1"] >= metrics["embedding_only"]["f1"]
    assert metrics["hybrid"]["f1"] >= 0.85
    assert report.stress_metrics["hard_negative_bank"]["f1"] >= 0.9


def test_abstention_improves_accepted_precision() -> None:
    extractor, train_examples, val_examples, test_examples = _hybrid_inputs()

    run = run_hybrid_matcher(train_examples, val_examples, test_examples, extractor)
    forced_precision = run.calibrated_metrics.precision

    assert run.accepted_precision >= forced_precision
    assert run.abstain_rate > 0.0


def test_feature_ablations_show_value_from_full_model() -> None:
    extractor, train_examples, val_examples, test_examples = _hybrid_inputs()

    ablations = {result.name: result.metrics for result in run_feature_ablations(train_examples, val_examples, test_examples, extractor)}

    assert ablations["full"].f1 >= 0.85
    assert ablations["full"].average_precision >= ablations["no_embedding"].average_precision - 0.05
