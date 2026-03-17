"""Tests for the honest benchmark and tiered dataset contract."""

from __future__ import annotations

from nyne_er_lab.datasets import available_datasets, load_benchmark_profiles, load_dataset
from nyne_er_lab.eval.benchmark import run_benchmark
from nyne_er_lab.features.extractor import PairFeatureExtractor


def test_available_datasets_match_new_registry() -> None:
    assert available_datasets() == ["real_curated_core", "hard_negative_bank", "synthetic_stress"]


def test_real_curated_core_is_headline_dataset() -> None:
    bundle = load_dataset("real_curated_core")
    assert bundle.headline is True
    assert bundle.contains_synthetic is False
    assert len(bundle.profiles) == len(load_benchmark_profiles("real_curated_core"))


def test_stress_datasets_are_larger_than_headline() -> None:
    headline = load_dataset("real_curated_core")
    hard_negative = load_dataset("hard_negative_bank")
    synthetic = load_dataset("synthetic_stress")

    assert len(hard_negative.profiles) > len(headline.profiles)
    assert len(synthetic.profiles) > len(headline.profiles)
    assert synthetic.contains_synthetic is True


def test_feature_extractor_tracks_train_only_fit_ids() -> None:
    train_profiles = load_dataset("real_curated_core").profiles[:8]
    extractor = PairFeatureExtractor().fit(train_profiles)

    assert extractor.fit_profile_ids == {profile.profile_id for profile in train_profiles}


def test_run_benchmark_returns_honest_report_shape() -> None:
    report = run_benchmark("real_curated_core", seeds=[7, 11])

    assert report.dataset_name == "real_curated_core"
    assert report.protocol == "grouped_cv"
    assert report.headline_metrics["dataset_name"] == "real_curated_core"
    assert len(report.model_metrics) == 4
    assert len(report.cv_summary) == 2
    assert len(report.leakage_checks) >= 5
    assert len(report.failure_slices) >= 1
    assert report.stress_metrics["hard_negative_bank"]["profile_count"] > report.headline_metrics["profile_count"]


def test_run_benchmark_has_no_leakage_failures_on_headline_dataset() -> None:
    report = run_benchmark("real_curated_core", seeds=[7])
    assert all(item["passed"] for item in report.leakage_checks), [item["detail"] for item in report.leakage_checks if not item["passed"]]


def test_run_benchmark_separates_headline_and_stress_metrics() -> None:
    report = run_benchmark("real_curated_core", seeds=[7])

    assert "hard_negative_bank" in report.stress_metrics
    assert "synthetic_stress" in report.stress_metrics
    assert report.dataset_summary["headline_dataset"] == "real_curated_core"
    assert report.open_world_retrieval["queries"] >= 1
