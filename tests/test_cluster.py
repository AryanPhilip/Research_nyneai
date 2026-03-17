from __future__ import annotations

from nyne_er_lab.cluster import generate_evidence_card, resolve_identities
from nyne_er_lab.datasets import load_benchmark_profiles
from nyne_er_lab.eval import bcubed_f1
from nyne_er_lab.features import (
    PairFeatureExtractor,
    assign_profile_splits,
    build_examples_for_profiles,
    build_pair_examples,
    build_split_candidates,
    profiles_for_split,
)
from nyne_er_lab.models import train_hybrid_matcher


def _cluster_inputs():
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
    matcher = train_hybrid_matcher(train_examples, val_examples, extractor)
    all_examples = build_examples_for_profiles(profiles, extractor)
    return profiles, extractor, matcher, all_examples


def test_evidence_card_contains_specific_supporting_or_contradicting_signals() -> None:
    profiles, _, _, all_examples = _cluster_inputs()
    lookup = {profile.profile_id: profile for profile in profiles}
    positive_example = next(example for example in all_examples if example.left_profile_id == "andrej_github" and example.right_profile_id == "andrej_personal")
    card = generate_evidence_card(
        lookup[positive_example.left_profile_id],
        lookup[positive_example.right_profile_id],
        positive_example,
        score=0.98,
        decision="match",
    )

    assert card.supporting_signals
    assert "high_name_similarity" in card.reason_codes
    assert "Decision=match" in card.final_explanation


def test_identity_resolution_separates_exact_name_hard_negative() -> None:
    profiles, extractor, matcher, all_examples = _cluster_inputs()
    identities, _ = resolve_identities(profiles, all_examples, matcher, extractor=extractor)

    identity_by_profile = {}
    for identity in identities:
        for profile_id in identity.member_profile_ids:
            identity_by_profile[profile_id] = identity.entity_id

    assert identity_by_profile["sebastian_github"] == identity_by_profile["sebastian_personal"]
    assert identity_by_profile["sebastian_github"] != identity_by_profile["sebastian_raschka_finance"]


def test_cluster_metric_is_high_on_seed_benchmark() -> None:
    profiles, extractor, matcher, all_examples = _cluster_inputs()
    identities, _ = resolve_identities(profiles, all_examples, matcher, extractor=extractor)
    score = bcubed_f1(profiles, identities)

    assert score >= 0.8


def test_clustered_identities_include_confidence_bands() -> None:
    profiles, extractor, matcher, all_examples = _cluster_inputs()
    identities, resolved_pairs = resolve_identities(profiles, all_examples, matcher, extractor=extractor)

    assert identities
    assert resolved_pairs
    assert all(identity.confidence_band in {"high", "medium", "low", "uncertain"} for identity in identities)
