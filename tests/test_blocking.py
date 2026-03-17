from __future__ import annotations

from nyne_er_lab.blocking import (
    blocking_recall,
    candidate_volume_ratio,
    domain_overlap_match,
    embedding_neighbor_candidates,
    exact_or_fuzzy_name_match,
    generate_block_candidates,
    gold_positive_pairs,
    org_or_title_overlap_match,
)
from nyne_er_lab.datasets import load_raw_pages
from nyne_er_lab.ingest import parse_raw_pages


def _profiles():
    return parse_raw_pages(load_raw_pages())


def test_exact_or_fuzzy_name_matching() -> None:
    profiles = _profiles()
    assert exact_or_fuzzy_name_match(profiles[0], profiles[1]) is True
    assert exact_or_fuzzy_name_match(profiles[3], profiles[9]) is False


def test_domain_overlap_matching() -> None:
    profiles = {profile.profile_id: profile for profile in _profiles()}
    assert domain_overlap_match(profiles["andrej_personal"], profiles["andrej_speaker"]) is True
    assert domain_overlap_match(profiles["chip_personal"], profiles["jason_builder"]) is False


def test_org_or_topic_overlap_matching() -> None:
    profiles = {profile.profile_id: profile for profile in _profiles()}
    assert org_or_title_overlap_match(profiles["chip_personal"], profiles["chip_github"]) is True
    assert org_or_title_overlap_match(profiles["chip_huynh_blog"], profiles["jason_builder"]) is False


def test_embedding_neighbors_return_semantic_candidates() -> None:
    profiles = _profiles()
    candidates = embedding_neighbor_candidates(profiles, top_k=2)
    keys = set(candidates)
    assert ("jay_github", "jay_personal") in keys
    assert ("andrej_github", "andrej_personal") in keys


def test_blocker_reaches_high_positive_pair_recall() -> None:
    profiles = _profiles()
    candidates = generate_block_candidates(profiles, top_k=3)
    recall = blocking_recall(profiles, candidates)
    volume_ratio = candidate_volume_ratio(profiles, candidates)

    assert recall >= 0.95
    assert volume_ratio < 0.75


def test_gold_positive_pairs_cover_three_seed_entities() -> None:
    profiles = _profiles()
    positives = gold_positive_pairs(profiles)
    assert len(positives) == 9


def test_blocker_emits_traceable_reasons() -> None:
    profiles = _profiles()
    candidates = generate_block_candidates(profiles, top_k=3)
    reasons = {
        (candidate.left_profile_id, candidate.right_profile_id): set(candidate.reasons)
        for candidate in candidates
    }

    assert "shared_domain" in reasons[("andrej_github", "andrej_personal")]
    assert "org_or_topic_overlap" in reasons[("chip_github", "chip_personal")]
