"""Cluster-level evaluation helpers."""

from __future__ import annotations

from collections import defaultdict

from nyne_er_lab.schemas import CanonicalIdentity, ProfileRecord


def bcubed_f1(profiles: list[ProfileRecord], identities: list[CanonicalIdentity]) -> float:
    """Compute B-cubed F1 for predicted identity clusters."""

    predicted_by_profile: dict[str, set[str]] = {}
    for identity in identities:
        members = set(identity.member_profile_ids)
        for profile_id in members:
            predicted_by_profile[profile_id] = members

    gold_by_profile: dict[str, set[str]] = defaultdict(set)
    for profile in profiles:
        gold_by_profile[profile.canonical_person_id].add(profile.profile_id)

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    for profile in profiles:
        predicted = predicted_by_profile.get(profile.profile_id, {profile.profile_id})
        gold = gold_by_profile[profile.canonical_person_id]
        overlap = len(predicted & gold)
        precision_scores.append(overlap / len(predicted))
        recall_scores.append(overlap / len(gold))

    precision = sum(precision_scores) / len(precision_scores)
    recall = sum(recall_scores) / len(recall_scores)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
