"""Cluster assembly and evidence generation for resolved identities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from nyne_er_lab.features.dataset import PairExample
from nyne_er_lab.models.hybrid import TrainedHybridMatcher
from nyne_er_lab.schemas import CanonicalIdentity, EvidenceCard, EvidenceSignal, ProfileRecord, TextSpan


@dataclass(frozen=True)
class ResolvedPair:
    """Scored pair with final decision and explanation."""

    left_profile_id: str
    right_profile_id: str
    score: float
    decision: str
    evidence_card: EvidenceCard


def _pair_key(left_profile_id: str, right_profile_id: str) -> tuple[str, str]:
    return tuple(sorted((left_profile_id, right_profile_id)))


def _select_span(profile: ProfileRecord, field_name: str) -> TextSpan | None:
    for span in profile.supporting_spans:
        if span.field_name == field_name:
            return span
    return profile.supporting_spans[0] if profile.supporting_spans else None


def generate_evidence_card(
    left: ProfileRecord,
    right: ProfileRecord,
    example: PairExample,
    score: float,
    decision: str,
) -> EvidenceCard:
    """Build an interpretable evidence card from pairwise features."""

    supporting_signals: list[EvidenceSignal] = []
    contradicting_signals: list[EvidenceSignal] = []
    reason_codes: list[str] = []
    contradiction_codes: list[str] = []
    features = example.features

    if features["name_similarity"] >= 0.95:
        reason_codes.append("high_name_similarity")
        supporting_signals.append(
            EvidenceSignal(
                signal_type="high_name_similarity",
                description=f"Both profiles present the name {left.display_name}.",
                weight=min(features["name_similarity"], 1.0),
                source_profile_id=left.profile_id,
                target_profile_id=right.profile_id,
                supporting_span=_select_span(left, "headline"),
            )
        )
    if features["shared_domain_count"] > 0:
        reason_codes.append("shared_domain")
        supporting_signals.append(
            EvidenceSignal(
                signal_type="shared_domain",
                description="Profiles link to the same personal domain.",
                weight=0.8,
                source_profile_id=left.profile_id,
                target_profile_id=right.profile_id,
                supporting_span=_select_span(left, "bio_text"),
            )
        )
    if features["org_overlap_count"] > 0:
        reason_codes.append("shared_org")
        supporting_signals.append(
            EvidenceSignal(
                signal_type="shared_org",
                description="Profiles mention overlapping organizations.",
                weight=min(1.0, 0.5 + (0.1 * features["org_overlap_count"])),
                source_profile_id=left.profile_id,
                target_profile_id=right.profile_id,
                supporting_span=_select_span(left, "bio_text"),
            )
        )
    if features["topic_overlap_count"] > 0:
        reason_codes.append("shared_topics")
        supporting_signals.append(
            EvidenceSignal(
                signal_type="shared_topics",
                description="Profiles share technical topics or headlines.",
                weight=min(1.0, 0.4 + (0.1 * features["topic_overlap_count"])),
                source_profile_id=left.profile_id,
                target_profile_id=right.profile_id,
                supporting_span=_select_span(right, "bio_text"),
            )
        )
    if features["location_conflict"] >= 1.0:
        contradiction_codes.append("location_conflict")
        contradicting_signals.append(
            EvidenceSignal(
                signal_type="location_conflict",
                description="Structured locations conflict with one another.",
                weight=0.9,
                source_profile_id=left.profile_id,
                target_profile_id=right.profile_id,
                supporting_span=_select_span(left, "bio_text"),
            )
        )

    visible_reason_codes = reason_codes + contradiction_codes if decision != "match" else reason_codes
    explanation_parts = [
        f"Decision={decision} at confidence {score:.2f}.",
        f"Supporting signals: {', '.join(reason_codes) if reason_codes else 'none'}.",
    ]
    if contradiction_codes:
        explanation_parts.append(f"Contradictions: {', '.join(contradiction_codes)}.")
    explanation = (
        " ".join(explanation_parts)
    )
    return EvidenceCard(
        left_profile_id=left.profile_id,
        right_profile_id=right.profile_id,
        supporting_signals=supporting_signals,
        contradicting_signals=contradicting_signals,
        reason_codes=visible_reason_codes,
        final_explanation=explanation,
    )


def _confidence_band(edge_scores: list[float]) -> str:
    if not edge_scores:
        return "uncertain"
    average_score = sum(edge_scores) / len(edge_scores)
    if average_score >= 0.9:
        return "high"
    if average_score >= 0.75:
        return "medium"
    if average_score >= 0.6:
        return "low"
    return "uncertain"


def resolve_identities(
    profiles: list[ProfileRecord],
    examples: list[PairExample],
    matcher: TrainedHybridMatcher,
    *,
    extractor,
) -> tuple[list[CanonicalIdentity], list[ResolvedPair]]:
    """Resolve canonical identities from scored pairwise examples."""

    profile_lookup = {profile.profile_id: profile for profile in profiles}
    scores, decisions = matcher.score_examples(examples, extractor)
    resolved_pairs: list[ResolvedPair] = []

    parent = {profile.profile_id: profile.profile_id for profile in profiles}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for example, score, decision in zip(examples, scores, decisions):
        left = profile_lookup[example.left_profile_id]
        right = profile_lookup[example.right_profile_id]
        evidence_card = generate_evidence_card(left, right, example, score, decision)
        resolved_pairs.append(
            ResolvedPair(
                left_profile_id=left.profile_id,
                right_profile_id=right.profile_id,
                score=score,
                decision=decision,
                evidence_card=evidence_card,
            )
        )
        if decision == "match":
            union(left.profile_id, right.profile_id)

    clusters: dict[str, list[ProfileRecord]] = defaultdict(list)
    edge_scores_by_cluster: dict[str, list[float]] = defaultdict(list)
    for profile in profiles:
        clusters[find(profile.profile_id)].append(profile)
    for resolved_pair in resolved_pairs:
        if resolved_pair.decision != "match":
            continue
        root = find(resolved_pair.left_profile_id)
        edge_scores_by_cluster[root].append(resolved_pair.score)

    identities: list[CanonicalIdentity] = []
    for cluster_profiles in clusters.values():
        member_ids = sorted(profile.profile_id for profile in cluster_profiles)
        display_name = Counter(profile.display_name for profile in cluster_profiles).most_common(1)[0][0]
        orgs = Counter(org.name for profile in cluster_profiles for org in profile.organizations)
        links = []
        seen_links = set()
        for profile in cluster_profiles:
            for link in profile.outbound_links:
                link_str = str(link)
                if link_str not in seen_links:
                    links.append(link)
                    seen_links.add(link_str)
        summary_topics = Counter(topic for profile in cluster_profiles for topic in profile.topics)
        summary = (
            f"{display_name} appears across {len(cluster_profiles)} profiles with topics "
            f"{', '.join(topic for topic, _ in summary_topics.most_common(3)) or 'not enough evidence'}."
        )
        root = find(cluster_profiles[0].profile_id)
        identities.append(
            CanonicalIdentity(
                entity_id=root,
                member_profile_ids=member_ids,
                canonical_name=display_name,
                summary=summary,
                confidence_band=_confidence_band(edge_scores_by_cluster[root]),
                key_orgs=[name for name, _ in orgs.most_common(3)],
                key_links=links[:3],
            )
        )

    identities.sort(key=lambda identity: (-len(identity.member_profile_ids), identity.entity_id))
    return identities, resolved_pairs
