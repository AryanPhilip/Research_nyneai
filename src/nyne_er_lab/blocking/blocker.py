"""Candidate generation and blocking for entity resolution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from nyne_er_lab.ingest import compose_normalized_text
from nyne_er_lab.schemas import ProfileRecord


@dataclass(frozen=True)
class BlockCandidate:
    """Candidate profile pair with explainable blocking reasons."""

    left_profile_id: str
    right_profile_id: str
    reasons: tuple[str, ...]


def _pair_key(left_profile_id: str, right_profile_id: str) -> tuple[str, str]:
    return tuple(sorted((left_profile_id, right_profile_id)))


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _name_tokens(profile: ProfileRecord) -> set[str]:
    values = [profile.display_name, *profile.aliases]
    tokens: set[str] = set()
    for value in values:
        tokens.update(token for token in _normalize_text(value).split() if len(token) > 1)
    return tokens


def _name_variants(profile: ProfileRecord) -> set[str]:
    base_name = _normalize_text(profile.display_name)
    parts = base_name.split()
    variants = {base_name, *(_normalize_text(alias) for alias in profile.aliases)}
    if len(parts) >= 2:
        variants.add(f"{parts[0]} {parts[-1][0]}")
        variants.add(f"{parts[0][0]} {parts[-1]}")
    return {variant.strip() for variant in variants if variant.strip()}


def _domains(profile: ProfileRecord) -> set[str]:
    urls = [*(str(link) for link in profile.outbound_links)]
    if profile.source_type in {"personal_site", "huggingface"}:
        urls.append(str(profile.url))
    return {
        urlparse(url).netloc.replace("www.", "")
        for url in urls
        if urlparse(url).netloc
    }


def _org_tokens(profile: ProfileRecord) -> set[str]:
    return {
        token
        for org in profile.organizations
        for token in _normalize_text(org.name).split()
        if len(token) > 2
    }


def _topic_tokens(profile: ProfileRecord) -> set[str]:
    parts = [profile.headline or "", *profile.topics]
    return {
        token
        for part in parts
        for token in _normalize_text(part).split()
        if len(token) > 2
    }


def _same_person_label(left: ProfileRecord, right: ProfileRecord) -> bool:
    return bool(left.canonical_person_id and left.canonical_person_id == right.canonical_person_id)


def exact_or_fuzzy_name_match(left: ProfileRecord, right: ProfileRecord) -> bool:
    left_variants = _name_variants(left)
    right_variants = _name_variants(right)
    if left_variants & right_variants:
        return True

    similarity = SequenceMatcher(None, _normalize_text(left.display_name), _normalize_text(right.display_name)).ratio()
    shared_tokens = _name_tokens(left) & _name_tokens(right)
    return similarity >= 0.85 or len(shared_tokens) >= 2


def alias_or_initial_match(left: ProfileRecord, right: ProfileRecord) -> bool:
    left_variants = _name_variants(left)
    right_variants = _name_variants(right)
    return bool(left_variants & right_variants)


def domain_overlap_match(left: ProfileRecord, right: ProfileRecord) -> bool:
    return bool(_domains(left) & _domains(right))


def org_or_title_overlap_match(left: ProfileRecord, right: ProfileRecord) -> bool:
    org_overlap = _org_tokens(left) & _org_tokens(right)
    topic_overlap = _topic_tokens(left) & _topic_tokens(right)
    return bool(org_overlap) or len(topic_overlap) >= 2


def rule_reasons(left: ProfileRecord, right: ProfileRecord) -> set[str]:
    reasons: set[str] = set()
    if exact_or_fuzzy_name_match(left, right):
        reasons.add("fuzzy_name")
    if alias_or_initial_match(left, right):
        reasons.add("alias_or_initial")
    if domain_overlap_match(left, right):
        reasons.add("shared_domain")
    if org_or_title_overlap_match(left, right):
        reasons.add("org_or_topic_overlap")
    return reasons


def embedding_neighbor_candidates(profiles: list[ProfileRecord], top_k: int = 3) -> dict[tuple[str, str], set[str]]:
    """Find semantically similar profiles using TF-IDF nearest neighbors."""

    if len(profiles) < 2:
        return {}

    normalized_texts = [compose_normalized_text(profile) for profile in profiles]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(normalized_texts)
    neighbor_count = min(top_k + 1, len(profiles))
    model = NearestNeighbors(metric="cosine", n_neighbors=neighbor_count)
    model.fit(matrix)
    _, indices = model.kneighbors(matrix)

    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row_index, neighbors in enumerate(indices):
        for neighbor_index in neighbors[1:]:
            key = _pair_key(profiles[row_index].profile_id, profiles[neighbor_index].profile_id)
            candidates[key].add("embedding_neighbor")
    return candidates


def generate_block_candidates(profiles: list[ProfileRecord], top_k: int = 3) -> list[BlockCandidate]:
    """Combine cheap blocking rules with embedding-neighbor fallback."""

    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    profile_lookup = {profile.profile_id: profile for profile in profiles}

    for left, right in combinations(profiles, 2):
        key = _pair_key(left.profile_id, right.profile_id)
        candidates[key].update(rule_reasons(left, right))

    for key, reasons in embedding_neighbor_candidates(profiles, top_k=top_k).items():
        candidates[key].update(reasons)

    return [
        BlockCandidate(left_profile_id=left_id, right_profile_id=right_id, reasons=tuple(sorted(reasons)))
        for (left_id, right_id), reasons in sorted(candidates.items())
        if reasons and left_id in profile_lookup and right_id in profile_lookup
    ]


def blocking_rule_stats(examples) -> dict[str, dict]:
    """Compute per-rule candidate count, true positives, and precision.

    Parameters
    ----------
    examples : list[PairExample]
        Pair examples with ``blocking_reasons`` and ``label`` attributes.

    Returns
    -------
    dict mapping rule name to {count, true_positives, precision}.
    """

    from collections import defaultdict

    rule_counts: dict[str, int] = defaultdict(int)
    rule_tp: dict[str, int] = defaultdict(int)
    for example in examples:
        for reason in example.blocking_reasons:
            rule_counts[reason] += 1
            if example.label == 1:
                rule_tp[reason] += 1
    return {
        rule: {
            "count": rule_counts[rule],
            "true_positives": rule_tp[rule],
            "precision": rule_tp[rule] / max(rule_counts[rule], 1),
        }
        for rule in sorted(rule_counts)
    }


def gold_positive_pairs(profiles: list[ProfileRecord]) -> set[tuple[str, str]]:
    """Return all pairwise gold-positive profile pairs from canonical ids."""

    positives: set[tuple[str, str]] = set()
    for left, right in combinations(profiles, 2):
        if _same_person_label(left, right):
            positives.add(_pair_key(left.profile_id, right.profile_id))
    return positives


def blocking_recall(profiles: list[ProfileRecord], candidates: list[BlockCandidate]) -> float:
    """Measure recall of gold-positive pairs captured by the blocker."""

    positives = gold_positive_pairs(profiles)
    if not positives:
        return 0.0

    candidate_keys = {
        _pair_key(candidate.left_profile_id, candidate.right_profile_id)
        for candidate in candidates
    }
    captured = positives & candidate_keys
    return len(captured) / len(positives)


def candidate_volume_ratio(profiles: list[ProfileRecord], candidates: list[BlockCandidate]) -> float:
    """Return ratio of generated candidates vs total possible profile pairs."""

    total_possible = max((len(profiles) * (len(profiles) - 1)) // 2, 1)
    return len(candidates) / total_possible
