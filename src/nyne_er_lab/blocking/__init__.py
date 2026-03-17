"""Blocking module."""

from .blocker import (
    BlockCandidate,
    blocking_recall,
    candidate_volume_ratio,
    domain_overlap_match,
    embedding_neighbor_candidates,
    exact_or_fuzzy_name_match,
    generate_block_candidates,
    gold_positive_pairs,
    org_or_title_overlap_match,
)

__all__ = [
    "BlockCandidate",
    "blocking_recall",
    "candidate_volume_ratio",
    "domain_overlap_match",
    "embedding_neighbor_candidates",
    "exact_or_fuzzy_name_match",
    "generate_block_candidates",
    "gold_positive_pairs",
    "org_or_title_overlap_match",
]
