"""Split validation helpers."""

from __future__ import annotations

from collections import Counter

from nyne_er_lab.features.dataset import PairExample, SplitName
from nyne_er_lab.schemas import ProfileRecord


def summarize_split_assignments(split_map: dict[str, SplitName]) -> dict[str, int]:
    counts = Counter(split_map.values())
    return dict(counts)


def assert_person_disjoint(profiles: list[ProfileRecord], split_map: dict[str, SplitName]) -> None:
    """Raise if a canonical identity appears in more than one split."""

    seen: dict[str, SplitName] = {}
    for profile in profiles:
        canonical_id = profile.canonical_person_id
        if not canonical_id:
            continue
        split = split_map[canonical_id]
        if canonical_id in seen and seen[canonical_id] != split:
            raise AssertionError(f"Canonical id {canonical_id} appears in multiple splits")
        seen[canonical_id] = split


def examples_by_split(examples: list[PairExample]) -> dict[str, int]:
    counts = Counter(example.split for example in examples)
    return dict(counts)
