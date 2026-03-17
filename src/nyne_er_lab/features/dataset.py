"""Dataset assembly for pairwise matching benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nyne_er_lab.blocking import BlockCandidate, generate_block_candidates
from nyne_er_lab.features.extractor import PairFeatureExtractor
from nyne_er_lab.schemas import ProfileRecord


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class PairExample:
    """Pairwise training or evaluation example."""

    left_profile_id: str
    right_profile_id: str
    left_canonical_id: str
    right_canonical_id: str
    split: SplitName
    label: int
    features: dict[str, float]
    blocking_reasons: tuple[str, ...]


def assign_profile_splits(profiles: list[ProfileRecord]) -> dict[str, SplitName]:
    """Assign canonical ids to deterministic person-disjoint splits."""

    canonical_ids = sorted({profile.canonical_person_id for profile in profiles if profile.canonical_person_id})
    total = len(canonical_ids)
    train_cut = max(1, round(total * 0.56))
    val_cut = max(train_cut + 1, round(total * 0.78))

    split_map: dict[str, SplitName] = {}
    for index, canonical_id in enumerate(canonical_ids):
        if index < train_cut:
            split_map[canonical_id] = "train"
        elif index < val_cut:
            split_map[canonical_id] = "val"
        else:
            split_map[canonical_id] = "test"
    return split_map


def profiles_for_split(profiles: list[ProfileRecord], split_map: dict[str, SplitName], split: SplitName) -> list[ProfileRecord]:
    return [
        profile
        for profile in profiles
        if profile.canonical_person_id and split_map[profile.canonical_person_id] == split
    ]


def build_split_candidates(profiles: list[ProfileRecord], split_map: dict[str, SplitName], split: SplitName) -> list[BlockCandidate]:
    split_profiles = profiles_for_split(profiles, split_map, split)
    return generate_block_candidates(split_profiles, top_k=3)


def build_pair_examples(
    profiles: list[ProfileRecord],
    candidates: list[BlockCandidate],
    extractor: PairFeatureExtractor,
    split_map: dict[str, SplitName],
    split: SplitName,
) -> list[PairExample]:
    """Convert blocked candidates into typed pair examples."""

    profile_lookup = {profile.profile_id: profile for profile in profiles_for_split(profiles, split_map, split)}
    examples: list[PairExample] = []
    for candidate in candidates:
        left = profile_lookup[candidate.left_profile_id]
        right = profile_lookup[candidate.right_profile_id]
        examples.append(
            PairExample(
                left_profile_id=left.profile_id,
                right_profile_id=right.profile_id,
                left_canonical_id=left.canonical_person_id or "unknown",
                right_canonical_id=right.canonical_person_id or "unknown",
                split=split,
                label=int(left.canonical_person_id == right.canonical_person_id),
                features=extractor.featurize_pair(left, right),
                blocking_reasons=candidate.reasons,
            )
        )
    return examples


def build_examples_for_profiles(
    profiles: list[ProfileRecord],
    extractor: PairFeatureExtractor,
    *,
    split: SplitName = "test",
) -> list[PairExample]:
    """Build candidate examples for an arbitrary profile subset."""

    profile_lookup = {profile.profile_id: profile for profile in profiles}
    examples: list[PairExample] = []
    for candidate in generate_block_candidates(profiles, top_k=3):
        left = profile_lookup[candidate.left_profile_id]
        right = profile_lookup[candidate.right_profile_id]
        examples.append(
            PairExample(
                left_profile_id=left.profile_id,
                right_profile_id=right.profile_id,
                left_canonical_id=left.canonical_person_id or "unknown",
                right_canonical_id=right.canonical_person_id or "unknown",
                split=split,
                label=int(left.canonical_person_id == right.canonical_person_id),
                features=extractor.featurize_pair(left, right),
                blocking_reasons=candidate.reasons,
            )
        )
    return examples
