"""Feature generation module."""

from .dataset import (
    PairExample,
    assign_profile_splits,
    build_examples_for_profiles,
    build_pair_examples,
    build_split_candidates,
    profiles_for_split,
)
from .extractor import PairFeatureExtractor

__all__ = [
    "PairExample",
    "PairFeatureExtractor",
    "assign_profile_splits",
    "build_examples_for_profiles",
    "build_pair_examples",
    "build_split_candidates",
    "profiles_for_split",
]
