"""Dataset loading helpers for curated fixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .schemas import OrganizationClaim, ProfileRecord, RawProfilePage, TextSpan


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class DatasetBundle:
    """Named dataset bundle used by benchmarks and demo flows."""

    name: str
    description: str
    profiles: list[ProfileRecord]
    contains_synthetic: bool = False
    headline: bool = False


def load_seed_profiles(path: Path | None = None) -> list[ProfileRecord]:
    fixture_path = path or DATA_DIR / "fixtures" / "seed_profiles.json"
    payload = json.loads(fixture_path.read_text())
    return [ProfileRecord.model_validate(item) for item in payload]


def load_raw_pages(path: Path | None = None) -> list[RawProfilePage]:
    fixture_path = path or DATA_DIR / "raw" / "seed_raw_pages.json"
    payload = json.loads(fixture_path.read_text())
    return [RawProfilePage.model_validate(item) for item in payload]


def _initial_alias(display_name: str) -> str:
    parts = display_name.split()
    if len(parts) < 2:
        return display_name
    return f"{parts[0][0]}. {parts[-1]}"


def _synthetic_positive_profile(canonical_id: str, profiles: list[ProfileRecord], index: int) -> ProfileRecord:
    anchor = profiles[0]
    org = profiles[0].organizations[:1]
    education = profiles[0].education[:1]
    topic_slice = profiles[0].topics[:2] if profiles[0].topics else ["machine learning"]
    source_type = "huggingface" if index % 2 == 0 else "company_profile"
    url = f"https://example.org/benchmark/{canonical_id}/{source_type}"
    headline = f"Research profile on {', '.join(topic_slice)}."
    bio_text = (
        f"{anchor.display_name} works across {', '.join(topic_slice)} and production-facing AI systems. "
        f"This synthetic profile is derived from curated public-profile evidence."
    )
    return ProfileRecord(
        profile_id=f"{canonical_id}_augmented",
        canonical_person_id=canonical_id,
        source_type=source_type,
        url=url,
        display_name=anchor.display_name,
        aliases=[_initial_alias(anchor.display_name)],
        headline=headline,
        bio_text=bio_text,
        organizations=org,
        education=education,
        locations=anchor.locations[:1] or ["United States"],
        outbound_links=anchor.outbound_links[:1],
        topics=topic_slice + ["ai systems"],
        timestamps=anchor.timestamps[-2:] if anchor.timestamps else ["2024"],
        raw_text=bio_text,
        supporting_spans=[TextSpan(field_name="bio_text", snippet=bio_text[:180])],
        metadata={"seed_group": "synthetic_positive", "generated_from": canonical_id},
    )


def _synthetic_conflict_profile(canonical_id: str, profiles: list[ProfileRecord], index: int) -> ProfileRecord:
    anchor = profiles[0]
    conflict_specs = [
        ("Policy Grid", "Research lead", ["ai governance", "regulation", "policy"], "Washington"),
        ("NorthBridge Capital", "Quant analyst", ["portfolio risk", "finance", "quant analytics"], "New York"),
        ("BrightFrame Studio", "Creative director", ["brand systems", "design ops", "media"], "Los Angeles"),
        ("Civic Compute", "Program manager", ["public sector", "compliance", "strategy"], "Chicago"),
        ("Blue River Health", "Data strategist", ["health analytics", "operations", "care systems"], "Boston"),
        ("Harbor Retail", "Analytics lead", ["consumer analytics", "pricing", "retail"], "Seattle"),
    ]
    org_name, role, topics, location = conflict_specs[index % len(conflict_specs)]
    bio_text = (
        f"{anchor.display_name} works on {', '.join(topics)} at {org_name}. "
        f"This is a hard-negative same-name collision with intentionally conflicting structured evidence."
    )
    return ProfileRecord(
        profile_id=f"{canonical_id}_conflict_generated",
        canonical_person_id=f"{canonical_id}_conflict_generated",
        source_type="company_profile",
        url=f"https://example.org/conflicts/{canonical_id}",
        display_name=anchor.display_name,
        aliases=[_initial_alias(anchor.display_name)],
        headline=f"{role} focused on {topics[0]}.",
        bio_text=bio_text,
        organizations=[OrganizationClaim(name=org_name, role=role)],
        education=[],
        locations=[location],
        outbound_links=[f"https://example.org/{org_name.lower().replace(' ', '-')}"],
        topics=topics,
        timestamps=["2024"],
        raw_text=bio_text,
        supporting_spans=[TextSpan(field_name="bio_text", snippet=bio_text[:180])],
        metadata={"seed_group": "synthetic_conflict", "generated_from": canonical_id},
    )


def _seed_group_profiles(seed_group: str) -> list[ProfileRecord]:
    seed_profiles = load_seed_profiles()
    return [profile for profile in seed_profiles if profile.metadata.get("seed_group") == seed_group]


def _stage1_profiles() -> list[ProfileRecord]:
    return _seed_group_profiles("stage1")


def _hard_negative_profiles() -> list[ProfileRecord]:
    return _seed_group_profiles("hard_negative")


def _synthetic_stress_profiles() -> list[ProfileRecord]:
    stage1_profiles = _stage1_profiles()
    grouped: dict[str, list[ProfileRecord]] = defaultdict(list)
    for profile in stage1_profiles:
        grouped[profile.canonical_person_id].append(profile)

    synthetic_profiles: list[ProfileRecord] = []
    for index, canonical_id in enumerate(sorted(grouped)):
        profiles = grouped[canonical_id]
        synthetic_profiles.append(_synthetic_positive_profile(canonical_id, profiles, index))
        synthetic_profiles.append(_synthetic_conflict_profile(canonical_id, profiles, index))
    return stage1_profiles + synthetic_profiles


def load_dataset(name: str = "real_curated_core") -> DatasetBundle:
    """Load a named dataset bundle."""

    datasets = {
        "real_curated_core": DatasetBundle(
            name="real_curated_core",
            description="Headline benchmark of curated public AI/ML builder profiles only.",
            profiles=_stage1_profiles(),
            contains_synthetic=False,
            headline=True,
        ),
        "hard_negative_bank": DatasetBundle(
            name="hard_negative_bank",
            description="Curated same-name and adjacent-domain distractors layered onto the real corpus.",
            profiles=_stage1_profiles() + _hard_negative_profiles(),
            contains_synthetic=False,
            headline=False,
        ),
        "synthetic_stress": DatasetBundle(
            name="synthetic_stress",
            description="Robustness-only dataset with generated positive and conflict variants.",
            profiles=_synthetic_stress_profiles(),
            contains_synthetic=True,
            headline=False,
        ),
    }
    if name not in datasets:
        available = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return datasets[name]


def available_datasets() -> list[str]:
    """Return stable dataset names."""

    return ["real_curated_core", "hard_negative_bank", "synthetic_stress"]


def load_benchmark_profiles(dataset_name: str = "real_curated_core") -> list[ProfileRecord]:
    """Compatibility wrapper returning profiles for a named benchmark dataset."""

    return load_dataset(dataset_name).profiles
