from __future__ import annotations

import json

from nyne_er_lab.datasets import load_raw_pages, load_seed_profiles
from nyne_er_lab.schemas import (
    CandidatePair,
    CanonicalIdentity,
    EvidenceCard,
    EvidenceSignal,
    ProfileRecord,
    TextSpan,
)


def test_seed_profiles_load_and_validate() -> None:
    profiles = load_seed_profiles()
    assert len(profiles) >= 60
    assert all(isinstance(profile, ProfileRecord) for profile in profiles)
    stage1_ids = {profile.canonical_person_id for profile in profiles if profile.metadata.get("seed_group") == "stage1"}
    assert {
        "andrej_karpathy",
        "chip_huyen",
        "jay_alammar",
        "lilian_weng",
        "sebastian_raschka",
        "hamel_husain",
    }.issubset(stage1_ids)


def test_raw_pages_load_and_validate() -> None:
    raw_pages = load_raw_pages()
    assert len(raw_pages) == 12
    assert raw_pages[0].source_type == "github"


def test_profile_round_trip_json() -> None:
    profile = load_seed_profiles()[0]
    payload = profile.model_dump_json()
    restored = ProfileRecord.model_validate_json(payload)
    assert restored == profile


def test_candidate_pair_contract() -> None:
    pair = CandidatePair(
        left_profile_id="andrej_github",
        right_profile_id="andrej_personal",
        blocking_reasons=["shared_name_token", "shared_org"],
        feature_vector={"name_similarity": 0.98, "org_overlap": 1.0},
        match_score=0.96,
        calibrated_confidence=0.94,
        decision="match",
    )
    assert pair.decision == "match"


def test_evidence_and_identity_contracts() -> None:
    signal = EvidenceSignal(
        signal_type="shared_org",
        description="Both profiles mention OpenAI.",
        weight=0.9,
        source_profile_id="andrej_github",
        target_profile_id="andrej_personal",
        supporting_span=TextSpan(field_name="bio_text", snippet="OpenAI"),
    )
    card = EvidenceCard(
        left_profile_id="andrej_github",
        right_profile_id="andrej_personal",
        supporting_signals=[signal],
        contradicting_signals=[],
        reason_codes=["shared_org", "shared_topic"],
        final_explanation="Strong name and organization overlap indicate a likely match.",
    )
    identity = CanonicalIdentity(
        entity_id="andrej_karpathy",
        member_profile_ids=["andrej_github", "andrej_personal"],
        canonical_name="Andrej Karpathy",
        summary="AI researcher and educator working on deep learning systems.",
        confidence_band="high",
        key_orgs=["OpenAI", "Tesla"],
        key_links=["https://karpathy.ai/"],
    )

    assert card.supporting_signals[0].signal_type == "shared_org"
    assert identity.confidence_band == "high"


def test_fixture_serialization_is_plain_json() -> None:
    profiles = load_seed_profiles()
    serialized = json.dumps([profile.model_dump(mode="json") for profile in profiles])
    restored = [ProfileRecord.model_validate(item) for item in json.loads(serialized)]
    assert restored[1].display_name == profiles[1].display_name
