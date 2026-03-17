"""Typed contracts used across ingestion, matching, evaluation, and demo layers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


SourceType = Literal[
    "github",
    "personal_site",
    "conference_bio",
    "company_profile",
    "podcast_guest",
    "huggingface",
]

DecisionType = Literal["match", "non_match", "abstain"]
ConfidenceBand = Literal["high", "medium", "low", "uncertain"]


class TextSpan(BaseModel):
    """Evidence span grounded in a source document."""

    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(description="Field or section that contains the evidence.")
    snippet: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_char_range(self) -> "TextSpan":
        if self.start_char is not None and self.end_char is not None and self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class OrganizationClaim(BaseModel):
    """Normalized organization mention extracted from a profile."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    role: str | None = None
    start_year: int | None = Field(default=None, ge=1950, le=2100)
    end_year: int | None = Field(default=None, ge=1950, le=2100)

    @model_validator(mode="after")
    def validate_year_range(self) -> "OrganizationClaim":
        if self.start_year and self.end_year and self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")
        return self


class EducationClaim(BaseModel):
    """Normalized education mention extracted from a profile."""

    model_config = ConfigDict(extra="forbid")

    institution: str = Field(min_length=1)
    degree: str | None = None
    graduation_year: int | None = Field(default=None, ge=1950, le=2100)


class RawProfilePage(BaseModel):
    """Raw public page before extraction."""

    model_config = ConfigDict(extra="forbid")

    page_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    source_type: SourceType
    url: HttpUrl
    canonical_person_id: str | None = Field(default=None, pattern=r"^[a-z0-9_\-]+$")
    display_name_hint: str | None = None
    fetched_at: datetime | None = None
    title: str | None = None
    html: str | None = None
    raw_text: str = Field(min_length=1)


class ProfileRecord(BaseModel):
    """Normalized public identity profile."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    canonical_person_id: str | None = Field(default=None, pattern=r"^[a-z0-9_\-]+$")
    source_type: SourceType
    url: HttpUrl
    display_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    headline: str | None = None
    bio_text: str = Field(min_length=1)
    organizations: list[OrganizationClaim] = Field(default_factory=list)
    education: list[EducationClaim] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    outbound_links: list[HttpUrl] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    timestamps: list[str] = Field(default_factory=list)
    raw_text: str = Field(min_length=1)
    supporting_spans: list[TextSpan] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("aliases", "locations", "topics", "timestamps")
    @classmethod
    def validate_str_lists(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value and value.strip()]

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_supporting_spans(self) -> "ProfileRecord":
        if not self.supporting_spans:
            raise ValueError("supporting_spans must contain at least one grounded span")
        return self


class EvidenceSignal(BaseModel):
    """Single supporting or contradicting signal between two profiles."""

    model_config = ConfigDict(extra="forbid")

    signal_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    source_profile_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    target_profile_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    supporting_span: TextSpan | None = None


class CandidatePair(BaseModel):
    """Scored candidate pair for entity resolution."""

    model_config = ConfigDict(extra="forbid")

    left_profile_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    right_profile_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    blocking_reasons: list[str] = Field(default_factory=list)
    feature_vector: dict[str, float] = Field(default_factory=dict)
    match_score: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    decision: DecisionType


class EvidenceCard(BaseModel):
    """Human-readable evidence summary for a pairwise decision."""

    model_config = ConfigDict(extra="forbid")

    left_profile_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    right_profile_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    supporting_signals: list[EvidenceSignal] = Field(default_factory=list)
    contradicting_signals: list[EvidenceSignal] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    final_explanation: str = Field(min_length=1)


class CanonicalIdentity(BaseModel):
    """Clustered canonical identity assembled from multiple profiles."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    member_profile_ids: list[str] = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    confidence_band: ConfidenceBand
    key_orgs: list[str] = Field(default_factory=list)
    key_links: list[HttpUrl] = Field(default_factory=list)
