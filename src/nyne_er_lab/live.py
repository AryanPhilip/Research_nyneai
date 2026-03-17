"""Live profile resolution — resolve ad-hoc profiles against known identities."""

from __future__ import annotations

import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone

from nyne_er_lab.cluster.resolver import ResolvedPair, generate_evidence_card
from nyne_er_lab.features.dataset import PairExample
from nyne_er_lab.features.extractor import PairFeatureExtractor
from nyne_er_lab.ingest import parse_raw_page
from nyne_er_lab.models.hybrid import TrainedHybridMatcher
from nyne_er_lab.schemas import (
    OrganizationClaim,
    ProfileRecord,
    RawProfilePage,
    TextSpan,
)


@dataclass
class LiveResult:
    """Result of resolving a live profile against the known corpus."""

    query_profile: ProfileRecord
    matches: list[ResolvedPair]
    best_match_profile: ProfileRecord | None
    best_match_identity_name: str | None


def _stable_id(text: str) -> str:
    return "live-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch raw HTML from a URL. Returns empty string on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EntityResolutionLab/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return ""


def profile_from_url(
    url: str,
    source_type: str,
    display_name_hint: str | None = None,
) -> ProfileRecord | None:
    """Fetch a URL, parse its HTML into a ProfileRecord."""
    html = fetch_url(url)
    if not html:
        return None
    page_id = _stable_id(url)
    try:
        raw_page = RawProfilePage(
            page_id=page_id,
            source_type=source_type,
            url=url,
            display_name_hint=display_name_hint,
            fetched_at=datetime.now(timezone.utc),
            title=display_name_hint,
            html=html,
            raw_text=html[:5000],
        )
        return parse_raw_page(raw_page)
    except (ValueError, Exception):
        return None


def profile_from_text(
    display_name: str,
    bio: str,
    source_type: str = "personal_site",
    headline: str = "",
    organizations: list[str] | None = None,
    locations: list[str] | None = None,
    topics: list[str] | None = None,
    url: str = "https://example.com/profile",
) -> ProfileRecord:
    """Construct a ProfileRecord directly from user-supplied text fields."""
    orgs = organizations or []
    page_id = _stable_id(f"{display_name}-{bio[:50]}")
    bio_text = bio if bio.strip() else f"{display_name} professional profile."
    return ProfileRecord(
        profile_id=page_id,
        canonical_person_id=None,
        source_type=source_type,
        url=url,
        display_name=display_name,
        aliases=[],
        headline=headline or None,
        bio_text=bio_text,
        organizations=[OrganizationClaim(name=o) for o in orgs if o.strip()],
        education=[],
        locations=locations or [],
        outbound_links=[],
        topics=topics or [],
        timestamps=[],
        raw_text=f"{display_name} {headline} {bio_text}",
        supporting_spans=[TextSpan(field_name="bio_text", snippet=bio_text[:240])],
        metadata={},
    )


def resolve_live_profile(
    query: ProfileRecord,
    corpus_profiles: list[ProfileRecord],
    extractor: PairFeatureExtractor,
    matcher: TrainedHybridMatcher,
    identities,
) -> LiveResult:
    """Run the full pipeline on a single query profile against the corpus."""
    from nyne_er_lab.blocking.blocker import rule_reasons

    # Generate pairs between query and every corpus profile
    examples: list[PairExample] = []
    for corpus_profile in corpus_profiles:
        reasons = rule_reasons(query, corpus_profile)
        # Always generate a pair even without blocking reasons for live search
        features = extractor.featurize_pair(query, corpus_profile)
        examples.append(PairExample(
            left_profile_id=query.profile_id,
            right_profile_id=corpus_profile.profile_id,
            left_canonical_id="unknown",
            right_canonical_id=corpus_profile.canonical_person_id or "unknown",
            split="test",
            label=0,
            features=features,
            blocking_reasons=tuple(sorted(reasons)) if reasons else ("live_search",),
        ))

    if not examples:
        return LiveResult(query_profile=query, matches=[], best_match_profile=None, best_match_identity_name=None)

    scores, decisions = matcher.score_examples(examples, extractor)

    profile_lookup = {p.profile_id: p for p in corpus_profiles + [query]}
    resolved: list[ResolvedPair] = []
    for ex, score, decision in zip(examples, scores, decisions):
        left = profile_lookup[ex.left_profile_id]
        right = profile_lookup[ex.right_profile_id]
        evidence_card = generate_evidence_card(left, right, ex, score, decision)
        resolved.append(ResolvedPair(
            left_profile_id=ex.left_profile_id,
            right_profile_id=ex.right_profile_id,
            score=score,
            decision=decision,
            evidence_card=evidence_card,
        ))

    # Sort by score descending
    resolved.sort(key=lambda r: -r.score)

    # Find best match
    best = None
    best_identity = None
    for rp in resolved:
        if rp.decision == "match":
            best = profile_lookup.get(rp.right_profile_id)
            # Find identity containing this profile
            if identities:
                for identity in identities:
                    if rp.right_profile_id in identity.member_profile_ids:
                        best_identity = identity.canonical_name
                        break
            break

    return LiveResult(
        query_profile=query,
        matches=resolved,
        best_match_profile=best,
        best_match_identity_name=best_identity,
    )
