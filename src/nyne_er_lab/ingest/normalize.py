"""Normalization helpers for public profile records."""

from __future__ import annotations

from urllib.parse import urlparse

from nyne_er_lab.schemas import ProfileRecord


def compose_normalized_text(profile: ProfileRecord) -> str:
    """Build a normalized text view used by blocking and feature extraction."""

    fields: list[str] = [
        profile.display_name,
        " ".join(profile.aliases),
        profile.headline or "",
        profile.bio_text,
        " ".join(org.name for org in profile.organizations),
        " ".join(filter(None, (org.role for org in profile.organizations))),
        " ".join(edu.institution for edu in profile.education),
        " ".join(filter(None, (edu.degree for edu in profile.education))),
        " ".join(profile.locations),
        " ".join(profile.topics),
        " ".join(profile.timestamps),
    ]

    outbound_domains = [
        urlparse(str(link)).netloc.replace("www.", "")
        for link in profile.outbound_links
    ]
    fields.append(" ".join(outbound_domains))

    normalized = " ".join(part.strip().lower() for part in fields if part and part.strip())
    return " ".join(normalized.split())
