"""Parsers for curated public profile page snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup

from nyne_er_lab.schemas import (
    EducationClaim,
    OrganizationClaim,
    ProfileRecord,
    RawProfilePage,
    TextSpan,
)


@dataclass(frozen=True)
class SourceSelectors:
    display_name: tuple[str, ...]
    headline: tuple[str, ...]
    bio: tuple[str, ...]
    organizations: tuple[str, ...]
    education: tuple[str, ...]
    locations: tuple[str, ...]
    topics: tuple[str, ...]
    timestamps: tuple[str, ...]
    links: tuple[str, ...]


SELECTORS: dict[str, SourceSelectors] = {
    "github": SourceSelectors(
        display_name=(".p-name",),
        headline=(".p-note",),
        bio=(".user-profile-bio",),
        organizations=(".org-list li",),
        education=(".edu-list li",),
        locations=(".profile-location",),
        topics=(".topic-list li",),
        timestamps=(".time-list li",),
        links=(".profile-links a",),
    ),
    "personal_site": SourceSelectors(
        display_name=("h1.page-title",),
        headline=("p.tagline",),
        bio=("section.about",),
        organizations=("ul.affiliations li",),
        education=("ul.education li",),
        locations=("span.location",),
        topics=("ul.topics li",),
        timestamps=("ul.timeline li",),
        links=("nav.links a",),
    ),
    "conference_bio": SourceSelectors(
        display_name=(".speaker-name",),
        headline=(".speaker-headline",),
        bio=(".speaker-bio",),
        organizations=(".speaker-orgs li",),
        education=(".speaker-education li",),
        locations=(".speaker-location",),
        topics=(".speaker-topics li",),
        timestamps=(".speaker-years li",),
        links=(".speaker-links a",),
    ),
    "company_profile": SourceSelectors(
        display_name=(".team-name",),
        headline=(".team-role",),
        bio=(".team-bio",),
        organizations=(".team-orgs li",),
        education=(".team-education li",),
        locations=(".team-location",),
        topics=(".team-topics li",),
        timestamps=(".team-years li",),
        links=(".team-links a",),
    ),
    "podcast_guest": SourceSelectors(
        display_name=(".guest-name",),
        headline=(".guest-headline",),
        bio=(".guest-bio",),
        organizations=(".guest-orgs li",),
        education=(".guest-education li",),
        locations=(".guest-location",),
        topics=(".guest-topics li",),
        timestamps=(".guest-years li",),
        links=(".guest-links a",),
    ),
    "huggingface": SourceSelectors(
        display_name=(".hf-name",),
        headline=(".hf-headline",),
        bio=(".hf-bio",),
        organizations=(".hf-orgs li",),
        education=(".hf-education li",),
        locations=(".hf-location",),
        topics=(".hf-topics li",),
        timestamps=(".hf-years li",),
        links=(".hf-links a",),
    ),
}


def _select_first_text(soup: BeautifulSoup, selectors: tuple[str, ...]) -> str | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if text:
                return text
    return None


def _select_all_text(soup: BeautifulSoup, selectors: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = node.get_text(" ", strip=True)
            if text:
                values.append(text)
    return values


def _parse_orgs(soup: BeautifulSoup, selectors: tuple[str, ...]) -> list[OrganizationClaim]:
    orgs: list[OrganizationClaim] = []
    for selector in selectors:
        for node in soup.select(selector):
            name = node.get("data-name") or node.get_text(" ", strip=True)
            role = node.get("data-role")
            start_year = int(node["data-start-year"]) if node.get("data-start-year") else None
            end_year = int(node["data-end-year"]) if node.get("data-end-year") else None
            orgs.append(
                OrganizationClaim(
                    name=name,
                    role=role,
                    start_year=start_year,
                    end_year=end_year,
                )
            )
    return orgs


def _parse_education(soup: BeautifulSoup, selectors: tuple[str, ...]) -> list[EducationClaim]:
    education: list[EducationClaim] = []
    for selector in selectors:
        for node in soup.select(selector):
            institution = node.get("data-institution") or node.get_text(" ", strip=True)
            degree = node.get("data-degree")
            graduation_year = int(node["data-graduation-year"]) if node.get("data-graduation-year") else None
            education.append(
                EducationClaim(
                    institution=institution,
                    degree=degree,
                    graduation_year=graduation_year,
                )
            )
    return education


def _parse_links(soup: BeautifulSoup, selectors: tuple[str, ...]) -> list[str]:
    links: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            href = node.get("href")
            if href:
                links.append(href)
    return links


def _make_supporting_spans(headline: str | None, bio_text: str, raw_text: str) -> list[TextSpan]:
    spans: list[TextSpan] = []
    if headline:
        spans.append(TextSpan(field_name="headline", snippet=headline))
    if bio_text:
        spans.append(TextSpan(field_name="bio_text", snippet=bio_text[:240]))
    if not spans:
        spans.append(TextSpan(field_name="raw_text", snippet=raw_text[:240]))
    return spans


def parse_raw_page(page: RawProfilePage) -> ProfileRecord:
    """Parse a raw public page snapshot into a normalized profile."""

    if page.source_type not in SELECTORS:
        raise ValueError(f"Unsupported source_type: {page.source_type}")

    selectors = SELECTORS[page.source_type]
    soup = BeautifulSoup(page.html or "<html></html>", "html.parser")

    display_name = _select_first_text(soup, selectors.display_name) or page.display_name_hint
    if not display_name:
        raise ValueError(f"Unable to parse display name for {page.page_id}")

    headline = _select_first_text(soup, selectors.headline)
    bio_text = _select_first_text(soup, selectors.bio) or page.raw_text

    return ProfileRecord(
        profile_id=page.page_id.removesuffix("_raw"),
        canonical_person_id=page.canonical_person_id,
        source_type=page.source_type,
        url=page.url,
        display_name=display_name,
        aliases=[],
        headline=headline,
        bio_text=bio_text,
        organizations=_parse_orgs(soup, selectors.organizations),
        education=_parse_education(soup, selectors.education),
        locations=_select_all_text(soup, selectors.locations),
        outbound_links=_parse_links(soup, selectors.links),
        topics=_select_all_text(soup, selectors.topics),
        timestamps=_select_all_text(soup, selectors.timestamps),
        raw_text=page.raw_text,
        supporting_spans=_make_supporting_spans(headline, bio_text, page.raw_text),
        metadata={"page_id": page.page_id, "title": page.title or ""},
    )


def parse_raw_pages(pages: list[RawProfilePage]) -> list[ProfileRecord]:
    """Parse a list of raw pages into normalized profiles."""

    return [parse_raw_page(page) for page in pages]
