"""Web search integration for discovering public profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchHit:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    source_type_guess: str


SOURCE_TYPE_HINTS = {
    "github.com": "github",
    "huggingface.co": "huggingface",
    "linkedin.com": "company_profile",
    "twitter.com": "personal_site",
    "x.com": "personal_site",
    "medium.com": "personal_site",
    "substack.com": "personal_site",
    "wordpress.com": "personal_site",
    "youtube.com": "podcast_guest",
    "podcasts.apple.com": "podcast_guest",
    "spotify.com": "podcast_guest",
    "speakerdeck.com": "conference_bio",
    "slideshare.net": "conference_bio",
    "scholar.google.com": "conference_bio",
    "arxiv.org": "conference_bio",
}


def _guess_source_type(url: str) -> str:
    """Guess source_type from URL domain."""
    url_lower = url.lower()
    for domain, source_type in SOURCE_TYPE_HINTS.items():
        if domain in url_lower:
            return source_type
    return "personal_site"


def search_person(name: str, max_results: int = 8) -> list[SearchHit]:
    """Search the web for public profiles of a person using DuckDuckGo.

    Returns a list of SearchHit results. Falls back to an empty list
    if the search fails (rate limit, network error, etc).
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []

    query = f"{name} profile"

    try:
        raw_results = DDGS().text(query, max_results=max_results)
    except Exception:
        # DuckDuckGo can rate-limit or fail — never crash the app
        return []

    hits: list[SearchHit] = []
    for result in raw_results:
        title = result.get("title", "")
        url = result.get("href", result.get("link", ""))
        snippet = result.get("body", result.get("snippet", ""))
        if not url:
            continue
        hits.append(SearchHit(
            title=title,
            url=url,
            snippet=snippet,
            source_type_guess=_guess_source_type(url),
        ))
    return hits


def search_and_build_profiles(
    name: str,
    max_results: int = 6,
) -> list[dict]:
    """Search for a person and return profile-ready dicts from snippets.

    Each dict has keys: display_name, headline, bio, source_type, url,
    topics — ready to be passed to `profile_from_text()`.
    """
    hits = search_person(name, max_results=max_results)
    profiles = []
    seen_urls = set()
    for hit in hits:
        if hit.url in seen_urls:
            continue
        seen_urls.add(hit.url)
        profiles.append({
            "display_name": name,
            "headline": hit.title[:120] if hit.title else "",
            "bio": hit.snippet[:500] if hit.snippet else f"Profile of {name}.",
            "source_type": hit.source_type_guess,
            "url": hit.url,
            "topics": [],
        })
    return profiles
