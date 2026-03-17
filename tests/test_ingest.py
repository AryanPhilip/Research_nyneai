from __future__ import annotations

from nyne_er_lab.datasets import load_raw_pages
from nyne_er_lab.ingest import compose_normalized_text, parse_raw_page, parse_raw_pages
from nyne_er_lab.schemas import RawProfilePage


def test_parse_all_seed_raw_pages() -> None:
    pages = load_raw_pages()
    profiles = parse_raw_pages(pages)
    assert len(profiles) == len(pages)
    assert all(profile.supporting_spans for profile in profiles)


def test_parse_github_profile_extracts_expected_fields() -> None:
    page = next(page for page in load_raw_pages() if page.page_id == "andrej_github_raw")
    profile = parse_raw_page(page)

    assert profile.display_name == "Andrej Karpathy"
    assert profile.headline == "Building foundation models and learning systems."
    assert profile.organizations[0].name == "OpenAI"
    assert "deep learning" in profile.topics
    assert str(profile.outbound_links[0]) == "https://karpathy.ai/"


def test_parse_personal_site_extracts_education_and_links() -> None:
    page = next(page for page in load_raw_pages() if page.page_id == "chip_personal_raw")
    profile = parse_raw_page(page)

    assert profile.education[0].institution == "Stanford University"
    assert profile.locations == ["San Francisco"]
    assert str(profile.outbound_links[0]) == "https://github.com/chiphuyen"


def test_parse_podcast_guest_extracts_bio_and_topics() -> None:
    page = next(page for page in load_raw_pages() if page.page_id == "jay_podcast_raw")
    profile = parse_raw_page(page)

    assert "helping engineers understand transformers" in profile.bio_text.lower()
    assert "language models" in profile.topics


def test_parser_falls_back_to_hint_and_raw_text_when_html_is_missing() -> None:
    page = RawProfilePage(
        page_id="fallback_profile_raw",
        source_type="personal_site",
        url="https://example.org/fallback",
        display_name_hint="Fallback Person",
        raw_text="Fallback Person writes about machine learning platforms.",
        html=None,
    )
    profile = parse_raw_page(page)

    assert profile.display_name == "Fallback Person"
    assert profile.bio_text == page.raw_text


def test_normalized_text_contains_core_retrieval_signals() -> None:
    page = next(page for page in load_raw_pages() if page.page_id == "jay_github_raw")
    profile = parse_raw_page(page)
    normalized = compose_normalized_text(profile)

    assert "jay alammar" in normalized
    assert "transformers" in normalized
    assert "jalammar.github.io" in normalized
