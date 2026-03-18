use crate::schemas::{
    EducationClaim, OrganizationClaim, ProfileRecord, RawProfilePage, SourceType, TextSpan,
};
use anyhow::Result;
use scraper::{ElementRef, Html, Selector};
use std::collections::BTreeMap;
use url::Url;

#[derive(Debug, Clone, Copy)]
struct SourceSelectors {
    display_name: &'static [&'static str],
    headline: &'static [&'static str],
    bio: &'static [&'static str],
    organizations: &'static [&'static str],
    education: &'static [&'static str],
    locations: &'static [&'static str],
    topics: &'static [&'static str],
    timestamps: &'static [&'static str],
    links: &'static [&'static str],
}

pub fn compose_normalized_text(profile: &ProfileRecord) -> String {
    let mut fields = vec![
        profile.display_name.clone(),
        profile.aliases.join(" "),
        profile.headline.clone().unwrap_or_default(),
        profile.bio_text.clone(),
        profile
            .organizations
            .iter()
            .map(|org| org.name.clone())
            .collect::<Vec<_>>()
            .join(" "),
        profile
            .organizations
            .iter()
            .filter_map(|org| org.role.clone())
            .collect::<Vec<_>>()
            .join(" "),
        profile
            .education
            .iter()
            .map(|edu| edu.institution.clone())
            .collect::<Vec<_>>()
            .join(" "),
        profile
            .education
            .iter()
            .filter_map(|edu| edu.degree.clone())
            .collect::<Vec<_>>()
            .join(" "),
        profile.locations.join(" "),
        profile.topics.join(" "),
        profile.timestamps.join(" "),
    ];

    let outbound_domains = profile
        .outbound_links
        .iter()
        .filter_map(|link| Some(strip_www(link.host_str()?.to_string())))
        .collect::<Vec<_>>()
        .join(" ");
    fields.push(outbound_domains);

    fields
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .map(|part| collapse_ws(&part.to_lowercase()))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn parse_raw_page(page: &RawProfilePage) -> Result<ProfileRecord> {
    let selectors = selectors_for(page.source_type);
    let html = Html::parse_document(page.html.as_deref().unwrap_or("<html></html>"));

    let display_name = select_first_text(&html, selectors.display_name)
        .or_else(|| page.display_name_hint.clone())
        .ok_or_else(|| anyhow::anyhow!("unable to parse display name for {}", page.page_id))?;
    let headline = select_first_text(&html, selectors.headline);
    let bio_text = select_first_text(&html, selectors.bio).unwrap_or_else(|| page.raw_text.clone());
    let profile_id = page
        .page_id
        .strip_suffix("_raw")
        .unwrap_or(&page.page_id)
        .to_string();

    ProfileRecord {
        profile_id,
        canonical_person_id: page.canonical_person_id.clone(),
        source_type: page.source_type,
        url: page.url.clone(),
        display_name,
        aliases: vec![],
        headline: headline.clone(),
        bio_text: bio_text.clone(),
        organizations: parse_orgs(&html, selectors.organizations)?,
        education: parse_education(&html, selectors.education)?,
        locations: select_all_text(&html, selectors.locations),
        outbound_links: parse_links(&html, selectors.links),
        topics: select_all_text(&html, selectors.topics),
        timestamps: select_all_text(&html, selectors.timestamps),
        raw_text: page.raw_text.clone(),
        supporting_spans: make_supporting_spans(headline, bio_text, page.raw_text.clone())?,
        metadata: BTreeMap::from([
            ("page_id".to_string(), page.page_id.clone()),
            ("title".to_string(), page.title.clone().unwrap_or_default()),
        ]),
    }
    .validated()
}

pub fn parse_raw_pages(pages: &[RawProfilePage]) -> Result<Vec<ProfileRecord>> {
    pages.iter().map(parse_raw_page).collect()
}

fn selectors_for(source_type: SourceType) -> SourceSelectors {
    match source_type {
        SourceType::Github => SourceSelectors {
            display_name: &[".p-name"],
            headline: &[".p-note"],
            bio: &[".user-profile-bio"],
            organizations: &[".org-list li"],
            education: &[".edu-list li"],
            locations: &[".profile-location"],
            topics: &[".topic-list li"],
            timestamps: &[".time-list li"],
            links: &[".profile-links a"],
        },
        SourceType::PersonalSite => SourceSelectors {
            display_name: &["h1.page-title"],
            headline: &["p.tagline"],
            bio: &["section.about"],
            organizations: &["ul.affiliations li"],
            education: &["ul.education li"],
            locations: &["span.location"],
            topics: &["ul.topics li"],
            timestamps: &["ul.timeline li"],
            links: &["nav.links a"],
        },
        SourceType::ConferenceBio => SourceSelectors {
            display_name: &[".speaker-name"],
            headline: &[".speaker-headline"],
            bio: &[".speaker-bio"],
            organizations: &[".speaker-orgs li"],
            education: &[".speaker-education li"],
            locations: &[".speaker-location"],
            topics: &[".speaker-topics li"],
            timestamps: &[".speaker-years li"],
            links: &[".speaker-links a"],
        },
        SourceType::CompanyProfile => SourceSelectors {
            display_name: &[".team-name"],
            headline: &[".team-role"],
            bio: &[".team-bio"],
            organizations: &[".team-orgs li"],
            education: &[".team-education li"],
            locations: &[".team-location"],
            topics: &[".team-topics li"],
            timestamps: &[".team-years li"],
            links: &[".team-links a"],
        },
        SourceType::PodcastGuest => SourceSelectors {
            display_name: &[".guest-name"],
            headline: &[".guest-headline"],
            bio: &[".guest-bio"],
            organizations: &[".guest-orgs li"],
            education: &[".guest-education li"],
            locations: &[".guest-location"],
            topics: &[".guest-topics li"],
            timestamps: &[".guest-years li"],
            links: &[".guest-links a"],
        },
        SourceType::Huggingface => SourceSelectors {
            display_name: &[".hf-name"],
            headline: &[".hf-headline"],
            bio: &[".hf-bio"],
            organizations: &[".hf-orgs li"],
            education: &[".hf-education li"],
            locations: &[".hf-location"],
            topics: &[".hf-topics li"],
            timestamps: &[".hf-years li"],
            links: &[".hf-links a"],
        },
    }
}

fn select_first_text(document: &Html, selectors: &[&str]) -> Option<String> {
    for selector in selectors {
        let selector = Selector::parse(selector).ok()?;
        if let Some(node) = document.select(&selector).next() {
            let text = node_text(node);
            if !text.is_empty() {
                return Some(text);
            }
        }
    }
    None
}

fn select_all_text(document: &Html, selectors: &[&str]) -> Vec<String> {
    let mut values = Vec::new();
    for selector in selectors {
        let Ok(selector) = Selector::parse(selector) else {
            continue;
        };
        for node in document.select(&selector) {
            let text = node_text(node);
            if !text.is_empty() {
                values.push(text);
            }
        }
    }
    values
}

fn parse_orgs(document: &Html, selectors: &[&str]) -> Result<Vec<OrganizationClaim>> {
    let mut orgs = Vec::new();
    for selector in selectors {
        let Ok(selector) = Selector::parse(selector) else {
            continue;
        };
        for node in document.select(&selector) {
            let name = node
                .value()
                .attr("data-name")
                .map(|value| value.to_string())
                .unwrap_or_else(|| node_text(node));
            let role = node.value().attr("data-role").map(|value| value.to_string());
            let start_year = node.value().attr("data-start-year").and_then(|value| value.parse::<i32>().ok());
            let end_year = node.value().attr("data-end-year").and_then(|value| value.parse::<i32>().ok());
            orgs.push(
                OrganizationClaim {
                    name,
                    role,
                    start_year,
                    end_year,
                }
                .validated()?,
            );
        }
    }
    Ok(orgs)
}

fn parse_education(document: &Html, selectors: &[&str]) -> Result<Vec<EducationClaim>> {
    let mut education = Vec::new();
    for selector in selectors {
        let Ok(selector) = Selector::parse(selector) else {
            continue;
        };
        for node in document.select(&selector) {
            let institution = node
                .value()
                .attr("data-institution")
                .map(|value| value.to_string())
                .unwrap_or_else(|| node_text(node));
            let degree = node.value().attr("data-degree").map(|value| value.to_string());
            let graduation_year = node
                .value()
                .attr("data-graduation-year")
                .and_then(|value| value.parse::<i32>().ok());
            education.push(
                EducationClaim {
                    institution,
                    degree,
                    graduation_year,
                }
                .validated()?,
            );
        }
    }
    Ok(education)
}

fn parse_links(document: &Html, selectors: &[&str]) -> Vec<Url> {
    let mut links = Vec::new();
    for selector in selectors {
        let Ok(selector) = Selector::parse(selector) else {
            continue;
        };
        for node in document.select(&selector) {
            if let Some(href) = node.value().attr("href") {
                if let Ok(url) = Url::parse(href) {
                    links.push(url);
                }
            }
        }
    }
    links
}

fn make_supporting_spans(headline: Option<String>, bio_text: String, raw_text: String) -> Result<Vec<TextSpan>> {
    let mut spans = Vec::new();
    if let Some(headline) = headline {
        spans.push(
            TextSpan {
                field_name: "headline".to_string(),
                snippet: headline,
                start_char: None,
                end_char: None,
            }
            .validated()?,
        );
    }
    if !bio_text.is_empty() {
        spans.push(
            TextSpan {
                field_name: "bio_text".to_string(),
                snippet: truncate_chars(&bio_text, 240),
                start_char: None,
                end_char: None,
            }
            .validated()?,
        );
    }
    if spans.is_empty() {
        spans.push(
            TextSpan {
                field_name: "raw_text".to_string(),
                snippet: truncate_chars(&raw_text, 240),
                start_char: None,
                end_char: None,
            }
            .validated()?,
        );
    }
    Ok(spans)
}

fn node_text(node: ElementRef<'_>) -> String {
    collapse_ws(&node.text().collect::<Vec<_>>().join(" "))
}

fn collapse_ws(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(value: &str, max_len: usize) -> String {
    value.chars().take(max_len).collect()
}

fn strip_www(domain: String) -> String {
    domain.strip_prefix("www.").unwrap_or(&domain).to_string()
}
