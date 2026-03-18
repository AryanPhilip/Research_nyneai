use crate::live::TextProfileInput;
use anyhow::Result;
use reqwest::blocking::Client;
use reqwest::header::USER_AGENT;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use url::Url;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub source_type_guess: String,
}

pub fn search_person(name: &str, max_results: usize) -> Result<Vec<SearchHit>> {
    let query = format!("{name} profile");
    let client = Client::builder().build()?;
    let response = match client
        .get("https://duckduckgo.com/html/")
        .query(&[("q", query.as_str())])
        .header(USER_AGENT, "EntityResolutionLab/0.1")
        .send()
    {
        Ok(response) => response,
        Err(_) => return Ok(Vec::new()),
    };
    let body = match response.text() {
        Ok(body) => body,
        Err(_) => return Ok(Vec::new()),
    };

    let document = Html::parse_document(&body);
    let result_selector = Selector::parse(".result").expect("valid result selector");
    let link_selector = Selector::parse("a.result__a").expect("valid link selector");
    let snippet_selector = Selector::parse(".result__snippet").expect("valid snippet selector");

    let mut hits = Vec::new();
    for result in document.select(&result_selector) {
        let Some(link) = result.select(&link_selector).next() else {
            continue;
        };
        let title = link.text().collect::<Vec<_>>().join(" ").trim().to_string();
        let Some(raw_href) = link.value().attr("href") else {
            continue;
        };
        let url = decode_duckduckgo_href(raw_href);
        if url.is_empty() {
            continue;
        }
        let snippet = result
            .select(&snippet_selector)
            .next()
            .map(|node| node.text().collect::<Vec<_>>().join(" ").trim().to_string())
            .unwrap_or_default();
        hits.push(SearchHit {
            title,
            url: url.clone(),
            snippet,
            source_type_guess: guess_source_type(&url).to_string(),
        });
        if hits.len() >= max_results {
            break;
        }
    }
    Ok(hits)
}

pub fn search_and_build_profiles(name: &str, max_results: usize) -> Result<Vec<TextProfileInput>> {
    let hits = search_person(name, max_results)?;
    let mut profiles = Vec::new();
    let mut seen_urls = BTreeSet::new();
    for hit in hits {
        if !seen_urls.insert(hit.url.clone()) {
            continue;
        }
        profiles.push(TextProfileInput {
            display_name: name.to_string(),
            headline: hit.title.chars().take(120).collect(),
            bio: if hit.snippet.trim().is_empty() {
                format!("Profile of {name}.")
            } else {
                hit.snippet.chars().take(500).collect()
            },
            source_type: hit.source_type_guess,
            organizations: Vec::new(),
            locations: Vec::new(),
            topics: Vec::new(),
            url: hit.url,
        });
    }
    Ok(profiles)
}

fn guess_source_type(url: &str) -> &'static str {
    let url_lower = url.to_lowercase();
    for (domain, source_type) in [
        ("github.com", "github"),
        ("huggingface.co", "huggingface"),
        ("linkedin.com", "company_profile"),
        ("twitter.com", "personal_site"),
        ("x.com", "personal_site"),
        ("medium.com", "personal_site"),
        ("substack.com", "personal_site"),
        ("wordpress.com", "personal_site"),
        ("youtube.com", "podcast_guest"),
        ("podcasts.apple.com", "podcast_guest"),
        ("spotify.com", "podcast_guest"),
        ("speakerdeck.com", "conference_bio"),
        ("slideshare.net", "conference_bio"),
        ("scholar.google.com", "conference_bio"),
        ("arxiv.org", "conference_bio"),
    ] {
        if url_lower.contains(domain) {
            return source_type;
        }
    }
    "personal_site"
}

fn decode_duckduckgo_href(raw_href: &str) -> String {
    if let Ok(url) = Url::parse(raw_href) {
        if let Some(target) = url
            .query_pairs()
            .find_map(|(key, value)| (key == "uddg").then_some(value.into_owned()))
        {
            return target;
        }
    }
    raw_href.to_string()
}
