use crate::blocking::rule_reasons;
use crate::cluster::{ResolvedPair, generate_evidence_card};
use crate::features::{PairExample, PairFeatureExtractor, SplitName};
use crate::models::TrainedHybridMatcher;
use crate::schemas::{OrganizationClaim, ProfileRecord, RawProfilePage, SourceType, TextSpan};
use anyhow::Result;
use chrono::Utc;
use reqwest::blocking::Client;
use reqwest::header::USER_AGENT;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::str::FromStr;
use url::Url;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextProfileInput {
    pub display_name: String,
    pub bio: String,
    #[serde(default = "default_source_type")]
    pub source_type: String,
    #[serde(default)]
    pub headline: String,
    #[serde(default)]
    pub organizations: Vec<String>,
    #[serde(default)]
    pub locations: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default = "default_profile_url")]
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveResult {
    pub query_profile: ProfileRecord,
    pub matches: Vec<ResolvedPair>,
    pub best_match_profile: Option<ProfileRecord>,
    pub best_match_identity_name: Option<String>,
}

pub fn fetch_url(url: &str) -> String {
    let client = match Client::builder().build() {
        Ok(client) => client,
        Err(_) => return String::new(),
    };
    let response = match client
        .get(url)
        .header(USER_AGENT, "EntityResolutionLab/0.1")
        .send()
    {
        Ok(response) => response,
        Err(_) => return String::new(),
    };
    response.text().unwrap_or_default()
}

pub fn profile_from_url(
    url: &str,
    source_type: &str,
    display_name_hint: Option<&str>,
) -> Option<ProfileRecord> {
    let html = fetch_url(url);
    if html.is_empty() {
        return None;
    }
    let raw_page = RawProfilePage {
        page_id: stable_id(url),
        source_type: SourceType::from_str(source_type).ok()?,
        url: Url::parse(url).ok()?,
        canonical_person_id: None,
        display_name_hint: display_name_hint.map(str::to_string),
        fetched_at: Some(Utc::now()),
        title: display_name_hint.map(str::to_string),
        html: Some(html.clone()),
        raw_text: html.chars().take(5_000).collect(),
    }
    .validated()
    .ok()?;
    crate::ingest::parse_raw_page(&raw_page).ok()
}

pub fn profile_from_text(input: TextProfileInput) -> Result<ProfileRecord> {
    let source_type = SourceType::from_str(&input.source_type)?;
    let url = Url::parse(&input.url)?;
    let display_name = input.display_name;
    let headline = input.headline;
    let bio = input.bio;
    let bio_text = if bio.trim().is_empty() {
        format!("{display_name} professional profile.")
    } else {
        bio.trim().to_string()
    };
    ProfileRecord {
        profile_id: stable_id(&format!(
            "{}-{}",
            display_name,
            bio.chars().take(50).collect::<String>()
        )),
        canonical_person_id: None,
        source_type,
        url,
        display_name: display_name.clone(),
        aliases: Vec::new(),
        headline: (!headline.trim().is_empty()).then_some(headline.trim().to_string()),
        bio_text: bio_text.clone(),
        organizations: input
            .organizations
            .into_iter()
            .filter(|value| !value.trim().is_empty())
            .map(|value| OrganizationClaim {
                name: value,
                role: None,
                start_year: None,
                end_year: None,
            })
            .collect(),
        education: Vec::new(),
        locations: input.locations,
        outbound_links: Vec::new(),
        topics: input.topics,
        timestamps: Vec::new(),
        raw_text: format!("{} {} {}", display_name, headline, bio_text)
            .trim()
            .to_string(),
        supporting_spans: vec![TextSpan {
            field_name: "bio_text".to_string(),
            snippet: bio_text.chars().take(240).collect(),
            start_char: None,
            end_char: None,
        }],
        metadata: BTreeMap::new(),
    }
    .validated()
}

pub fn resolve_live_profile(
    query: &ProfileRecord,
    corpus_profiles: &[ProfileRecord],
    extractor: &PairFeatureExtractor,
    matcher: &TrainedHybridMatcher,
    identities: &[crate::schemas::CanonicalIdentity],
) -> Result<LiveResult> {
    let examples: Vec<PairExample> = corpus_profiles
        .iter()
        .map(|corpus_profile| PairExample {
            left_profile_id: query.profile_id.clone(),
            right_profile_id: corpus_profile.profile_id.clone(),
            left_canonical_id: "unknown".to_string(),
            right_canonical_id: corpus_profile
                .canonical_person_id
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            split: SplitName::Test,
            label: 0,
            features: extractor.featurize_pair(query, corpus_profile),
            blocking_reasons: {
                let reasons = rule_reasons(query, corpus_profile);
                if reasons.is_empty() {
                    vec!["live_search".to_string()]
                } else {
                    reasons.into_iter().collect()
                }
            },
        })
        .collect();
    if examples.is_empty() {
        return Ok(LiveResult {
            query_profile: query.clone(),
            matches: Vec::new(),
            best_match_profile: None,
            best_match_identity_name: None,
        });
    }

    let (scores, decisions) = matcher.score_examples(&examples, extractor, None);
    let profile_lookup: BTreeMap<String, ProfileRecord> = corpus_profiles
        .iter()
        .cloned()
        .chain(std::iter::once(query.clone()))
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect();

    let mut resolved = Vec::new();
    for ((example, score), decision) in examples.iter().zip(scores.iter()).zip(decisions.iter()) {
        let left = profile_lookup.get(&example.left_profile_id).expect("query exists");
        let right = profile_lookup.get(&example.right_profile_id).expect("candidate exists");
        resolved.push(ResolvedPair {
            left_profile_id: example.left_profile_id.clone(),
            right_profile_id: example.right_profile_id.clone(),
            score: *score,
            decision: decision.clone(),
            evidence_card: generate_evidence_card(left, right, example, *score, decision)?,
        });
    }
    resolved.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best_match_profile = None;
    let mut best_match_identity_name = None;
    for pair in &resolved {
        if pair.decision != "match" {
            continue;
        }
        best_match_profile = profile_lookup.get(&pair.right_profile_id).cloned();
        best_match_identity_name = identities
            .iter()
            .find(|identity| identity.member_profile_ids.contains(&pair.right_profile_id))
            .map(|identity| identity.canonical_name.clone());
        break;
    }

    Ok(LiveResult {
        query_profile: query.clone(),
        matches: resolved,
        best_match_profile,
        best_match_identity_name,
    })
}

fn stable_id(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    format!("live-{:x}", digest)[..17].to_string()
}

fn default_source_type() -> String {
    "personal_site".to_string()
}

fn default_profile_url() -> String {
    "https://example.com/profile".to_string()
}
