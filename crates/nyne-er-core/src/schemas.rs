use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use std::collections::BTreeMap;
use url::Url;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Github,
    PersonalSite,
    ConferenceBio,
    CompanyProfile,
    PodcastGuest,
    Huggingface,
}

impl SourceType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Github => "github",
            Self::PersonalSite => "personal_site",
            Self::ConferenceBio => "conference_bio",
            Self::CompanyProfile => "company_profile",
            Self::PodcastGuest => "podcast_guest",
            Self::Huggingface => "huggingface",
        }
    }
}

impl fmt::Display for SourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SourceType {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "github" => Ok(Self::Github),
            "personal_site" => Ok(Self::PersonalSite),
            "conference_bio" => Ok(Self::ConferenceBio),
            "company_profile" => Ok(Self::CompanyProfile),
            "podcast_guest" => Ok(Self::PodcastGuest),
            "huggingface" => Ok(Self::Huggingface),
            _ => bail!("unsupported source_type '{value}'"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    Match,
    NonMatch,
    Abstain,
}

impl DecisionType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::NonMatch => "non_match",
            Self::Abstain => "abstain",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceBand {
    High,
    Medium,
    Low,
    Uncertain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextSpan {
    pub field_name: String,
    pub snippet: String,
    pub start_char: Option<usize>,
    pub end_char: Option<usize>,
}

impl TextSpan {
    pub fn validated(mut self) -> Result<Self> {
        self.field_name = self.field_name.trim().to_string();
        self.snippet = self.snippet.trim().to_string();
        if self.field_name.is_empty() {
            bail!("field_name must not be empty");
        }
        if self.snippet.is_empty() {
            bail!("snippet must not be empty");
        }
        if let (Some(start), Some(end)) = (self.start_char, self.end_char) {
            if end < start {
                bail!("end_char must be >= start_char");
            }
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrganizationClaim {
    pub name: String,
    pub role: Option<String>,
    pub start_year: Option<i32>,
    pub end_year: Option<i32>,
}

impl OrganizationClaim {
    pub fn validated(mut self) -> Result<Self> {
        self.name = self.name.trim().to_string();
        self.role = trim_opt(self.role);
        if self.name.is_empty() {
            bail!("organization name must not be empty");
        }
        validate_year(self.start_year)?;
        validate_year(self.end_year)?;
        if let (Some(start), Some(end)) = (self.start_year, self.end_year) {
            if end < start {
                bail!("organization end_year must be >= start_year");
            }
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EducationClaim {
    pub institution: String,
    pub degree: Option<String>,
    pub graduation_year: Option<i32>,
}

impl EducationClaim {
    pub fn validated(mut self) -> Result<Self> {
        self.institution = self.institution.trim().to_string();
        self.degree = trim_opt(self.degree);
        if self.institution.is_empty() {
            bail!("institution must not be empty");
        }
        validate_year(self.graduation_year)?;
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawProfilePage {
    pub page_id: String,
    pub source_type: SourceType,
    pub url: Url,
    pub canonical_person_id: Option<String>,
    pub display_name_hint: Option<String>,
    pub fetched_at: Option<DateTime<Utc>>,
    pub title: Option<String>,
    pub html: Option<String>,
    pub raw_text: String,
}

impl RawProfilePage {
    pub fn validated(mut self) -> Result<Self> {
        self.page_id = self.page_id.trim().to_string();
        self.canonical_person_id = normalize_id_opt(self.canonical_person_id)?;
        self.display_name_hint = trim_opt(self.display_name_hint);
        self.title = trim_opt(self.title);
        self.raw_text = self.raw_text.trim().to_string();
        validate_id(&self.page_id, "page_id")?;
        if self.raw_text.is_empty() {
            bail!("raw_text must not be empty");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileRecord {
    pub profile_id: String,
    pub canonical_person_id: Option<String>,
    pub source_type: SourceType,
    pub url: Url,
    pub display_name: String,
    pub aliases: Vec<String>,
    pub headline: Option<String>,
    pub bio_text: String,
    pub organizations: Vec<OrganizationClaim>,
    pub education: Vec<EducationClaim>,
    pub locations: Vec<String>,
    pub outbound_links: Vec<Url>,
    pub topics: Vec<String>,
    pub timestamps: Vec<String>,
    pub raw_text: String,
    pub supporting_spans: Vec<TextSpan>,
    pub metadata: BTreeMap<String, String>,
}

impl ProfileRecord {
    pub fn validated(mut self) -> Result<Self> {
        self.profile_id = self.profile_id.trim().to_string();
        self.canonical_person_id = normalize_id_opt(self.canonical_person_id)?;
        self.display_name = collapse_ws(&self.display_name);
        self.aliases = normalize_str_list(self.aliases);
        self.headline = trim_opt(self.headline);
        self.bio_text = self.bio_text.trim().to_string();
        self.organizations = self
            .organizations
            .into_iter()
            .map(OrganizationClaim::validated)
            .collect::<Result<Vec<_>>>()?;
        self.education = self
            .education
            .into_iter()
            .map(EducationClaim::validated)
            .collect::<Result<Vec<_>>>()?;
        self.locations = normalize_str_list(self.locations);
        self.topics = normalize_str_list(self.topics);
        self.timestamps = normalize_str_list(self.timestamps);
        self.raw_text = self.raw_text.trim().to_string();
        self.supporting_spans = self
            .supporting_spans
            .into_iter()
            .map(TextSpan::validated)
            .collect::<Result<Vec<_>>>()?;
        self.metadata = self
            .metadata
            .into_iter()
            .map(|(k, v)| Ok((k.trim().to_string(), v.trim().to_string())))
            .collect::<Result<BTreeMap<_, _>>>()?;

        validate_id(&self.profile_id, "profile_id")?;
        if self.display_name.is_empty() {
            bail!("display_name must not be empty");
        }
        if self.bio_text.is_empty() {
            bail!("bio_text must not be empty");
        }
        if self.raw_text.is_empty() {
            bail!("raw_text must not be empty");
        }
        if self.supporting_spans.is_empty() {
            bail!("supporting_spans must contain at least one item");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSignal {
    pub signal_type: String,
    pub description: String,
    pub weight: f64,
    pub source_profile_id: String,
    pub target_profile_id: String,
    pub supporting_span: Option<TextSpan>,
}

impl EvidenceSignal {
    pub fn validated(mut self) -> Result<Self> {
        self.signal_type = self.signal_type.trim().to_string();
        self.description = self.description.trim().to_string();
        if self.signal_type.is_empty() || self.description.is_empty() {
            bail!("evidence signal text must not be empty");
        }
        if !(0.0..=1.0).contains(&self.weight) {
            bail!("evidence signal weight must be between 0 and 1");
        }
        validate_id(&self.source_profile_id, "source_profile_id")?;
        validate_id(&self.target_profile_id, "target_profile_id")?;
        self.supporting_span = self.supporting_span.map(TextSpan::validated).transpose()?;
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePair {
    pub left_profile_id: String,
    pub right_profile_id: String,
    pub blocking_reasons: Vec<String>,
    pub feature_vector: BTreeMap<String, f64>,
    pub match_score: f64,
    pub calibrated_confidence: f64,
    pub decision: DecisionType,
}

impl CandidatePair {
    pub fn validated(self) -> Result<Self> {
        validate_pair_ids(&self.left_profile_id, &self.right_profile_id)?;
        validate_probability(self.match_score, "match_score")?;
        validate_probability(self.calibrated_confidence, "calibrated_confidence")?;
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCard {
    pub left_profile_id: String,
    pub right_profile_id: String,
    pub supporting_signals: Vec<EvidenceSignal>,
    pub contradicting_signals: Vec<EvidenceSignal>,
    pub reason_codes: Vec<String>,
    pub final_explanation: String,
}

impl EvidenceCard {
    pub fn validated(mut self) -> Result<Self> {
        validate_pair_ids(&self.left_profile_id, &self.right_profile_id)?;
        self.supporting_signals = self
            .supporting_signals
            .into_iter()
            .map(EvidenceSignal::validated)
            .collect::<Result<Vec<_>>>()?;
        self.contradicting_signals = self
            .contradicting_signals
            .into_iter()
            .map(EvidenceSignal::validated)
            .collect::<Result<Vec<_>>>()?;
        self.final_explanation = self.final_explanation.trim().to_string();
        if self.final_explanation.is_empty() {
            bail!("final_explanation must not be empty");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalIdentity {
    pub entity_id: String,
    pub member_profile_ids: Vec<String>,
    pub canonical_name: String,
    pub summary: String,
    pub confidence_band: ConfidenceBand,
    pub key_orgs: Vec<String>,
    pub key_links: Vec<Url>,
}

impl CanonicalIdentity {
    pub fn validated(mut self) -> Result<Self> {
        validate_id(&self.entity_id, "entity_id")?;
        self.member_profile_ids = normalize_str_list(self.member_profile_ids);
        if self.member_profile_ids.is_empty() {
            bail!("member_profile_ids must not be empty");
        }
        self.canonical_name = self.canonical_name.trim().to_string();
        self.summary = self.summary.trim().to_string();
        self.key_orgs = normalize_str_list(self.key_orgs);
        if self.canonical_name.is_empty() || self.summary.is_empty() {
            bail!("canonical_name and summary must not be empty");
        }
        Ok(self)
    }
}

fn trim_opt(value: Option<String>) -> Option<String> {
    value.and_then(|item| {
        let trimmed = item.trim().to_string();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}

fn normalize_str_list(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| {
            let trimmed = value.trim().to_string();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .collect()
}

fn normalize_id_opt(value: Option<String>) -> Result<Option<String>> {
    match value {
        Some(id) => {
            let normalized = id.trim().to_string();
            validate_id(&normalized, "id")?;
            Ok(Some(normalized))
        }
        None => Ok(None),
    }
}

fn collapse_ws(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn validate_probability(value: f64, field: &str) -> Result<()> {
    if !(0.0..=1.0).contains(&value) || value.is_nan() {
        return Err(anyhow!("{field} must be between 0 and 1"));
    }
    Ok(())
}

fn validate_pair_ids(left: &str, right: &str) -> Result<()> {
    validate_id(left, "left_profile_id")?;
    validate_id(right, "right_profile_id")?;
    Ok(())
}

fn validate_year(value: Option<i32>) -> Result<()> {
    if let Some(year) = value {
        if !(1950..=2100).contains(&year) {
            bail!("year {year} must be between 1950 and 2100");
        }
    }
    Ok(())
}

fn validate_id(value: &str, field: &str) -> Result<()> {
    if value.is_empty() || !value.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-') {
        bail!("{field} must match ^[a-z0-9_-]+$");
    }
    Ok(())
}
