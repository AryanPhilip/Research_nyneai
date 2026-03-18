use crate::schemas::{OrganizationClaim, ProfileRecord, RawProfilePage, TextSpan};
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetBundle {
    pub name: String,
    pub description: String,
    pub profiles: Vec<ProfileRecord>,
    pub contains_synthetic: bool,
    pub headline: bool,
}

pub fn load_seed_profiles(path: Option<&Path>) -> Result<Vec<ProfileRecord>> {
    let fixture_path = path.map(PathBuf::from).unwrap_or_else(|| data_dir().join("fixtures/seed_profiles.json"));
    let payload = fs::read_to_string(&fixture_path)
        .with_context(|| format!("failed to read {}", fixture_path.display()))?;
    let profiles: Vec<ProfileRecord> = serde_json::from_str(&payload)?;
    profiles.into_iter().map(ProfileRecord::validated).collect()
}

pub fn load_raw_pages(path: Option<&Path>) -> Result<Vec<RawProfilePage>> {
    let fixture_path = path.map(PathBuf::from).unwrap_or_else(|| data_dir().join("raw/seed_raw_pages.json"));
    let payload = fs::read_to_string(&fixture_path)
        .with_context(|| format!("failed to read {}", fixture_path.display()))?;
    let pages: Vec<RawProfilePage> = serde_json::from_str(&payload)?;
    pages.into_iter().map(RawProfilePage::validated).collect()
}

pub fn load_dataset(name: &str) -> Result<DatasetBundle> {
    let dataset = match name {
        "real_curated_core" => DatasetBundle {
            name: "real_curated_core".to_string(),
            description: "Headline benchmark of curated public AI/ML builder profiles only.".to_string(),
            profiles: stage1_profiles()?,
            contains_synthetic: false,
            headline: true,
        },
        "hard_negative_bank" => {
            let mut profiles = stage1_profiles()?;
            profiles.extend(hard_negative_profiles()?);
            DatasetBundle {
                name: "hard_negative_bank".to_string(),
                description: "Curated same-name and adjacent-domain distractors layered onto the real corpus.".to_string(),
                profiles,
                contains_synthetic: false,
                headline: false,
            }
        }
        "synthetic_stress" => DatasetBundle {
            name: "synthetic_stress".to_string(),
            description: "Robustness-only dataset with generated positive and conflict variants.".to_string(),
            profiles: synthetic_stress_profiles()?,
            contains_synthetic: true,
            headline: false,
        },
        other => bail!("Unknown dataset '{other}'. Available datasets: {}", available_datasets().join(", ")),
    };
    Ok(dataset)
}

pub fn available_datasets() -> Vec<String> {
    vec![
        "real_curated_core".to_string(),
        "hard_negative_bank".to_string(),
        "synthetic_stress".to_string(),
    ]
}

pub fn load_benchmark_profiles(dataset_name: &str) -> Result<Vec<ProfileRecord>> {
    Ok(load_dataset(dataset_name)?.profiles)
}

fn root_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn data_dir() -> PathBuf {
    root_dir().join("data")
}

fn seed_group_profiles(seed_group: &str) -> Result<Vec<ProfileRecord>> {
    Ok(load_seed_profiles(None)?
        .into_iter()
        .filter(|profile| profile.metadata.get("seed_group").map(String::as_str) == Some(seed_group))
        .collect())
}

fn stage1_profiles() -> Result<Vec<ProfileRecord>> {
    seed_group_profiles("stage1")
}

fn hard_negative_profiles() -> Result<Vec<ProfileRecord>> {
    seed_group_profiles("hard_negative")
}

fn synthetic_stress_profiles() -> Result<Vec<ProfileRecord>> {
    let stage1 = stage1_profiles()?;
    let mut grouped: BTreeMap<String, Vec<ProfileRecord>> = BTreeMap::new();
    for profile in &stage1 {
        if let Some(canonical_id) = &profile.canonical_person_id {
            grouped.entry(canonical_id.clone()).or_default().push(profile.clone());
        }
    }

    let mut synthetic_profiles = Vec::new();
    for (index, (canonical_id, profiles)) in grouped.into_iter().enumerate() {
        synthetic_profiles.push(synthetic_positive_profile(&canonical_id, &profiles, index)?);
        synthetic_profiles.push(synthetic_conflict_profile(&canonical_id, &profiles, index)?);
    }

    let mut merged = stage1;
    merged.extend(synthetic_profiles);
    Ok(merged)
}

fn initial_alias(display_name: &str) -> String {
    let parts: Vec<&str> = display_name.split_whitespace().collect();
    if parts.len() < 2 {
        display_name.to_string()
    } else {
        format!("{}. {}", &parts[0][..1], parts[parts.len() - 1])
    }
}

fn synthetic_positive_profile(canonical_id: &str, profiles: &[ProfileRecord], index: usize) -> Result<ProfileRecord> {
    let anchor = profiles.first().context("missing anchor profile")?;
    let topic_slice = if anchor.topics.is_empty() {
        vec!["machine learning".to_string()]
    } else {
        anchor.topics.iter().take(2).cloned().collect()
    };
    let source_type = if index % 2 == 0 {
        crate::schemas::SourceType::Huggingface
    } else {
        crate::schemas::SourceType::CompanyProfile
    };
    let url = format!("https://example.org/benchmark/{canonical_id}/{}", source_type.as_str());
    let headline = format!("Research profile on {}.", topic_slice.join(", "));
    let bio_text = format!(
        "{} works across {} and production-facing AI systems. This synthetic profile is derived from curated public-profile evidence.",
        anchor.display_name,
        topic_slice.join(", ")
    );
    ProfileRecord {
        profile_id: format!("{canonical_id}_augmented"),
        canonical_person_id: Some(canonical_id.to_string()),
        source_type,
        url: url.parse()?,
        display_name: anchor.display_name.clone(),
        aliases: vec![initial_alias(&anchor.display_name)],
        headline: Some(headline),
        bio_text: bio_text.clone(),
        organizations: anchor.organizations.iter().take(1).cloned().collect(),
        education: anchor.education.iter().take(1).cloned().collect(),
        locations: anchor
            .locations
            .iter()
            .take(1)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .chain((anchor.locations.is_empty()).then_some("United States".to_string()))
            .collect(),
        outbound_links: anchor.outbound_links.iter().take(1).cloned().collect(),
        topics: topic_slice
            .into_iter()
            .chain(std::iter::once("ai systems".to_string()))
            .collect(),
        timestamps: if anchor.timestamps.is_empty() {
            vec!["2024".to_string()]
        } else {
            anchor.timestamps.iter().rev().take(2).cloned().collect::<Vec<_>>().into_iter().rev().collect()
        },
        raw_text: bio_text.clone(),
        supporting_spans: vec![TextSpan {
            field_name: "bio_text".to_string(),
            snippet: bio_text.chars().take(180).collect(),
            start_char: None,
            end_char: None,
        }],
        metadata: BTreeMap::from([
            ("seed_group".to_string(), "synthetic_positive".to_string()),
            ("generated_from".to_string(), canonical_id.to_string()),
        ]),
    }
    .validated()
}

fn synthetic_conflict_profile(canonical_id: &str, profiles: &[ProfileRecord], index: usize) -> Result<ProfileRecord> {
    let anchor = profiles.first().context("missing anchor profile")?;
    let conflict_specs = [
        ("Policy Grid", "Research lead", ["ai governance", "regulation", "policy"], "Washington"),
        ("NorthBridge Capital", "Quant analyst", ["portfolio risk", "finance", "quant analytics"], "New York"),
        ("BrightFrame Studio", "Creative director", ["brand systems", "design ops", "media"], "Los Angeles"),
        ("Civic Compute", "Program manager", ["public sector", "compliance", "strategy"], "Chicago"),
        ("Blue River Health", "Data strategist", ["health analytics", "operations", "care systems"], "Boston"),
        ("Harbor Retail", "Analytics lead", ["consumer analytics", "pricing", "retail"], "Seattle"),
    ];
    let (org_name, role, topics, location) = &conflict_specs[index % conflict_specs.len()];
    let bio_text = format!(
        "{} works on {} at {}. This is a hard-negative same-name collision with intentionally conflicting structured evidence.",
        anchor.display_name,
        topics.join(", "),
        org_name
    );
    ProfileRecord {
        profile_id: format!("{canonical_id}_conflict_generated"),
        canonical_person_id: Some(format!("{canonical_id}_conflict_generated")),
        source_type: crate::schemas::SourceType::CompanyProfile,
        url: format!("https://example.org/conflicts/{canonical_id}").parse()?,
        display_name: anchor.display_name.clone(),
        aliases: vec![initial_alias(&anchor.display_name)],
        headline: Some(format!("{role} focused on {}.", topics[0])),
        bio_text: bio_text.clone(),
        organizations: vec![OrganizationClaim {
            name: (*org_name).to_string(),
            role: Some((*role).to_string()),
            start_year: None,
            end_year: None,
        }
        .validated()?],
        education: vec![],
        locations: vec![(*location).to_string()],
        outbound_links: vec![format!("https://example.org/{}", org_name.to_lowercase().replace(' ', "-")).parse()?],
        topics: topics.iter().map(|topic| topic.to_string()).collect(),
        timestamps: vec!["2024".to_string()],
        raw_text: bio_text.clone(),
        supporting_spans: vec![TextSpan {
            field_name: "bio_text".to_string(),
            snippet: bio_text.chars().take(180).collect(),
            start_char: None,
            end_char: None,
        }],
        metadata: BTreeMap::from([
            ("seed_group".to_string(), "synthetic_conflict".to_string()),
            ("generated_from".to_string(), canonical_id.to_string()),
        ]),
    }
    .validated()
}

#[allow(dead_code)]
fn unique_ids(profiles: &[ProfileRecord]) -> BTreeSet<String> {
    profiles.iter().filter_map(|profile| profile.canonical_person_id.clone()).collect()
}
