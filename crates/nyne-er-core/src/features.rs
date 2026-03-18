use crate::blocking::{generate_block_candidates, BlockCandidate};
use crate::ingest::compose_normalized_text;
use crate::schemas::ProfileRecord;
use crate::similarity::{normalize_text, sequence_similarity};
use crate::tfidf::{cosine_similarity, TfidfVectorizer};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitName {
    Train,
    Val,
    Test,
}

impl SplitName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairExample {
    pub left_profile_id: String,
    pub right_profile_id: String,
    pub left_canonical_id: String,
    pub right_canonical_id: String,
    pub split: SplitName,
    pub label: i32,
    pub features: BTreeMap<String, f64>,
    pub blocking_reasons: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct PairFeatureExtractor {
    pub vectorizer: Option<TfidfVectorizer>,
    pub profile_vectors: BTreeMap<String, Vec<f64>>,
    pub fit_profile_ids: BTreeSet<String>,
}

pub const FEATURE_ORDER: [&str; 15] = [
    "name_similarity",
    "alias_similarity",
    "headline_similarity",
    "bio_similarity",
    "shared_domain_count",
    "org_overlap_count",
    "org_jaccard",
    "topic_overlap_count",
    "topic_jaccard",
    "location_overlap",
    "temporal_overlap_count",
    "temporal_distance",
    "location_conflict",
    "same_source_type",
    "embedding_cosine",
];

impl PairFeatureExtractor {
    pub fn fit(mut self, profiles: &[ProfileRecord]) -> Result<Self> {
        let texts: Vec<String> = profiles.iter().map(compose_normalized_text).collect();
        let (vectorizer, matrix) = TfidfVectorizer::fit_transform(&texts);
        self.profile_vectors = profiles
            .iter()
            .zip(matrix.into_iter())
            .map(|(profile, vector)| (profile.profile_id.clone(), vector))
            .collect();
        self.fit_profile_ids = profiles.iter().map(|profile| profile.profile_id.clone()).collect();
        self.vectorizer = Some(vectorizer);
        Ok(self)
    }

    pub fn featurize_pair(&self, left: &ProfileRecord, right: &ProfileRecord) -> BTreeMap<String, f64> {
        let left_aliases = std::iter::once(left.display_name.clone())
            .chain(left.aliases.clone())
            .collect::<Vec<_>>();
        let right_aliases = std::iter::once(right.display_name.clone())
            .chain(right.aliases.clone())
            .collect::<Vec<_>>();
        let left_orgs = token_set(org_names(left));
        let right_orgs = token_set(org_names(right));
        let left_topics = token_set(topic_values(left));
        let right_topics = token_set(topic_values(right));
        let left_locations = token_set(left.locations.clone());
        let right_locations = token_set(right.locations.clone());
        let left_years = year_values(left);
        let right_years = year_values(right);

        let temporal_distance = if left_years.is_empty() || right_years.is_empty() {
            0.0
        } else {
            ((*left_years.iter().min().unwrap() - *right_years.iter().min().unwrap()).abs()) as f64
        };

        BTreeMap::from([
            ("name_similarity".to_string(), sequence_similarity(&left.display_name, &right.display_name)),
            (
                "alias_similarity".to_string(),
                left_aliases
                    .iter()
                    .flat_map(|left_alias| {
                        right_aliases
                            .iter()
                            .map(move |right_alias| sequence_similarity(left_alias, right_alias))
                    })
                    .fold(0.0, f64::max),
            ),
            (
                "headline_similarity".to_string(),
                sequence_similarity(left.headline.as_deref().unwrap_or(""), right.headline.as_deref().unwrap_or("")),
            ),
            ("bio_similarity".to_string(), sequence_similarity(&left.bio_text, &right.bio_text)),
            ("shared_domain_count".to_string(), domains(left).intersection(&domains(right)).count() as f64),
            ("org_overlap_count".to_string(), left_orgs.intersection(&right_orgs).count() as f64),
            ("org_jaccard".to_string(), jaccard(&left_orgs, &right_orgs)),
            ("topic_overlap_count".to_string(), left_topics.intersection(&right_topics).count() as f64),
            ("topic_jaccard".to_string(), jaccard(&left_topics, &right_topics)),
            (
                "location_overlap".to_string(),
                (!left_locations.is_disjoint(&right_locations)) as i32 as f64,
            ),
            (
                "temporal_overlap_count".to_string(),
                left_years.intersection(&right_years).count() as f64,
            ),
            ("temporal_distance".to_string(), temporal_distance),
            (
                "location_conflict".to_string(),
                (!left_locations.is_empty() && !right_locations.is_empty() && left_locations.is_disjoint(&right_locations))
                    as i32 as f64,
            ),
            ("same_source_type".to_string(), (left.source_type == right.source_type) as i32 as f64),
            ("embedding_cosine".to_string(), self.embedding_cosine(left, right)),
        ])
    }

    pub fn vectorize_features(
        &self,
        feature_rows: &[BTreeMap<String, f64>],
        feature_names: Option<&[&str]>,
        include_embedding: bool,
    ) -> Vec<Vec<f64>> {
        let mut order = feature_names
            .map(|items| items.to_vec())
            .unwrap_or_else(|| FEATURE_ORDER.to_vec());
        if !include_embedding {
            order.retain(|feature_name| *feature_name != "embedding_cosine");
        }
        feature_rows
            .iter()
            .map(|row| order.iter().map(|feature_name| *row.get(*feature_name).unwrap_or(&0.0)).collect())
            .collect()
    }

    fn embedding_cosine(&self, left: &ProfileRecord, right: &ProfileRecord) -> f64 {
        let Some(left_vec) = self.profile_vector(left) else {
            return 0.0;
        };
        let Some(right_vec) = self.profile_vector(right) else {
            return 0.0;
        };
        cosine_similarity(&left_vec, &right_vec)
    }

    fn profile_vector(&self, profile: &ProfileRecord) -> Option<Vec<f64>> {
        self.profile_vectors
            .get(&profile.profile_id)
            .cloned()
            .or_else(|| self.transformed_vector(profile))
    }

    pub fn transformed_vector(&self, profile: &ProfileRecord) -> Option<Vec<f64>> {
        self.vectorizer.as_ref().map(|vectorizer| vectorizer.transform(&compose_normalized_text(profile)))
    }
}

pub fn assign_profile_splits(profiles: &[ProfileRecord]) -> BTreeMap<String, SplitName> {
    let canonical_ids: Vec<String> = profiles
        .iter()
        .filter_map(|profile| profile.canonical_person_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let total = canonical_ids.len();
    let train_cut = ((total as f64 * 0.56).round() as usize).max(1);
    let val_cut = (((total as f64) * 0.78).round() as usize).max(train_cut + 1);
    canonical_ids
        .into_iter()
        .enumerate()
        .map(|(index, canonical_id)| {
            let split = if index < train_cut {
                SplitName::Train
            } else if index < val_cut {
                SplitName::Val
            } else {
                SplitName::Test
            };
            (canonical_id, split)
        })
        .collect()
}

pub fn profiles_for_split(
    profiles: &[ProfileRecord],
    split_map: &BTreeMap<String, SplitName>,
    split: SplitName,
) -> Vec<ProfileRecord> {
    profiles
        .iter()
        .filter(|profile| {
            profile
                .canonical_person_id
                .as_ref()
                .and_then(|canonical_id| split_map.get(canonical_id))
                .copied()
                == Some(split)
        })
        .cloned()
        .collect()
}

pub fn build_split_candidates(
    profiles: &[ProfileRecord],
    split_map: &BTreeMap<String, SplitName>,
    split: SplitName,
) -> Result<Vec<BlockCandidate>> {
    generate_block_candidates(&profiles_for_split(profiles, split_map, split), 3)
}

pub fn build_pair_examples(
    profiles: &[ProfileRecord],
    candidates: &[BlockCandidate],
    extractor: &PairFeatureExtractor,
    split_map: &BTreeMap<String, SplitName>,
    split: SplitName,
) -> Result<Vec<PairExample>> {
    let lookup: BTreeMap<String, ProfileRecord> = profiles_for_split(profiles, split_map, split)
        .into_iter()
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect();
    let mut examples = Vec::new();
    for candidate in candidates {
        let left = lookup.get(&candidate.left_profile_id).expect("candidate left exists");
        let right = lookup.get(&candidate.right_profile_id).expect("candidate right exists");
        examples.push(PairExample {
            left_profile_id: left.profile_id.clone(),
            right_profile_id: right.profile_id.clone(),
            left_canonical_id: left.canonical_person_id.clone().unwrap_or_else(|| "unknown".to_string()),
            right_canonical_id: right.canonical_person_id.clone().unwrap_or_else(|| "unknown".to_string()),
            split,
            label: (left.canonical_person_id == right.canonical_person_id) as i32,
            features: extractor.featurize_pair(left, right),
            blocking_reasons: candidate.reasons.clone(),
        });
    }
    Ok(examples)
}

pub fn build_examples_for_profiles(
    profiles: &[ProfileRecord],
    extractor: &PairFeatureExtractor,
    split: SplitName,
) -> Result<Vec<PairExample>> {
    let lookup: BTreeMap<String, ProfileRecord> = profiles
        .iter()
        .cloned()
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect();
    let candidates = generate_block_candidates(profiles, 3)?;
    let mut examples = Vec::new();
    for candidate in candidates {
        let left = lookup.get(&candidate.left_profile_id).expect("candidate left exists");
        let right = lookup.get(&candidate.right_profile_id).expect("candidate right exists");
        examples.push(PairExample {
            left_profile_id: left.profile_id.clone(),
            right_profile_id: right.profile_id.clone(),
            left_canonical_id: left.canonical_person_id.clone().unwrap_or_else(|| "unknown".to_string()),
            right_canonical_id: right.canonical_person_id.clone().unwrap_or_else(|| "unknown".to_string()),
            split,
            label: (left.canonical_person_id == right.canonical_person_id) as i32,
            features: extractor.featurize_pair(left, right),
            blocking_reasons: candidate.reasons,
        });
    }
    Ok(examples)
}

pub fn summarize_split_assignments(split_map: &BTreeMap<String, SplitName>) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for split in split_map.values() {
        *counts.entry(split.as_str().to_string()).or_insert(0) += 1;
    }
    counts
}

pub fn assert_person_disjoint(profiles: &[ProfileRecord], split_map: &BTreeMap<String, SplitName>) -> Result<()> {
    let mut seen: BTreeMap<String, SplitName> = BTreeMap::new();
    for profile in profiles {
        let Some(canonical_id) = &profile.canonical_person_id else {
            continue;
        };
        let split = split_map
            .get(canonical_id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing split for canonical id {canonical_id}"))?;
        if let Some(previous) = seen.insert(canonical_id.clone(), split) {
            if previous != split {
                anyhow::bail!("Canonical id {canonical_id} appears in multiple splits");
            }
        }
    }
    Ok(())
}

fn token_set(values: Vec<String>) -> BTreeSet<String> {
    values
        .into_iter()
        .flat_map(|value| {
            normalize_text(&value)
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 0.0;
    }
    let union = left.union(right).count();
    if union == 0 {
        0.0
    } else {
        left.intersection(right).count() as f64 / union as f64
    }
}

fn domains(profile: &ProfileRecord) -> BTreeSet<String> {
    let mut urls = profile.outbound_links.clone();
    if matches!(profile.source_type, crate::schemas::SourceType::PersonalSite | crate::schemas::SourceType::Huggingface) {
        urls.push(profile.url.clone());
    }
    urls.into_iter()
        .filter_map(|url| Some(url.host_str()?.replace("www.", "")))
        .collect()
}

fn org_names(profile: &ProfileRecord) -> Vec<String> {
    profile.organizations.iter().map(|org| org.name.clone()).collect()
}

fn topic_values(profile: &ProfileRecord) -> Vec<String> {
    let mut values = profile.topics.clone();
    if let Some(headline) = &profile.headline {
        values.push(headline.clone());
    }
    values
}

fn year_values(profile: &ProfileRecord) -> BTreeSet<i32> {
    let mut values = BTreeSet::new();
    for timestamp in &profile.timestamps {
        if let Some(prefix) = timestamp.get(..4) {
            if let Ok(year) = prefix.parse::<i32>() {
                values.insert(year);
            }
        }
    }
    for org in &profile.organizations {
        if let Some(start_year) = org.start_year {
            values.insert(start_year);
        }
        if let Some(end_year) = org.end_year {
            values.insert(end_year);
        }
    }
    values
}
