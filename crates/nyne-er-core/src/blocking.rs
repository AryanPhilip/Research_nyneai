use crate::ingest::compose_normalized_text;
use crate::schemas::ProfileRecord;
use crate::similarity::{normalize_text, sequence_similarity};
use crate::tfidf::{cosine_similarity, TfidfVectorizer};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockCandidate {
    pub left_profile_id: String,
    pub right_profile_id: String,
    pub reasons: Vec<String>,
}

pub fn exact_or_fuzzy_name_match(left: &ProfileRecord, right: &ProfileRecord) -> bool {
    let left_variants = name_variants(left);
    let right_variants = name_variants(right);
    if !left_variants.is_disjoint(&right_variants) {
        return true;
    }
    let similarity = sequence_similarity(&left.display_name, &right.display_name);
    let shared_tokens: BTreeSet<String> = name_tokens(left)
        .intersection(&name_tokens(right))
        .cloned()
        .collect();
    similarity >= 0.85 || shared_tokens.len() >= 2
}

pub fn domain_overlap_match(left: &ProfileRecord, right: &ProfileRecord) -> bool {
    !domains(left).is_disjoint(&domains(right))
}

pub fn org_or_title_overlap_match(left: &ProfileRecord, right: &ProfileRecord) -> bool {
    let org_overlap: BTreeSet<String> = org_tokens(left).intersection(&org_tokens(right)).cloned().collect();
    let topic_overlap: BTreeSet<String> = topic_tokens(left).intersection(&topic_tokens(right)).cloned().collect();
    !org_overlap.is_empty() || topic_overlap.len() >= 2
}

pub fn embedding_neighbor_candidates(
    profiles: &[ProfileRecord],
    top_k: usize,
) -> Result<BTreeMap<(String, String), BTreeSet<String>>> {
    if profiles.len() < 2 {
        return Ok(BTreeMap::new());
    }
    let texts: Vec<String> = profiles.iter().map(compose_normalized_text).collect();
    let (vectorizer, matrix) = TfidfVectorizer::fit_transform(&texts);
    let mut candidates: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for (index, profile) in profiles.iter().enumerate() {
        let mut scores = Vec::new();
        for (other_index, other) in profiles.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let score = cosine_similarity(&matrix[index], &matrix[other_index]);
            scores.push((other_index, other.profile_id.clone(), score));
        }
        scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        for (other_index, _, _) in scores.into_iter().take(top_k.min(profiles.len().saturating_sub(1))) {
            let key = pair_key(&profile.profile_id, &profiles[other_index].profile_id);
            candidates.entry(key).or_default().insert("embedding_neighbor".to_string());
        }
    }
    let _ = vectorizer;
    Ok(candidates)
}

pub fn generate_block_candidates(profiles: &[ProfileRecord], top_k: usize) -> Result<Vec<BlockCandidate>> {
    let mut candidates: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for left_index in 0..profiles.len() {
        for right_index in left_index + 1..profiles.len() {
            let reasons = rule_reasons(&profiles[left_index], &profiles[right_index]);
            if !reasons.is_empty() {
                candidates
                    .entry(pair_key(&profiles[left_index].profile_id, &profiles[right_index].profile_id))
                    .or_default()
                    .extend(reasons);
            }
        }
    }
    for (key, reasons) in embedding_neighbor_candidates(profiles, top_k)? {
        candidates.entry(key).or_default().extend(reasons);
    }
    Ok(candidates
        .into_iter()
        .filter(|(_, reasons)| !reasons.is_empty())
        .map(|((left_profile_id, right_profile_id), reasons)| BlockCandidate {
            left_profile_id,
            right_profile_id,
            reasons: reasons.into_iter().collect(),
        })
        .collect())
}

pub fn gold_positive_pairs(profiles: &[ProfileRecord]) -> BTreeSet<(String, String)> {
    let mut positives = BTreeSet::new();
    for left_index in 0..profiles.len() {
        for right_index in left_index + 1..profiles.len() {
            if same_person_label(&profiles[left_index], &profiles[right_index]) {
                positives.insert(pair_key(
                    &profiles[left_index].profile_id,
                    &profiles[right_index].profile_id,
                ));
            }
        }
    }
    positives
}

pub fn blocking_recall(profiles: &[ProfileRecord], candidates: &[BlockCandidate]) -> f64 {
    let positives = gold_positive_pairs(profiles);
    if positives.is_empty() {
        return 0.0;
    }
    let candidate_keys: BTreeSet<(String, String)> = candidates
        .iter()
        .map(|candidate| pair_key(&candidate.left_profile_id, &candidate.right_profile_id))
        .collect();
    positives.intersection(&candidate_keys).count() as f64 / positives.len() as f64
}

pub fn candidate_volume_ratio(profiles: &[ProfileRecord], candidates: &[BlockCandidate]) -> f64 {
    let total_possible = ((profiles.len() * profiles.len().saturating_sub(1)) / 2).max(1);
    candidates.len() as f64 / total_possible as f64
}

pub fn blocking_rule_stats(examples: &[crate::features::PairExample]) -> BTreeMap<String, serde_json::Value> {
    let mut rule_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut rule_tp: BTreeMap<String, usize> = BTreeMap::new();
    for example in examples {
        for reason in &example.blocking_reasons {
            *rule_counts.entry(reason.clone()).or_insert(0) += 1;
            if example.label == 1 {
                *rule_tp.entry(reason.clone()).or_insert(0) += 1;
            }
        }
    }

    rule_counts
        .into_iter()
        .map(|(rule, count)| {
            let tp = rule_tp.get(&rule).copied().unwrap_or(0);
            (
                rule,
                serde_json::json!({
                    "count": count,
                    "true_positives": tp,
                    "precision": tp as f64 / count.max(1) as f64,
                }),
            )
        })
        .collect()
}

pub(crate) fn rule_reasons(left: &ProfileRecord, right: &ProfileRecord) -> BTreeSet<String> {
    let mut reasons = BTreeSet::new();
    if exact_or_fuzzy_name_match(left, right) {
        reasons.insert("fuzzy_name".to_string());
    }
    if alias_or_initial_match(left, right) {
        reasons.insert("alias_or_initial".to_string());
    }
    if domain_overlap_match(left, right) {
        reasons.insert("shared_domain".to_string());
    }
    if org_or_title_overlap_match(left, right) {
        reasons.insert("org_or_topic_overlap".to_string());
    }
    reasons
}

fn alias_or_initial_match(left: &ProfileRecord, right: &ProfileRecord) -> bool {
    !name_variants(left).is_disjoint(&name_variants(right))
}

fn pair_key(left_profile_id: &str, right_profile_id: &str) -> (String, String) {
    if left_profile_id <= right_profile_id {
        (left_profile_id.to_string(), right_profile_id.to_string())
    } else {
        (right_profile_id.to_string(), left_profile_id.to_string())
    }
}

fn name_tokens(profile: &ProfileRecord) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    for value in std::iter::once(&profile.display_name).chain(profile.aliases.iter()) {
        for token in normalize_text(value).split_whitespace().filter(|token| token.len() > 1) {
            tokens.insert(token.to_string());
        }
    }
    tokens
}

fn name_variants(profile: &ProfileRecord) -> BTreeSet<String> {
    let base_name = normalize_text(&profile.display_name);
    let parts: Vec<&str> = base_name.split_whitespace().collect();
    let mut variants = BTreeSet::from([base_name.clone()]);
    variants.extend(profile.aliases.iter().map(|alias| normalize_text(alias)));
    if parts.len() >= 2 {
        variants.insert(format!("{} {}", parts[0], &parts[parts.len() - 1][..1]));
        variants.insert(format!("{} {}", &parts[0][..1], parts[parts.len() - 1]));
    }
    variants.into_iter().filter(|variant| !variant.trim().is_empty()).collect()
}

fn domains(profile: &ProfileRecord) -> BTreeSet<String> {
    let mut urls: Vec<String> = profile.outbound_links.iter().map(|url| url.as_str().to_string()).collect();
    if matches!(profile.source_type, crate::schemas::SourceType::PersonalSite | crate::schemas::SourceType::Huggingface) {
        urls.push(profile.url.as_str().to_string());
    }
    urls.into_iter()
        .filter_map(|url| {
            let parsed = url.parse::<url::Url>().ok()?;
            Some(parsed.host_str()?.trim_start_matches("www.").to_string())
        })
        .collect()
}

fn org_tokens(profile: &ProfileRecord) -> BTreeSet<String> {
    profile
        .organizations
        .iter()
        .flat_map(|org| normalize_text(&org.name).split_whitespace().map(str::to_string).collect::<Vec<_>>())
        .filter(|token| token.len() > 2)
        .collect()
}

fn topic_tokens(profile: &ProfileRecord) -> BTreeSet<String> {
    let mut parts = profile.topics.clone();
    if let Some(headline) = &profile.headline {
        parts.push(headline.clone());
    }
    parts.into_iter()
        .flat_map(|part| normalize_text(&part).split_whitespace().map(str::to_string).collect::<Vec<_>>())
        .filter(|token| token.len() > 2)
        .collect()
}

fn same_person_label(left: &ProfileRecord, right: &ProfileRecord) -> bool {
    left.canonical_person_id.is_some() && left.canonical_person_id == right.canonical_person_id
}

