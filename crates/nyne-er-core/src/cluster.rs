use crate::features::{PairExample, PairFeatureExtractor};
use crate::models::TrainedHybridMatcher;
use crate::schemas::{
    CanonicalIdentity, ConfidenceBand, EvidenceCard, EvidenceSignal, ProfileRecord, TextSpan,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedPair {
    pub left_profile_id: String,
    pub right_profile_id: String,
    pub score: f64,
    pub decision: String,
    pub evidence_card: EvidenceCard,
}

pub fn generate_evidence_card(
    left: &ProfileRecord,
    right: &ProfileRecord,
    example: &PairExample,
    score: f64,
    decision: &str,
) -> Result<EvidenceCard> {
    let mut supporting_signals = Vec::new();
    let mut contradicting_signals = Vec::new();
    let mut reason_codes = Vec::new();
    let mut contradiction_codes = Vec::new();

    if feature(example, "name_similarity") >= 0.95 {
        reason_codes.push("high_name_similarity".to_string());
        supporting_signals.push(
            EvidenceSignal {
                signal_type: "high_name_similarity".to_string(),
                description: format!("Both profiles present the name {}.", left.display_name),
                weight: feature(example, "name_similarity").min(1.0),
                source_profile_id: left.profile_id.clone(),
                target_profile_id: right.profile_id.clone(),
                supporting_span: select_span(left, "headline"),
            }
            .validated()?,
        );
    }
    if feature(example, "shared_domain_count") > 0.0 {
        reason_codes.push("shared_domain".to_string());
        supporting_signals.push(
            EvidenceSignal {
                signal_type: "shared_domain".to_string(),
                description: "Profiles link to the same personal domain.".to_string(),
                weight: 0.8,
                source_profile_id: left.profile_id.clone(),
                target_profile_id: right.profile_id.clone(),
                supporting_span: select_span(left, "bio_text"),
            }
            .validated()?,
        );
    }
    if feature(example, "org_overlap_count") > 0.0 {
        reason_codes.push("shared_org".to_string());
        supporting_signals.push(
            EvidenceSignal {
                signal_type: "shared_org".to_string(),
                description: "Profiles mention overlapping organizations.".to_string(),
                weight: (0.5 + 0.1 * feature(example, "org_overlap_count")).min(1.0),
                source_profile_id: left.profile_id.clone(),
                target_profile_id: right.profile_id.clone(),
                supporting_span: select_span(left, "bio_text"),
            }
            .validated()?,
        );
    }
    if feature(example, "topic_overlap_count") > 0.0 {
        reason_codes.push("shared_topics".to_string());
        supporting_signals.push(
            EvidenceSignal {
                signal_type: "shared_topics".to_string(),
                description: "Profiles share technical topics or headlines.".to_string(),
                weight: (0.4 + 0.1 * feature(example, "topic_overlap_count")).min(1.0),
                source_profile_id: left.profile_id.clone(),
                target_profile_id: right.profile_id.clone(),
                supporting_span: select_span(right, "bio_text"),
            }
            .validated()?,
        );
    }
    if feature(example, "location_conflict") >= 1.0 {
        contradiction_codes.push("location_conflict".to_string());
        contradicting_signals.push(
            EvidenceSignal {
                signal_type: "location_conflict".to_string(),
                description: "Structured locations conflict with one another.".to_string(),
                weight: 0.9,
                source_profile_id: left.profile_id.clone(),
                target_profile_id: right.profile_id.clone(),
                supporting_span: select_span(left, "bio_text"),
            }
            .validated()?,
        );
    }

    let visible_reason_codes = if decision == "match" {
        reason_codes.clone()
    } else {
        reason_codes
            .iter()
            .cloned()
            .chain(contradiction_codes.iter().cloned())
            .collect()
    };
    let mut explanation_parts = vec![
        format!("Decision={decision} at confidence {:.2}.", score),
        format!(
            "Supporting signals: {}.",
            if reason_codes.is_empty() {
                "none".to_string()
            } else {
                reason_codes.join(", ")
            }
        ),
    ];
    if !contradiction_codes.is_empty() {
        explanation_parts.push(format!("Contradictions: {}.", contradiction_codes.join(", ")));
    }

    EvidenceCard {
        left_profile_id: left.profile_id.clone(),
        right_profile_id: right.profile_id.clone(),
        supporting_signals,
        contradicting_signals,
        reason_codes: visible_reason_codes,
        final_explanation: explanation_parts.join(" "),
    }
    .validated()
}

pub fn resolve_identities(
    profiles: &[ProfileRecord],
    examples: &[PairExample],
    matcher: &TrainedHybridMatcher,
    extractor: &PairFeatureExtractor,
) -> Result<(Vec<CanonicalIdentity>, Vec<ResolvedPair>)> {
    let profile_lookup: BTreeMap<String, ProfileRecord> = profiles
        .iter()
        .cloned()
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect();
    let (scores, decisions) = matcher.score_examples(examples, extractor, None);
    let mut resolved_pairs = Vec::new();
    let mut parent: BTreeMap<String, String> = profiles
        .iter()
        .map(|profile| (profile.profile_id.clone(), profile.profile_id.clone()))
        .collect();

    fn find(parent: &mut BTreeMap<String, String>, node: &str) -> String {
        let mut current = node.to_string();
        while parent.get(&current).map(|value| value.as_str()) != Some(current.as_str()) {
            let parent_value = parent.get(&current).cloned().unwrap();
            let grandparent = parent.get(&parent_value).cloned().unwrap();
            parent.insert(current.clone(), grandparent.clone());
            current = grandparent;
        }
        current
    }

    fn union(parent: &mut BTreeMap<String, String>, left_id: &str, right_id: &str) {
        let left_root = find(parent, left_id);
        let right_root = find(parent, right_id);
        if left_root != right_root {
            parent.insert(right_root, left_root);
        }
    }

    for ((example, score), decision) in examples.iter().zip(scores.iter()).zip(decisions.iter()) {
        let left = profile_lookup.get(&example.left_profile_id).expect("left profile exists");
        let right = profile_lookup.get(&example.right_profile_id).expect("right profile exists");
        resolved_pairs.push(ResolvedPair {
            left_profile_id: left.profile_id.clone(),
            right_profile_id: right.profile_id.clone(),
            score: *score,
            decision: decision.clone(),
            evidence_card: generate_evidence_card(left, right, example, *score, decision)?,
        });
        if decision == "match" {
            union(&mut parent, &left.profile_id, &right.profile_id);
        }
    }

    let mut clusters: BTreeMap<String, Vec<ProfileRecord>> = BTreeMap::new();
    let mut edge_scores_by_cluster: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for profile in profiles {
        let root = find(&mut parent, &profile.profile_id);
        clusters.entry(root).or_default().push(profile.clone());
    }
    for pair in &resolved_pairs {
        if pair.decision != "match" {
            continue;
        }
        let root = find(&mut parent, &pair.left_profile_id);
        edge_scores_by_cluster.entry(root).or_default().push(pair.score);
    }

    let mut identities = Vec::new();
    for cluster_profiles in clusters.values() {
        let member_ids = cluster_profiles
            .iter()
            .map(|profile| profile.profile_id.clone())
            .collect::<Vec<_>>();
        let canonical_name = most_common(
            cluster_profiles
                .iter()
                .map(|profile| profile.display_name.clone())
                .collect(),
        )
        .unwrap_or_else(|| cluster_profiles[0].display_name.clone());
        let orgs = counts(
            cluster_profiles
                .iter()
                .flat_map(|profile| profile.organizations.iter().map(|org| org.name.clone()))
                .collect(),
        );
        let summary_topics = counts(
            cluster_profiles
                .iter()
                .flat_map(|profile| profile.topics.clone())
                .collect(),
        );
        let topic_summary = summary_topics
            .into_iter()
            .take(3)
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
            .join(", ");
        let mut key_links = Vec::new();
        let mut seen_links = BTreeSet::new();
        for profile in cluster_profiles {
            for link in &profile.outbound_links {
                let link_string = link.to_string();
                if seen_links.insert(link_string) {
                    key_links.push(link.clone());
                }
            }
        }
        let root = find(&mut parent, &cluster_profiles[0].profile_id);
        identities.push(
            CanonicalIdentity {
                entity_id: root.clone(),
                member_profile_ids: member_ids,
                canonical_name,
                summary: format!(
                    "{} appears across {} profiles with topics {}.",
                    cluster_profiles[0].display_name,
                    cluster_profiles.len(),
                    if topic_summary.is_empty() {
                        "not enough evidence".to_string()
                    } else {
                        topic_summary
                    }
                ),
                confidence_band: confidence_band(
                    edge_scores_by_cluster.get(&root).cloned().unwrap_or_default(),
                ),
                key_orgs: orgs.into_iter().take(3).map(|(name, _)| name).collect(),
                key_links: key_links.into_iter().take(3).collect(),
            }
            .validated()?,
        );
    }

    identities.sort_by(|left, right| {
        right
            .member_profile_ids
            .len()
            .cmp(&left.member_profile_ids.len())
            .then_with(|| left.entity_id.cmp(&right.entity_id))
    });
    Ok((identities, resolved_pairs))
}

pub fn bcubed_f1(profiles: &[ProfileRecord], identities: &[CanonicalIdentity]) -> f64 {
    let mut predicted_by_profile: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for identity in identities {
        let members: BTreeSet<String> = identity.member_profile_ids.iter().cloned().collect();
        for profile_id in &identity.member_profile_ids {
            predicted_by_profile.insert(profile_id.clone(), members.clone());
        }
    }

    let mut gold_by_profile: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for profile in profiles {
        if let Some(canonical_id) = &profile.canonical_person_id {
            gold_by_profile
                .entry(canonical_id.clone())
                .or_default()
                .insert(profile.profile_id.clone());
        }
    }

    let mut precision_scores = Vec::new();
    let mut recall_scores = Vec::new();
    for profile in profiles {
        let predicted = predicted_by_profile
            .get(&profile.profile_id)
            .cloned()
            .unwrap_or_else(|| BTreeSet::from([profile.profile_id.clone()]));
        let gold = profile
            .canonical_person_id
            .as_ref()
            .and_then(|canonical_id| gold_by_profile.get(canonical_id))
            .cloned()
            .unwrap_or_else(|| BTreeSet::from([profile.profile_id.clone()]));
        let overlap = predicted.intersection(&gold).count() as f64;
        precision_scores.push(overlap / predicted.len() as f64);
        recall_scores.push(overlap / gold.len() as f64);
    }
    let precision = precision_scores.iter().sum::<f64>() / precision_scores.len().max(1) as f64;
    let recall = recall_scores.iter().sum::<f64>() / recall_scores.len().max(1) as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn select_span(profile: &ProfileRecord, field_name: &str) -> Option<TextSpan> {
    profile
        .supporting_spans
        .iter()
        .find(|span| span.field_name == field_name)
        .cloned()
        .or_else(|| profile.supporting_spans.first().cloned())
}

fn feature(example: &PairExample, key: &str) -> f64 {
    *example.features.get(key).unwrap_or(&0.0)
}

fn confidence_band(edge_scores: Vec<f64>) -> ConfidenceBand {
    if edge_scores.is_empty() {
        return ConfidenceBand::Uncertain;
    }
    let average = edge_scores.iter().sum::<f64>() / edge_scores.len() as f64;
    if average >= 0.9 {
        ConfidenceBand::High
    } else if average >= 0.75 {
        ConfidenceBand::Medium
    } else if average >= 0.6 {
        ConfidenceBand::Low
    } else {
        ConfidenceBand::Uncertain
    }
}

fn counts(values: Vec<String>) -> Vec<(String, usize)> {
    let mut counts = BTreeMap::<String, usize>::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    let mut items: Vec<(String, usize)> = counts.into_iter().collect();
    items.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    items
}

fn most_common(values: Vec<String>) -> Option<String> {
    counts(values).into_iter().next().map(|(name, _)| name)
}
