use std::collections::BTreeMap;

use anyhow::Result;
use nyne_er_core::{
    blocking_recall, candidate_volume_ratio, domain_overlap_match, embedding_neighbor_candidates,
    exact_or_fuzzy_name_match, generate_block_candidates, gold_positive_pairs,
    load_raw_pages, org_or_title_overlap_match,
};
use nyne_er_core::ingest::parse_raw_pages;

fn profiles() -> Result<Vec<nyne_er_core::ProfileRecord>> {
    parse_raw_pages(&load_raw_pages(None)?)
}

#[test]
fn test_exact_or_fuzzy_name_matching() -> Result<()> {
    let profiles = profiles()?;
    assert!(exact_or_fuzzy_name_match(&profiles[0], &profiles[1]));
    assert!(!exact_or_fuzzy_name_match(&profiles[3], &profiles[9]));
    Ok(())
}

#[test]
fn test_domain_and_org_overlap_matching() -> Result<()> {
    let profiles = profiles()?
        .into_iter()
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect::<BTreeMap<_, _>>();
    assert!(domain_overlap_match(
        profiles.get("andrej_personal").unwrap(),
        profiles.get("andrej_speaker").unwrap(),
    ));
    assert!(!domain_overlap_match(
        profiles.get("chip_personal").unwrap(),
        profiles.get("jason_builder").unwrap(),
    ));
    assert!(org_or_title_overlap_match(
        profiles.get("chip_personal").unwrap(),
        profiles.get("chip_github").unwrap(),
    ));
    assert!(!org_or_title_overlap_match(
        profiles.get("chip_huynh_blog").unwrap(),
        profiles.get("jason_builder").unwrap(),
    ));
    Ok(())
}

#[test]
fn test_blocker_reaches_thresholds_and_emits_reasons() -> Result<()> {
    let profiles = profiles()?;
    let embedding = embedding_neighbor_candidates(&profiles, 2)?;
    assert!(embedding.contains_key(&("jay_github".to_string(), "jay_personal".to_string())));
    assert!(embedding.contains_key(&("andrej_github".to_string(), "andrej_personal".to_string())));

    let candidates = generate_block_candidates(&profiles, 3)?;
    let recall = blocking_recall(&profiles, &candidates);
    let volume_ratio = candidate_volume_ratio(&profiles, &candidates);
    assert!(recall >= 0.95, "blocking recall {recall}");
    assert!(volume_ratio < 0.75, "volume ratio {volume_ratio}");
    assert_eq!(gold_positive_pairs(&profiles).len(), 9);

    let reasons = candidates
        .into_iter()
        .map(|candidate| {
            (
                (candidate.left_profile_id, candidate.right_profile_id),
                candidate.reasons,
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert!(reasons[&("andrej_github".to_string(), "andrej_personal".to_string())]
        .contains(&"shared_domain".to_string()));
    assert!(reasons[&("chip_github".to_string(), "chip_personal".to_string())]
        .contains(&"org_or_topic_overlap".to_string()));
    Ok(())
}
