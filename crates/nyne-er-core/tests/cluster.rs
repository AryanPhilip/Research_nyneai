use anyhow::Result;
use nyne_er_core::{
    PairFeatureExtractor, SplitName, bcubed_f1, build_examples_for_profiles, build_pair_examples,
    build_split_candidates, load_dataset, profiles_for_split, resolve_identities,
    train_hybrid_matcher,
};

fn cluster_inputs() -> Result<(
    Vec<nyne_er_core::ProfileRecord>,
    PairFeatureExtractor,
    nyne_er_core::TrainedHybridMatcher,
    Vec<nyne_er_core::PairExample>,
)> {
    let profiles = load_dataset("hard_negative_bank")?.profiles;
    let split_map = nyne_er_core::assign_profile_splits(&profiles);
    let extractor = PairFeatureExtractor::default().fit(&profiles_for_split(&profiles, &split_map, SplitName::Train))?;
    let train_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Train)?,
        &extractor,
        &split_map,
        SplitName::Train,
    )?;
    let val_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Val)?,
        &extractor,
        &split_map,
        SplitName::Val,
    )?;
    let matcher = train_hybrid_matcher(&train_examples, &val_examples, &extractor, None)?;
    let all_examples = build_examples_for_profiles(&profiles, &extractor, SplitName::Test)?;
    Ok((profiles, extractor, matcher, all_examples))
}

#[test]
fn test_cluster_resolution_and_metrics() -> Result<()> {
    let (profiles, extractor, matcher, all_examples) = cluster_inputs()?;
    let lookup = profiles
        .iter()
        .cloned()
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect::<std::collections::BTreeMap<_, _>>();
    let positive = all_examples
        .iter()
        .find(|example| example.left_profile_id == "andrej_github" && example.right_profile_id == "andrej_personal")
        .unwrap();
    let card = nyne_er_core::generate_evidence_card(
        lookup.get(&positive.left_profile_id).unwrap(),
        lookup.get(&positive.right_profile_id).unwrap(),
        positive,
        0.98,
        "match",
    )?;
    assert!(!card.supporting_signals.is_empty());
    assert!(card.reason_codes.contains(&"high_name_similarity".to_string()));
    assert!(card.final_explanation.contains("Decision=match"));

    let (identities, resolved_pairs) = resolve_identities(&profiles, &all_examples, &matcher, &extractor)?;
    let mut identity_by_profile = std::collections::BTreeMap::new();
    for identity in &identities {
        for profile_id in &identity.member_profile_ids {
            identity_by_profile.insert(profile_id.clone(), identity.entity_id.clone());
        }
    }
    assert_eq!(
        identity_by_profile["sebastian_github"],
        identity_by_profile["sebastian_personal"]
    );
    assert_ne!(
        identity_by_profile["sebastian_github"],
        identity_by_profile["sebastian_raschka_finance"]
    );
    assert!(bcubed_f1(&profiles, &identities) >= 0.8);
    assert!(!resolved_pairs.is_empty());
    assert!(identities.iter().all(|identity| {
        matches!(
            identity.confidence_band,
            nyne_er_core::ConfidenceBand::High
                | nyne_er_core::ConfidenceBand::Medium
                | nyne_er_core::ConfidenceBand::Low
                | nyne_er_core::ConfidenceBand::Uncertain
        )
    }));
    Ok(())
}
