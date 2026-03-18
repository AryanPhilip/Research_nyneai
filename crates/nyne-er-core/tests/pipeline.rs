use anyhow::Result;
use nyne_er_core::{
    assign_profile_splits, bcubed_f1, blocking_recall, build_examples_for_profiles,
    build_pair_examples, build_split_candidates, candidate_volume_ratio, generate_block_candidates,
    gold_positive_pairs, load_benchmark_profiles, load_dataset, load_raw_pages, parse_raw_pages,
    profiles_for_split, run_benchmark, run_feature_ablations, run_hybrid_matcher,
    run_lexical_baseline, train_hybrid_matcher, PairFeatureExtractor, SplitName,
};

#[test]
fn parses_raw_pages_and_validates_contracts() -> Result<()> {
    let pages = load_raw_pages(None)?;
    let profiles = parse_raw_pages(&pages)?;
    assert_eq!(profiles.len(), pages.len());
    assert!(profiles.iter().all(|profile| !profile.supporting_spans.is_empty()));
    Ok(())
}

#[test]
fn blocker_meets_recall_and_volume_targets() -> Result<()> {
    let profiles = parse_raw_pages(&load_raw_pages(None)?)?;
    let candidates = generate_block_candidates(&profiles, 3)?;
    let recall = blocking_recall(&profiles, &candidates);
    let volume_ratio = candidate_volume_ratio(&profiles, &candidates);
    assert!(recall >= 0.95, "blocking recall was {recall}");
    assert!(volume_ratio < 0.75, "candidate volume ratio was {volume_ratio}");
    assert_eq!(gold_positive_pairs(&profiles).len(), 9);
    Ok(())
}

#[test]
fn hybrid_pipeline_hits_quality_floors() -> Result<()> {
    let profiles = load_benchmark_profiles("real_curated_core")?;
    let split_map = assign_profile_splits(&profiles);
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
    let test_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Test)?,
        &extractor,
        &split_map,
        SplitName::Test,
    )?;

    let lexical = run_lexical_baseline(&train_examples, &val_examples, &test_examples, &extractor)?;
    let hybrid = run_hybrid_matcher(&train_examples, &val_examples, &test_examples, &extractor, Some(42), None)?;
    let ablations = run_feature_ablations(&train_examples, &val_examples, &test_examples, &extractor)?;
    let full_ablation = ablations.iter().find(|item| item.name == "full").unwrap();

    assert!(lexical.metrics.f1 >= 0.65, "lexical f1 was {}", lexical.metrics.f1);
    assert!(
        lexical.metrics.average_precision >= 0.7,
        "lexical AP was {}",
        lexical.metrics.average_precision
    );
    assert!(hybrid.calibrated_metrics.f1 >= 0.85, "hybrid f1 was {}", hybrid.calibrated_metrics.f1);
    assert!(hybrid.calibrated_brier <= hybrid.raw_brier + 1e-9);
    assert!(full_ablation.metrics.f1 >= 0.85);
    Ok(())
}

#[test]
fn benchmark_report_and_clustering_hold_up() -> Result<()> {
    let report = run_benchmark("real_curated_core", "grouped_cv", Some(vec![7, 11]))?;
    assert_eq!(report.dataset_name, "real_curated_core");
    assert_eq!(report.protocol, "grouped_cv");
    assert_eq!(report.model_metrics.len(), 4);
    assert_eq!(report.cv_summary.len(), 2);
    assert!(report.leakage_checks.iter().all(|item| item["passed"].as_bool().unwrap_or(false)));
    assert!(report.failure_slices.len() >= 1);
    assert!(report.stress_metrics["hard_negative_bank"]["f1"].as_f64().unwrap_or(0.0) >= 0.9);
    assert!(report.open_world_retrieval["queries"].as_u64().unwrap_or(0) >= 1);

    let profiles = load_dataset("hard_negative_bank")?.profiles;
    let split_map = assign_profile_splits(&profiles);
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
    let matcher = train_hybrid_matcher(&train_examples, &val_examples, &extractor, Some(42))?;
    let all_examples = build_examples_for_profiles(&profiles, &extractor, SplitName::Test)?;
    let (identities, _resolved_pairs) = nyne_er_core::resolve_identities(&profiles, &all_examples, &matcher, &extractor)?;
    let bcubed = bcubed_f1(&profiles, &identities);
    assert!(bcubed >= 0.8, "bcubed f1 was {bcubed}");

    let mut identity_by_profile = std::collections::BTreeMap::new();
    for identity in identities {
        for profile_id in identity.member_profile_ids {
            identity_by_profile.insert(profile_id, identity.entity_id.clone());
        }
    }
    assert_eq!(
        identity_by_profile.get("sebastian_github"),
        identity_by_profile.get("sebastian_personal")
    );
    assert_ne!(
        identity_by_profile.get("sebastian_github"),
        identity_by_profile.get("sebastian_raschka_finance")
    );
    Ok(())
}
