use nyne_er_core::{
    PairFeatureExtractor, SplitName, assert_person_disjoint, assign_profile_splits, bcubed_f1,
    blocking_recall, build_examples_for_profiles, build_pair_examples, build_split_candidates,
    candidate_volume_ratio, generate_evidence_card, generate_block_candidates, load_benchmark_profiles,
    load_dataset, load_raw_pages, parse_raw_pages, profiles_for_split, resolve_identities,
    run_benchmark, run_feature_ablations, run_hybrid_matcher, run_lexical_baseline, run_name_baseline,
    train_hybrid_matcher,
};

#[test]
fn parses_seed_raw_pages() {
    let pages = load_raw_pages(None).expect("raw pages load");
    let profiles = parse_raw_pages(&pages).expect("pages parse");
    assert!(!profiles.is_empty());
    assert_eq!(profiles.len(), pages.len());
}

#[test]
fn blocker_reaches_expected_quality() {
    let profiles = parse_raw_pages(&load_raw_pages(None).unwrap()).unwrap();
    let candidates = generate_block_candidates(&profiles, 3).unwrap();
    let recall = blocking_recall(&profiles, &candidates);
    let volume_ratio = candidate_volume_ratio(&profiles, &candidates);

    assert!(recall >= 0.95, "blocking recall was {recall}");
    assert!(volume_ratio < 0.75, "volume ratio was {volume_ratio}");
}

#[test]
fn feature_and_baseline_pipeline_runs() {
    let profiles = load_benchmark_profiles("real_curated_core").unwrap();
    let split_map = assign_profile_splits(&profiles);
    assert_person_disjoint(&profiles, &split_map).unwrap();

    let train_profiles = profiles_for_split(&profiles, &split_map, SplitName::Train);
    let val_profiles = profiles_for_split(&profiles, &split_map, SplitName::Val);
    let test_profiles = profiles_for_split(&profiles, &split_map, SplitName::Test);
    let extractor = PairFeatureExtractor::default().fit(&train_profiles).unwrap();

    let train_examples = build_examples_for_profiles(&train_profiles, &extractor, SplitName::Train).unwrap();
    let val_examples = build_examples_for_profiles(&val_profiles, &extractor, SplitName::Val).unwrap();
    let test_examples = build_examples_for_profiles(&test_profiles, &extractor, SplitName::Test).unwrap();

    let name_run = run_name_baseline(&val_examples, &test_examples);
    let lexical_run = run_lexical_baseline(&train_examples, &val_examples, &test_examples, &extractor).unwrap();

    assert!(extractor.fit_profile_ids.len() == train_profiles.len());
    assert!(name_run.metrics.f1 >= 0.0);
    assert!(lexical_run.metrics.f1 >= 0.65, "lexical baseline f1 was {}", lexical_run.metrics.f1);
    assert!(lexical_run.metrics.average_precision >= 0.7);
}

#[test]
fn hybrid_and_ablations_hit_expected_floor() {
    let profiles = load_benchmark_profiles("real_curated_core").unwrap();
    let split_map = assign_profile_splits(&profiles);
    let extractor = PairFeatureExtractor::default()
        .fit(&profiles_for_split(&profiles, &split_map, SplitName::Train))
        .unwrap();

    let train_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Train).unwrap(),
        &extractor,
        &split_map,
        SplitName::Train,
    )
    .unwrap();
    let val_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Val).unwrap(),
        &extractor,
        &split_map,
        SplitName::Val,
    )
    .unwrap();
    let test_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Test).unwrap(),
        &extractor,
        &split_map,
        SplitName::Test,
    )
    .unwrap();

    let run = run_hybrid_matcher(&train_examples, &val_examples, &test_examples, &extractor, Some(7), None).unwrap();
    let ablations = run_feature_ablations(&train_examples, &val_examples, &test_examples, &extractor).unwrap();
    let full = ablations.iter().find(|item| item.name == "full").unwrap();

    assert!(run.calibrated_brier <= run.raw_brier + 1e-9);
    assert!(run.calibrated_metrics.f1 >= 0.85, "hybrid f1 was {}", run.calibrated_metrics.f1);
    assert!(run.abstain_rate > 0.0);
    assert!(full.metrics.f1 >= 0.85);
}

#[test]
fn clustering_and_benchmark_report_hold_shape() {
    let profiles = load_dataset("hard_negative_bank").unwrap().profiles;
    let split_map = assign_profile_splits(&profiles);
    let extractor = PairFeatureExtractor::default()
        .fit(&profiles_for_split(&profiles, &split_map, SplitName::Train))
        .unwrap();
    let train_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Train).unwrap(),
        &extractor,
        &split_map,
        SplitName::Train,
    )
    .unwrap();
    let val_examples = build_pair_examples(
        &profiles,
        &build_split_candidates(&profiles, &split_map, SplitName::Val).unwrap(),
        &extractor,
        &split_map,
        SplitName::Val,
    )
    .unwrap();
    let matcher = train_hybrid_matcher(&train_examples, &val_examples, &extractor, Some(7)).unwrap();
    let all_examples = build_examples_for_profiles(&profiles, &extractor, SplitName::Test).unwrap();
    let (identities, _) = resolve_identities(&profiles, &all_examples, &matcher, &extractor).unwrap();
    let score = bcubed_f1(&profiles, &identities);

    let identity_by_profile = identities
        .iter()
        .flat_map(|identity| {
            identity
                .member_profile_ids
                .iter()
                .map(move |profile_id| (profile_id.clone(), identity.entity_id.clone()))
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    let report = run_benchmark("real_curated_core", "grouped_cv", Some(vec![7, 11])).unwrap();

    assert!(score >= 0.8, "bcubed f1 was {score}");
    assert_eq!(identity_by_profile["sebastian_github"], identity_by_profile["sebastian_personal"]);
    assert_ne!(
        identity_by_profile["sebastian_github"],
        identity_by_profile["sebastian_raschka_finance"]
    );
    assert_eq!(report.dataset_name, "real_curated_core");
    assert_eq!(report.model_metrics.len(), 4);
    assert_eq!(report.cv_summary.len(), 2);
    assert!(report.leakage_checks.iter().all(|item| item["passed"].as_bool().unwrap_or(false)));
    assert!(report.stress_metrics["hard_negative_bank"]["f1"].as_f64().unwrap_or_default() >= 0.9);
}

#[test]
fn evidence_card_contains_reason_codes() {
    let profiles = load_dataset("hard_negative_bank").unwrap().profiles;
    let split_map = assign_profile_splits(&profiles);
    let extractor = PairFeatureExtractor::default()
        .fit(&profiles_for_split(&profiles, &split_map, SplitName::Train))
        .unwrap();
    let examples = build_examples_for_profiles(&profiles, &extractor, SplitName::Test).unwrap();
    let lookup = profiles
        .iter()
        .cloned()
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect::<std::collections::BTreeMap<_, _>>();
    let example = examples
        .iter()
        .find(|example| example.left_profile_id == "andrej_github" && example.right_profile_id == "andrej_personal")
        .unwrap();
    let card = generate_evidence_card(
        &lookup[&example.left_profile_id],
        &lookup[&example.right_profile_id],
        example,
        0.98,
        "match",
    )
    .unwrap();

    assert!(!card.supporting_signals.is_empty());
    assert!(card.reason_codes.iter().any(|code| code == "high_name_similarity"));
    assert!(card.final_explanation.contains("Decision=match"));
}
