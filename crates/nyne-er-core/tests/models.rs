use anyhow::Result;
use nyne_er_core::{
    PairFeatureExtractor, SplitName, assert_person_disjoint, assign_profile_splits,
    build_pair_examples, build_split_candidates, load_benchmark_profiles, profiles_for_split,
    run_benchmark, run_embedding_baseline, run_feature_ablations, run_hybrid_matcher,
    run_lexical_baseline, run_name_baseline,
};

fn benchmark_inputs(
) -> Result<(
    Vec<nyne_er_core::ProfileRecord>,
    std::collections::BTreeMap<String, SplitName>,
    PairFeatureExtractor,
    Vec<nyne_er_core::PairExample>,
    Vec<nyne_er_core::PairExample>,
    Vec<nyne_er_core::PairExample>,
)> {
    let profiles = load_benchmark_profiles("real_curated_core")?;
    let split_map = assign_profile_splits(&profiles);
    assert_person_disjoint(&profiles, &split_map)?;
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
    Ok((profiles, split_map, extractor, train_examples, val_examples, test_examples))
}

#[test]
fn test_feature_extractor_and_baselines() -> Result<()> {
    let (_profiles, split_map, extractor, train_examples, val_examples, test_examples) = benchmark_inputs()?;
    let counts = nyne_er_core::summarize_split_assignments(&split_map);
    assert!(counts["train"] >= 4);
    assert!(counts["val"] >= 1);
    assert!(counts["test"] >= 1);

    let feature_row = &train_examples[0].features;
    assert!(feature_row.contains_key("name_similarity"));
    assert!(feature_row.contains_key("org_overlap_count"));
    assert!(feature_row.contains_key("embedding_cosine"));
    let matrix = extractor.vectorize_features(std::slice::from_ref(feature_row), None, true);
    assert!(matrix[0].len() >= 10);

    let name_run = run_name_baseline(&val_examples, &test_examples);
    let embedding_run = run_embedding_baseline(&val_examples, &test_examples);
    let lexical_run = run_lexical_baseline(&train_examples, &val_examples, &test_examples, &extractor)?;

    assert!((0.0..=1.0).contains(&name_run.metrics.f1));
    assert!((0.0..=1.0).contains(&embedding_run.metrics.average_precision));
    assert!((0.0..=1.0).contains(&lexical_run.metrics.precision));
    assert_eq!(lexical_run.scores.len(), test_examples.len());
    assert!(lexical_run.metrics.f1 >= 0.65);
    assert!(lexical_run.metrics.average_precision >= 0.7);
    assert!(lexical_run.metrics.f1 >= name_run.metrics.f1 - 0.15);
    Ok(())
}

#[test]
fn test_hybrid_metrics_and_ablations() -> Result<()> {
    let (_profiles, _split_map, extractor, train_examples, val_examples, test_examples) = benchmark_inputs()?;
    let first = run_hybrid_matcher(&train_examples, &val_examples, &test_examples, &extractor, Some(7), None)?;
    let second = run_hybrid_matcher(&train_examples, &val_examples, &test_examples, &extractor, Some(7), None)?;
    assert_eq!(first.raw_scores, second.raw_scores);
    assert_eq!(first.calibrated_scores, second.calibrated_scores);
    assert_eq!(first.decisions, second.decisions);
    assert!(first.calibrated_brier <= first.raw_brier);
    assert!(first.accepted_precision >= first.calibrated_metrics.precision);
    assert!(first.abstain_rate > 0.0);

    let ablations = run_feature_ablations(&train_examples, &val_examples, &test_examples, &extractor)?;
    let map = ablations.into_iter().map(|item| (item.name, item.metrics)).collect::<std::collections::BTreeMap<_, _>>();
    assert!(map["full"].f1 >= 0.85);
    assert!(map["full"].average_precision >= map["no_embedding"].average_precision - 0.05);
    Ok(())
}

#[test]
fn test_benchmark_headline_thresholds() -> Result<()> {
    let report = run_benchmark("real_curated_core", "grouped_cv", Some(vec![7, 11, 17]))?;
    let metrics = report
        .model_metrics
        .iter()
        .map(|item| {
            (
                item["name"].as_str().unwrap().to_string(),
                item["f1"].as_f64().unwrap(),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert!(metrics["hybrid"] >= metrics["embedding_only"]);
    assert!(metrics["hybrid"] >= 0.85);
    assert!(report.stress_metrics["hard_negative_bank"]["f1"].as_f64().unwrap() >= 0.9);
    Ok(())
}
