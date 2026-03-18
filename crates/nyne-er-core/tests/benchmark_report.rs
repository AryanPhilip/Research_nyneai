use anyhow::Result;
use nyne_er_core::{available_datasets, load_benchmark_profiles, load_dataset, run_benchmark};

#[test]
fn test_dataset_registry_and_report_shape() -> Result<()> {
    assert_eq!(
        available_datasets(),
        vec![
            "real_curated_core".to_string(),
            "hard_negative_bank".to_string(),
            "synthetic_stress".to_string(),
        ]
    );

    let bundle = load_dataset("real_curated_core")?;
    assert!(bundle.headline);
    assert!(!bundle.contains_synthetic);
    assert_eq!(bundle.profiles.len(), load_benchmark_profiles("real_curated_core")?.len());

    let hard_negative = load_dataset("hard_negative_bank")?;
    let synthetic = load_dataset("synthetic_stress")?;
    assert!(hard_negative.profiles.len() > bundle.profiles.len());
    assert!(synthetic.profiles.len() > bundle.profiles.len());
    assert!(synthetic.contains_synthetic);

    let report = run_benchmark("real_curated_core", "grouped_cv", Some(vec![7, 11]))?;
    assert_eq!(report.dataset_name, "real_curated_core");
    assert_eq!(report.protocol, "grouped_cv");
    assert_eq!(report.headline_metrics["dataset_name"], "real_curated_core");
    assert_eq!(report.model_metrics.len(), 4);
    assert_eq!(report.cv_summary.len(), 2);
    assert!(report.leakage_checks.len() >= 5);
    assert!(!report.failure_slices.is_empty());
    assert!(
        report.stress_metrics["hard_negative_bank"]["profile_count"]
            .as_u64()
            .unwrap()
            > report.headline_metrics["profile_count"].as_u64().unwrap()
    );
    assert!(report.leakage_checks.iter().all(|item| item["passed"].as_bool().unwrap()));
    assert!(report.stress_metrics.get("hard_negative_bank").is_some());
    assert!(report.stress_metrics.get("synthetic_stress").is_some());
    assert_eq!(report.dataset_summary["headline_dataset"], "real_curated_core");
    assert!(report.open_world_retrieval["queries"].as_u64().unwrap() >= 1);
    Ok(())
}
