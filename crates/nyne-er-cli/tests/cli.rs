use std::fs;

use anyhow::Result;
use assert_cmd::Command;
use tempfile::NamedTempFile;

#[test]
fn test_benchmark_command_writes_json() -> Result<()> {
    let output = NamedTempFile::new()?;
    Command::cargo_bin("nyne-er")?
        .args([
            "benchmark",
            "--dataset",
            "real_curated_core",
            "--seeds",
            "7",
            "--output",
            output.path().to_str().unwrap(),
        ])
        .assert()
        .success();
    let payload: serde_json::Value = serde_json::from_str(&fs::read_to_string(output.path())?)?;
    assert_eq!(payload["dataset_name"], "real_curated_core");
    assert!(payload["headline_metrics"]["f1"].as_f64().unwrap() >= 0.85);
    Ok(())
}

#[test]
fn test_resolve_command_accepts_text_json() -> Result<()> {
    let input = NamedTempFile::new()?;
    fs::write(
        input.path(),
        serde_json::json!({
            "display_name": "Andrej Karpathy",
            "bio": "Researcher and engineer focused on deep learning, autonomous systems, and AI education.",
            "source_type": "personal_site",
            "headline": "Building foundation models and learning systems.",
            "organizations": ["OpenAI", "Tesla"],
            "locations": ["San Francisco"],
            "topics": ["deep learning", "llms"],
            "url": "https://example.com/andrej"
        })
        .to_string(),
    )?;
    Command::cargo_bin("nyne-er")?
        .args([
            "resolve",
            "--text-json",
            input.path().to_str().unwrap(),
            "--dataset",
            "real_curated_core",
            "--output",
            "-",
        ])
        .assert()
        .success();
    Ok(())
}
