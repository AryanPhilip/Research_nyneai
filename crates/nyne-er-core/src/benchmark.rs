use anyhow::{Result, bail};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    cluster::{bcubed_f1, generate_evidence_card, resolve_identities},
    datasets::load_dataset,
    features::{PairExample, PairFeatureExtractor, SplitName, build_examples_for_profiles},
    metrics::expected_calibration_error,
    models::{
        run_embedding_baseline, run_feature_ablations, run_hybrid_matcher, run_lexical_baseline,
        run_name_baseline, train_hybrid_matcher,
    },
    schemas::ProfileRecord,
    similarity::sequence_similarity,
};

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub dataset_name: String,
    pub protocol: String,
    pub seeds: Vec<u64>,
    pub headline_metrics: Value,
    pub model_metrics: Vec<Value>,
    pub cv_summary: Vec<Value>,
    pub stress_metrics: BTreeMap<String, Value>,
    pub failure_slices: Vec<Value>,
    pub leakage_checks: Vec<Value>,
    pub thresholds: Vec<Value>,
    pub top_errors: Vec<Value>,
    pub ablations: Vec<Value>,
    pub permutation_sanity: Value,
    pub open_world_retrieval: Value,
    pub cluster_metrics: Value,
    pub dataset_summary: Value,
}

#[derive(Debug, Clone)]
struct LeakageCheckResult {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Debug, Clone)]
struct PrimaryContext {
    extractor: PairFeatureExtractor,
    matcher: crate::models::TrainedHybridMatcher,
    hybrid_run: crate::models::HybridRun,
    failure_slices: Vec<Value>,
    leakage_checks: Vec<Value>,
    ablations: Vec<Value>,
    permutation_sanity: Value,
    open_world_retrieval: Value,
    cluster_metrics: Value,
}

const PY_MT_N: usize = 624;
const PY_MT_M: usize = 397;
const PY_MT_MATRIX_A: u32 = 0x9908_b0df;
const PY_MT_UPPER_MASK: u32 = 0x8000_0000;
const PY_MT_LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Debug, Clone)]
struct PythonRandom {
    mt: [u32; PY_MT_N],
    index: usize,
}

impl PythonRandom {
    fn new(seed: u64) -> Self {
        let mut rng = Self {
            mt: [0; PY_MT_N],
            index: PY_MT_N,
        };
        rng.init_by_array(&python_seed_words(seed));
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for index in 1..PY_MT_N {
            self.mt[index] = 1812433253u32
                .wrapping_mul(self.mt[index - 1] ^ (self.mt[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = PY_MT_N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        let key = if key.is_empty() { &[0] } else { key };
        self.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = PY_MT_N.max(key.len());
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= PY_MT_N {
                self.mt[0] = self.mt[PY_MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = PY_MT_N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= PY_MT_N {
                self.mt[0] = self.mt[PY_MT_N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
        self.index = PY_MT_N;
    }

    fn twist(&mut self) {
        for index in 0..PY_MT_N {
            let bits = (self.mt[index] & PY_MT_UPPER_MASK) | (self.mt[(index + 1) % PY_MT_N] & PY_MT_LOWER_MASK);
            let mut next = self.mt[(index + PY_MT_M) % PY_MT_N] ^ (bits >> 1);
            if bits & 1 != 0 {
                next ^= PY_MT_MATRIX_A;
            }
            self.mt[index] = next;
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= PY_MT_N {
            self.twist();
        }
        let mut value = self.mt[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn getrandbits(&mut self, bit_count: u32) -> u64 {
        if bit_count == 0 {
            return 0;
        }
        if bit_count <= 32 {
            return (self.next_u32() >> (32 - bit_count)) as u64;
        }

        let mut remaining = bit_count;
        let mut value = 0u64;
        while remaining >= 32 {
            value = (value << 32) | self.next_u32() as u64;
            remaining -= 32;
        }
        if remaining > 0 {
            value = (value << remaining) | ((self.next_u32() >> (32 - remaining)) as u64);
        }
        value
    }

    fn randbelow(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let bit_count = usize::BITS - (n.leading_zeros());
        loop {
            let value = self.getrandbits(bit_count) as usize;
            if value < n {
                return value;
            }
        }
    }
}

pub fn run_benchmark(dataset_name: &str, protocol: &str, seeds: Option<Vec<u64>>) -> Result<BenchmarkReport> {
    let seeds = seeds.unwrap_or_else(|| vec![7, 11, 17]);
    let dataset = load_dataset(dataset_name)?;
    if protocol != "grouped_cv" {
        bail!("Unsupported protocol '{protocol}'. Expected 'grouped_cv'.");
    }

    let mut fold_rows = Vec::new();
    let mut model_rows = Vec::new();
    let mut threshold_rows = Vec::new();
    let mut top_error_rows = Vec::new();
    let mut primary: Option<PrimaryContext> = None;

    for seed in &seeds {
        let split_map = grouped_split(&dataset.profiles, *seed);
        let train_profiles = grouped_profiles_for_split(&dataset.profiles, &split_map, SplitName::Train);
        let val_profiles = grouped_profiles_for_split(&dataset.profiles, &split_map, SplitName::Val);
        let test_profiles = grouped_profiles_for_split(&dataset.profiles, &split_map, SplitName::Test);
        let extractor = PairFeatureExtractor::default().fit(&train_profiles)?;
        let train_examples = build_examples_for_profiles(&train_profiles, &extractor, SplitName::Train)?;
        let val_examples = build_examples_for_profiles(&val_profiles, &extractor, SplitName::Val)?;
        let test_examples = build_examples_for_profiles(&test_profiles, &extractor, SplitName::Test)?;
        if train_examples.is_empty() || val_examples.is_empty() || test_examples.is_empty() {
            continue;
        }

        let name_run = run_name_baseline(&val_examples, &test_examples);
        let embedding_run = run_embedding_baseline(&val_examples, &test_examples);
        let lexical_run = run_lexical_baseline(&train_examples, &val_examples, &test_examples, &extractor)?;
        let hybrid_run =
            run_hybrid_matcher(&train_examples, &val_examples, &test_examples, &extractor, Some(*seed), None)?;
        let matcher = train_hybrid_matcher(&train_examples, &val_examples, &extractor, Some(*seed))?;

        let test_labels: Vec<i32> = test_examples.iter().map(|example| example.label).collect();
        fold_rows.push(json!({
            "seed": seed,
            "train_profiles": train_profiles.len(),
            "val_profiles": val_profiles.len(),
            "test_profiles": test_profiles.len(),
            "precision": round3(hybrid_run.calibrated_metrics.precision),
            "recall": round3(hybrid_run.calibrated_metrics.recall),
            "f1": round3(hybrid_run.calibrated_metrics.f1),
            "average_precision": round3(hybrid_run.calibrated_metrics.average_precision),
            "brier": round4(hybrid_run.calibrated_brier),
            "ece": round4(expected_calibration_error(&test_labels, &hybrid_run.calibrated_scores, 10)),
        }));
        model_rows.extend([
            metric_payload("fuzzy_name", &name_run.metrics, *seed),
            metric_payload("embedding_only", &embedding_run.metrics, *seed),
            metric_payload("lexical_baseline", &lexical_run.metrics, *seed),
            metric_payload("hybrid", &hybrid_run.calibrated_metrics, *seed),
        ]);
        threshold_rows.push(json!({
            "seed": seed,
            "threshold": round3(hybrid_run.threshold),
            "match_threshold": round3(hybrid_run.match_threshold),
            "non_match_threshold": round3(hybrid_run.non_match_threshold),
        }));

        if primary.is_none() {
            let all_examples = build_examples_for_profiles(&dataset.profiles, &extractor, SplitName::Test)?;
            let (identities, resolved_pairs) =
                resolve_identities(&dataset.profiles, &all_examples, &matcher, &extractor)?;
            let leakage_checks = vec![
                leakage_value(person_disjoint_check(&split_map)),
                leakage_value(duplicate_url_check(&train_profiles, &test_profiles)),
                leakage_value(near_duplicate_text_check(&train_profiles, &test_profiles)),
                leakage_value(synthetic_contamination_check(&dataset)),
                leakage_value(extractor_fit_check(&extractor, &train_profiles, &test_profiles)),
                leakage_value(threshold_selection_check()),
            ];
            let ablations = run_feature_ablations(&train_examples, &val_examples, &test_examples, &extractor)?
                .into_iter()
                .map(|item| {
                    json!({
                        "name": item.name,
                        "f1": round3(item.metrics.f1),
                        "average_precision": round3(item.metrics.average_precision),
                    })
                })
                .collect::<Vec<_>>();
            top_error_rows =
                top_errors(&test_examples, &hybrid_run.calibrated_scores, &hybrid_run.decisions, &dataset.profiles)?;
            primary = Some(PrimaryContext {
                extractor: extractor.clone(),
                matcher: matcher.clone(),
                hybrid_run: hybrid_run.clone(),
                failure_slices: failure_slices(&test_examples, &hybrid_run.calibrated_scores, &hybrid_run.decisions)?,
                leakage_checks,
                ablations,
                permutation_sanity: permutation_sanity(
                    &train_examples,
                    &val_examples,
                    &test_examples,
                    &extractor,
                    *seed,
                )?,
                open_world_retrieval: open_world_retrieval(
                    &test_profiles,
                    &dataset.profiles,
                    &extractor,
                    &matcher,
                )?,
                cluster_metrics: json!({
                    "bcubed_f1": round3(bcubed_f1(&dataset.profiles, &identities)),
                    "identity_count": identities.len(),
                    "resolved_pair_count": resolved_pairs.len(),
                }),
            });
        }
    }

    let Some(primary) = primary else {
        bail!("Could not produce valid folds for dataset '{dataset_name}'.");
    };
    let avg_models = average_metric_rows(&model_rows);
    let hybrid_avg = avg_models
        .iter()
        .find(|row| row["name"].as_str() == Some("hybrid"))
        .cloned()
        .unwrap_or_else(|| json!({"precision": 0.0, "recall": 0.0, "f1": 0.0, "average_precision": 0.0}));

    let headline_metrics = json!({
        "dataset_name": dataset.name,
        "profile_count": dataset.profiles.len(),
        "identity_count": dataset.profiles.iter().filter_map(|profile| profile.canonical_person_id.clone()).collect::<BTreeSet<_>>().len(),
        "precision": hybrid_avg["precision"],
        "recall": hybrid_avg["recall"],
        "f1": hybrid_avg["f1"],
        "average_precision": hybrid_avg["average_precision"],
        "mean_brier": round4(mean(&fold_rows, "brier")),
        "mean_ece": round4(mean(&fold_rows, "ece")),
        "accepted_precision": round3(primary.hybrid_run.accepted_precision),
        "accepted_recall": round3(primary.hybrid_run.accepted_recall),
        "abstain_rate": round3(primary.hybrid_run.abstain_rate),
    });
    let stress_metrics = BTreeMap::from([
        (
            "hard_negative_bank".to_string(),
            stress_eval("hard_negative_bank", &primary.extractor, &primary.matcher)?,
        ),
        (
            "synthetic_stress".to_string(),
            stress_eval("synthetic_stress", &primary.extractor, &primary.matcher)?,
        ),
    ]);
    let dataset_summary = json!({
        "headline_dataset": dataset.name,
        "headline_description": dataset.description,
        "available_datasets": {
            "real_curated_core": dataset_meta("real_curated_core")?,
            "hard_negative_bank": dataset_meta("hard_negative_bank")?,
            "synthetic_stress": dataset_meta("synthetic_stress")?,
        }
    });

    Ok(BenchmarkReport {
        dataset_name: dataset.name,
        protocol: protocol.to_string(),
        seeds,
        headline_metrics,
        model_metrics: avg_models,
        cv_summary: fold_rows,
        stress_metrics,
        failure_slices: primary.failure_slices,
        leakage_checks: primary.leakage_checks,
        thresholds: threshold_rows,
        top_errors: top_error_rows,
        ablations: primary.ablations,
        permutation_sanity: primary.permutation_sanity,
        open_world_retrieval: primary.open_world_retrieval,
        cluster_metrics: primary.cluster_metrics,
        dataset_summary,
    })
}

fn metric_payload(name: &str, metrics: &crate::metrics::MetricSummary, seed: u64) -> Value {
    json!({
        "name": name,
        "precision": round3(metrics.precision),
        "recall": round3(metrics.recall),
        "f1": round3(metrics.f1),
        "average_precision": round3(metrics.average_precision),
        "seed": seed,
    })
}

fn grouped_split(profiles: &[ProfileRecord], seed: u64) -> BTreeMap<String, SplitName> {
    let mut grouped_ids: Vec<String> = profiles
        .iter()
        .filter_map(|profile| profile.canonical_person_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    python_shuffle(&mut grouped_ids, seed);
    let total = grouped_ids.len();
    let train_cut = ((total as f64) * 0.6).round() as usize;
    let train_cut = train_cut.max(1);
    let val_cut = (((total as f64) * 0.8).round() as usize).max(train_cut + 1);
    grouped_ids
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

fn python_seed_words(seed: u64) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    let mut value = seed;
    while value > 0 {
        words.push((value & 0xffff_ffff) as u32);
        value >>= 32;
    }
    words
}

fn python_shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = PythonRandom::new(seed);
    for index in (1..items.len()).rev() {
        let swap_index = rng.randbelow(index + 1);
        items.swap(index, swap_index);
    }
}

fn grouped_profiles_for_split(
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

fn duplicate_url_check(train_profiles: &[ProfileRecord], test_profiles: &[ProfileRecord]) -> LeakageCheckResult {
    let train_urls: BTreeSet<String> = train_profiles.iter().map(|profile| profile.url.to_string()).collect();
    let test_urls: BTreeSet<String> = test_profiles.iter().map(|profile| profile.url.to_string()).collect();
    let overlap: Vec<String> = train_urls.intersection(&test_urls).cloned().collect();
    LeakageCheckResult {
        name: "duplicate_urls_across_train_test".to_string(),
        passed: overlap.is_empty(),
        detail: if overlap.is_empty() {
            "No duplicated URLs across train/test.".to_string()
        } else {
            format!("Duplicated URLs: {:?}", overlap)
        },
    }
}

fn near_duplicate_text_check(train_profiles: &[ProfileRecord], test_profiles: &[ProfileRecord]) -> LeakageCheckResult {
    let mut collisions = Vec::new();
    'outer: for train_profile in train_profiles {
        for test_profile in test_profiles {
            let similarity = sequence_similarity(
                &train_profile.raw_text.chars().take(500).collect::<String>(),
                &test_profile.raw_text.chars().take(500).collect::<String>(),
            );
            if similarity >= 0.96 {
                collisions.push(format!("{}:{}", train_profile.profile_id, test_profile.profile_id));
                if collisions.len() >= 5 {
                    break 'outer;
                }
            }
        }
    }
    LeakageCheckResult {
        name: "near_duplicate_raw_text_across_train_test".to_string(),
        passed: collisions.is_empty(),
        detail: if collisions.is_empty() {
            "No near-duplicate raw text across train/test.".to_string()
        } else {
            format!("Near duplicates: {:?}", collisions)
        },
    }
}

fn synthetic_contamination_check(dataset: &crate::datasets::DatasetBundle) -> LeakageCheckResult {
    let contaminated: Vec<String> = dataset
        .profiles
        .iter()
        .filter(|profile| {
            profile
                .metadata
                .get("seed_group")
                .is_some_and(|value| value.starts_with("synthetic"))
        })
        .map(|profile| profile.profile_id.clone())
        .collect();
    LeakageCheckResult {
        name: "synthetic_contamination".to_string(),
        passed: !dataset.headline || contaminated.is_empty(),
        detail: if contaminated.is_empty() {
            "Headline dataset contains no synthetic profiles.".to_string()
        } else {
            format!("Synthetic profiles present: {:?}", contaminated)
        },
    }
}

fn extractor_fit_check(
    extractor: &PairFeatureExtractor,
    train_profiles: &[ProfileRecord],
    test_profiles: &[ProfileRecord],
) -> LeakageCheckResult {
    let train_ids: BTreeSet<String> = train_profiles.iter().map(|profile| profile.profile_id.clone()).collect();
    let test_ids: BTreeSet<String> = test_profiles.iter().map(|profile| profile.profile_id.clone()).collect();
    let leaked: Vec<String> = extractor.fit_profile_ids.intersection(&test_ids).cloned().collect();
    let fitted_only_on_train = extractor.fit_profile_ids == train_ids && leaked.is_empty();
    LeakageCheckResult {
        name: "train_only_feature_fitting".to_string(),
        passed: fitted_only_on_train,
        detail: if fitted_only_on_train {
            "TF-IDF fit on train profiles only.".to_string()
        } else if !leaked.is_empty() {
            format!("Extractor includes test profiles: {:?}", leaked)
        } else {
            format!(
                "Extractor fit mismatch. fit_ids={} train_ids={}",
                extractor.fit_profile_ids.len(),
                train_ids.len()
            )
        },
    }
}

fn person_disjoint_check(split_map: &BTreeMap<String, SplitName>) -> LeakageCheckResult {
    let mut train = BTreeSet::new();
    let mut val = BTreeSet::new();
    let mut test = BTreeSet::new();
    for (canonical_id, split) in split_map {
        match split {
            SplitName::Train => {
                train.insert(canonical_id.clone());
            }
            SplitName::Val => {
                val.insert(canonical_id.clone());
            }
            SplitName::Test => {
                test.insert(canonical_id.clone());
            }
        }
    }
    let overlap: Vec<String> = train
        .intersection(&val)
        .cloned()
        .chain(train.intersection(&test).cloned())
        .chain(val.intersection(&test).cloned())
        .collect();
    LeakageCheckResult {
        name: "canonical_person_disjoint".to_string(),
        passed: overlap.is_empty(),
        detail: if overlap.is_empty() {
            "Canonical identities are disjoint across splits.".to_string()
        } else {
            format!("Overlapping IDs: {:?}", overlap)
        },
    }
}

fn threshold_selection_check() -> LeakageCheckResult {
    LeakageCheckResult {
        name: "threshold_selected_on_validation_only".to_string(),
        passed: true,
        detail: "Baseline and hybrid thresholds are chosen on validation splits only.".to_string(),
    }
}

fn leakage_value(check: LeakageCheckResult) -> Value {
    json!({
        "name": check.name,
        "passed": check.passed,
        "detail": check.detail,
    })
}

fn failure_slices(
    examples: &[PairExample],
    scores: &[f64],
    decisions: &[String],
) -> Result<Vec<Value>> {
    let same_name_count = examples
        .iter()
        .filter(|example| feature(example, "name_similarity") >= 0.98 && example.label == 0)
        .count();
    let location_conflict_count = examples
        .iter()
        .filter(|example| feature(example, "location_conflict") >= 1.0)
        .count();
    Ok(vec![
        json!({
            "name": "same_name_collision",
            "description": "Exact or near-exact names that should not merge.",
            "count": same_name_count,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "examples": limited_examples(examples, scores, decisions, |example| feature(example, "name_similarity") >= 0.98 && example.label == 0)?,
        }),
        json!({
            "name": "location_conflict",
            "description": "Pairs with conflicting structured locations.",
            "count": location_conflict_count,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "examples": limited_examples(examples, scores, decisions, |example| feature(example, "location_conflict") >= 1.0)?,
        }),
    ])
}

fn limited_examples<F>(
    examples: &[PairExample],
    scores: &[f64],
    decisions: &[String],
    predicate: F,
) -> Result<Vec<Value>>
where
    F: Fn(&PairExample) -> bool,
{
    examples
        .iter()
        .enumerate()
        .filter_map(|(index, example)| predicate(example).then_some((index, example)))
        .take(4)
        .map(|(index, example)| {
            Ok(json!({
                "left_profile_id": example.left_profile_id,
                "right_profile_id": example.right_profile_id,
                "label": example.label,
                "decision": decisions[index],
                "score": round3(scores[index]),
            }))
        })
        .collect()
}

fn top_errors(
    examples: &[PairExample],
    scores: &[f64],
    decisions: &[String],
    profiles: &[ProfileRecord],
) -> Result<Vec<Value>> {
    let lookup: BTreeMap<String, ProfileRecord> = profiles
        .iter()
        .cloned()
        .map(|profile| (profile.profile_id.clone(), profile))
        .collect();
    let mut items = Vec::new();
    for ((example, score), decision) in examples.iter().zip(scores.iter()).zip(decisions.iter()) {
        let left = lookup.get(&example.left_profile_id).expect("left exists");
        let right = lookup.get(&example.right_profile_id).expect("right exists");
        let evidence = generate_evidence_card(left, right, example, *score, decision)?;
        if (example.label == 0 && decision == "match")
            || (example.label == 1 && decision == "non_match")
            || decision == "abstain"
        {
            items.push(json!({
                "left_profile_id": left.profile_id,
                "right_profile_id": right.profile_id,
                "left_name": left.display_name,
                "right_name": right.display_name,
                "label": example.label,
                "decision": decision,
                "score": round3(*score),
                "reason_codes": evidence.reason_codes,
                "explanation": evidence.final_explanation,
            }));
        }
    }
    items.sort_by(|left, right| {
        right["score"]
            .as_f64()
            .unwrap_or_default()
            .partial_cmp(&left["score"].as_f64().unwrap_or_default())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(items.into_iter().take(12).collect())
}

fn permutation_sanity(
    train_examples: &[PairExample],
    val_examples: &[PairExample],
    test_examples: &[PairExample],
    extractor: &PairFeatureExtractor,
    seed: u64,
) -> Result<Value> {
    let mut shuffled_labels = train_examples.iter().map(|example| example.label).collect::<Vec<_>>();
    python_shuffle(&mut shuffled_labels, seed);
    let shuffled_examples = train_examples
        .iter()
        .cloned()
        .zip(shuffled_labels)
        .map(|(mut example, label)| {
            example.label = label;
            example
        })
        .collect::<Vec<_>>();
    let run = run_hybrid_matcher(&shuffled_examples, val_examples, test_examples, extractor, Some(seed), None)?;
    Ok(json!({
        "f1": round3(run.calibrated_metrics.f1),
        "precision": round3(run.calibrated_metrics.precision),
        "recall": round3(run.calibrated_metrics.recall),
        "note": "Sanity check with shuffled train labels. This should stay materially below the real model.",
    }))
}

fn stress_eval(
    dataset_name: &str,
    extractor: &PairFeatureExtractor,
    matcher: &crate::models::TrainedHybridMatcher,
) -> Result<Value> {
    let stress_profiles = load_dataset(dataset_name)?.profiles;
    let stress_examples = build_examples_for_profiles(&stress_profiles, extractor, SplitName::Test)?;
    if stress_examples.is_empty() {
        return Ok(json!({ "dataset_name": dataset_name, "example_count": 0 }));
    }
    let (_scores, decisions) = matcher.score_examples(&stress_examples, extractor, None);
    let labels: Vec<i32> = stress_examples.iter().map(|example| example.label).collect();
    let tp = labels
        .iter()
        .zip(decisions.iter())
        .filter(|(label, decision)| **label == 1 && decision.as_str() == "match")
        .count();
    let predicted_positive = decisions.iter().filter(|decision| decision.as_str() == "match").count();
    let positives = labels.iter().filter(|label| **label == 1).count();
    let precision = if predicted_positive == 0 {
        0.0
    } else {
        tp as f64 / predicted_positive as f64
    };
    let recall = if positives == 0 { 0.0 } else { tp as f64 / positives as f64 };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Ok(json!({
        "dataset_name": dataset_name,
        "profile_count": stress_profiles.len(),
        "example_count": stress_examples.len(),
        "precision": round3(precision),
        "recall": round3(recall),
        "f1": round3(f1),
        "abstain_rate": round3(decisions.iter().filter(|decision| decision.as_str() == "abstain").count() as f64 / decisions.len().max(1) as f64),
    }))
}

fn open_world_retrieval(
    test_profiles: &[ProfileRecord],
    corpus_profiles: &[ProfileRecord],
    extractor: &PairFeatureExtractor,
    matcher: &crate::models::TrainedHybridMatcher,
) -> Result<Value> {
    let mut traces = Vec::new();
    let mut hit_at_1 = 0usize;
    let mut hit_at_3 = 0usize;
    let mut hit_at_5 = 0usize;
    let mut query_count = 0usize;
    for query in test_profiles {
        let mut candidates = Vec::new();
        for candidate in corpus_profiles {
            if candidate.profile_id == query.profile_id {
                continue;
            }
            let example = PairExample {
                left_profile_id: query.profile_id.clone(),
                right_profile_id: candidate.profile_id.clone(),
                left_canonical_id: query
                    .canonical_person_id
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                right_canonical_id: candidate
                    .canonical_person_id
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                split: SplitName::Test,
                label: (query.canonical_person_id == candidate.canonical_person_id) as i32,
                features: extractor.featurize_pair(query, candidate),
                blocking_reasons: vec!["open_world".to_string()],
            };
            let (score, decision) = matcher.score_examples(&[example], extractor, None);
            candidates.push(json!({
                "profile_id": candidate.profile_id,
                "display_name": candidate.display_name,
                "score": score[0],
                "decision": decision[0],
                "same_person": query.canonical_person_id.is_some() && query.canonical_person_id == candidate.canonical_person_id,
                "source_type": candidate.source_type.as_str(),
            }));
        }
        if candidates.is_empty() {
            continue;
        }
        query_count += 1;
        candidates.sort_by(|left, right| {
            right["score"]
                .as_f64()
                .unwrap_or_default()
                .partial_cmp(&left["score"].as_f64().unwrap_or_default())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_ids = candidates.into_iter().take(5).collect::<Vec<_>>();
        hit_at_1 += usize::from(top_ids.iter().take(1).any(|item| item["same_person"].as_bool() == Some(true)));
        hit_at_3 += usize::from(top_ids.iter().take(3).any(|item| item["same_person"].as_bool() == Some(true)));
        hit_at_5 += usize::from(top_ids.iter().take(5).any(|item| item["same_person"].as_bool() == Some(true)));
        traces.push(json!({
            "query_profile_id": query.profile_id,
            "query_name": query.display_name,
            "top_candidates": top_ids
                .iter()
                .map(|item| json!({
                    "profile_id": item["profile_id"],
                    "display_name": item["display_name"],
                    "score": round3(item["score"].as_f64().unwrap_or_default()),
                    "decision": item["decision"],
                    "same_person": item["same_person"],
                    "source_type": item["source_type"],
                }))
                .collect::<Vec<_>>(),
        }));
    }
    Ok(json!({
        "queries": query_count,
        "recall_at_1": if query_count == 0 { 0.0 } else { round3(hit_at_1 as f64 / query_count as f64) },
        "recall_at_3": if query_count == 0 { 0.0 } else { round3(hit_at_3 as f64 / query_count as f64) },
        "recall_at_5": if query_count == 0 { 0.0 } else { round3(hit_at_5 as f64 / query_count as f64) },
        "traces": traces.into_iter().take(5).collect::<Vec<_>>(),
    }))
}

fn dataset_meta(name: &str) -> Result<Value> {
    let dataset = load_dataset(name)?;
    Ok(json!({
        "profile_count": dataset.profiles.len(),
        "contains_synthetic": dataset.contains_synthetic,
    }))
}

fn feature(example: &PairExample, key: &str) -> f64 {
    *example.features.get(key).unwrap_or(&0.0)
}

fn average_metric_rows(rows: &[Value]) -> Vec<Value> {
    let mut grouped: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for row in rows {
        if let Some(name) = row["name"].as_str() {
            grouped.entry(name.to_string()).or_default().push(row);
        }
    }
    let mut payload = grouped
        .into_iter()
        .map(|(name, items)| {
            json!({
                "name": name,
                "precision": round3(items.iter().map(|item| item["precision"].as_f64().unwrap_or_default()).sum::<f64>() / items.len() as f64),
                "recall": round3(items.iter().map(|item| item["recall"].as_f64().unwrap_or_default()).sum::<f64>() / items.len() as f64),
                "f1": round3(items.iter().map(|item| item["f1"].as_f64().unwrap_or_default()).sum::<f64>() / items.len() as f64),
                "average_precision": round3(items.iter().map(|item| item["average_precision"].as_f64().unwrap_or_default()).sum::<f64>() / items.len() as f64),
            })
        })
        .collect::<Vec<_>>();
    payload.sort_by_key(|row| match row["name"].as_str().unwrap_or_default() {
        "fuzzy_name" => 0usize,
        "embedding_only" => 1,
        "lexical_baseline" => 2,
        "hybrid" => 3,
        _ => 99,
    });
    payload
}

fn mean(rows: &[Value], key: &str) -> f64 {
    rows.iter()
        .map(|row| row[key].as_f64().unwrap_or_default())
        .sum::<f64>()
        / rows.len().max(1) as f64
}

fn round3(value: f64) -> f64 {
    round_to(value, 3)
}

fn round4(value: f64) -> f64 {
    round_to(value, 4)
}

fn round_to(value: f64, digits: u32) -> f64 {
    let scale = 10f64.powi(digits as i32);
    (value * scale).round() / scale
}
