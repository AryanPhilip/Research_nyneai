use crate::features::{PairExample, PairFeatureExtractor, FEATURE_ORDER};
use crate::metrics::{brier_score, optimize_threshold, summarize_predictions, MetricSummary};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const LEXICAL_FEATURES: [&str; 5] = [
    "name_similarity",
    "alias_similarity",
    "headline_similarity",
    "bio_similarity",
    "same_source_type",
];
pub const STRUCTURED_FEATURES: [&str; 9] = [
    "shared_domain_count",
    "org_overlap_count",
    "org_jaccard",
    "topic_overlap_count",
    "topic_jaccard",
    "location_overlap",
    "temporal_overlap_count",
    "temporal_distance",
    "location_conflict",
];
pub const FULL_FEATURES: [&str; 15] = FEATURE_ORDER;

pub trait LlmAdjudicator {
    fn adjudicate(&self, _example: &PairExample, _score: f64) -> Option<f64> {
        None
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NullLlmAdjudicator;

impl LlmAdjudicator for NullLlmAdjudicator {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineRun {
    pub name: String,
    pub threshold: f64,
    pub metrics: MetricSummary,
    pub scores: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridRun {
    pub name: String,
    pub threshold: f64,
    pub match_threshold: f64,
    pub non_match_threshold: f64,
    pub raw_metrics: MetricSummary,
    pub calibrated_metrics: MetricSummary,
    pub raw_brier: f64,
    pub calibrated_brier: f64,
    pub accepted_precision: f64,
    pub accepted_recall: f64,
    pub abstain_rate: f64,
    pub raw_scores: Vec<f64>,
    pub calibrated_scores: Vec<f64>,
    pub decisions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationResult {
    pub name: String,
    pub metrics: MetricSummary,
}

#[derive(Debug, Clone)]
pub struct TrainedHybridMatcher {
    pub model: LogisticModel,
    pub calibrator: Calibrator,
    pub feature_names: Vec<&'static str>,
    pub threshold: f64,
    pub match_threshold: f64,
    pub non_match_threshold: f64,
}

impl TrainedHybridMatcher {
    pub fn score_examples(
        &self,
        examples: &[PairExample],
        extractor: &PairFeatureExtractor,
        adjudicator: Option<&dyn LlmAdjudicator>,
    ) -> (Vec<f64>, Vec<String>) {
        let rows = feature_rows(examples);
        let matrix = extractor.vectorize_features(&rows, Some(&self.feature_names), true);
        let raw_scores = self.model.predict_proba_batch(&matrix);
        let calibrated_scores = self.calibrator.transform_many(&raw_scores);
        let adjudicator = adjudicator.unwrap_or(&NullLlmAdjudicator);

        let mut final_scores = Vec::new();
        let mut decisions = Vec::new();
        for (example, score) in examples.iter().zip(calibrated_scores.iter().copied()) {
            let score = adjudicator.adjudicate(example, score).unwrap_or(score);
            final_scores.push(score);
            if contradiction_veto(example, score, self.match_threshold) {
                decisions.push("abstain".to_string());
            } else if score >= self.match_threshold {
                decisions.push("match".to_string());
            } else if score <= self.non_match_threshold {
                decisions.push("non_match".to_string());
            } else {
                decisions.push("abstain".to_string());
            }
        }
        (final_scores, decisions)
    }
}

pub fn run_name_baseline(val_examples: &[PairExample], test_examples: &[PairExample]) -> BaselineRun {
    let val_scores: Vec<f64> = val_examples
        .iter()
        .map(|example| feature_value(example, "name_similarity"))
        .collect();
    let threshold = optimize_threshold(&labels(val_examples), &val_scores);
    let test_scores: Vec<f64> = test_examples
        .iter()
        .map(|example| feature_value(example, "name_similarity"))
        .collect();
    let metrics = summarize_predictions(&labels(test_examples), &test_scores, threshold);
    BaselineRun {
        name: "fuzzy_name".to_string(),
        threshold,
        metrics,
        scores: test_scores,
    }
}

pub fn run_embedding_baseline(val_examples: &[PairExample], test_examples: &[PairExample]) -> BaselineRun {
    let val_scores: Vec<f64> = val_examples
        .iter()
        .map(|example| feature_value(example, "embedding_cosine"))
        .collect();
    let threshold = optimize_threshold(&labels(val_examples), &val_scores);
    let test_scores: Vec<f64> = test_examples
        .iter()
        .map(|example| feature_value(example, "embedding_cosine"))
        .collect();
    let metrics = summarize_predictions(&labels(test_examples), &test_scores, threshold);
    BaselineRun {
        name: "embedding_only".to_string(),
        threshold,
        metrics,
        scores: test_scores,
    }
}

pub fn run_lexical_baseline(
    train_examples: &[PairExample],
    val_examples: &[PairExample],
    test_examples: &[PairExample],
    extractor: &PairFeatureExtractor,
) -> Result<BaselineRun> {
    let train_rows = project_feature_rows(train_examples, &LEXICAL_FEATURES);
    let val_rows = project_feature_rows(val_examples, &LEXICAL_FEATURES);
    let test_rows = project_feature_rows(test_examples, &LEXICAL_FEATURES);

    let train_matrix = extractor.vectorize_features(&train_rows, Some(&LEXICAL_FEATURES), false);
    let val_matrix = extractor.vectorize_features(&val_rows, Some(&LEXICAL_FEATURES), false);
    let test_matrix = extractor.vectorize_features(&test_rows, Some(&LEXICAL_FEATURES), false);

    let model = LogisticModel::fit(&train_matrix, &labels(train_examples), 42);
    let val_scores = model.predict_proba_batch(&val_matrix);
    let threshold = optimize_threshold(&labels(val_examples), &val_scores);
    let test_scores = model.predict_proba_batch(&test_matrix);
    let metrics = summarize_predictions(&labels(test_examples), &test_scores, threshold);

    Ok(BaselineRun {
        name: "lexical_logistic".to_string(),
        threshold,
        metrics,
        scores: test_scores,
    })
}

pub fn run_hybrid_matcher(
    train_examples: &[PairExample],
    val_examples: &[PairExample],
    test_examples: &[PairExample],
    extractor: &PairFeatureExtractor,
    random_state: Option<u64>,
    adjudicator: Option<&dyn LlmAdjudicator>,
) -> Result<HybridRun> {
    let feature_names = FULL_FEATURES.to_vec();
    let train_matrix = extractor.vectorize_features(&feature_rows(train_examples), Some(&feature_names), true);
    let val_matrix = extractor.vectorize_features(&feature_rows(val_examples), Some(&feature_names), true);
    let test_matrix = extractor.vectorize_features(&feature_rows(test_examples), Some(&feature_names), true);

    let model = LogisticModel::fit(&train_matrix, &labels(train_examples), random_state.unwrap_or(42));
    let raw_val_scores = model.predict_proba_batch(&val_matrix);
    let raw_test_scores = model.predict_proba_batch(&test_matrix);
    let raw_threshold = optimize_threshold(&labels(val_examples), &raw_val_scores);
    let raw_metrics = summarize_predictions(&labels(test_examples), &raw_test_scores, raw_threshold);

    let (calibrated_val_scores, calibrator) =
        fit_best_calibrator(&raw_val_scores, &labels(val_examples), random_state.unwrap_or(42));
    let calibrated_test_scores = calibrator.transform_many(&raw_test_scores);
    let calibrated_threshold = optimize_threshold(&labels(val_examples), &calibrated_val_scores);
    let calibrated_metrics =
        summarize_predictions(&labels(test_examples), &calibrated_test_scores, calibrated_threshold);

    let (non_match_threshold, match_threshold) = optimize_abstain_band(
        val_examples,
        &calibrated_val_scores,
        calibrated_threshold.max(0.6),
        calibrated_threshold.min(0.4),
    );

    let trained_matcher = TrainedHybridMatcher {
        model,
        calibrator,
        feature_names,
        threshold: calibrated_threshold,
        match_threshold,
        non_match_threshold,
    };
    let (final_scores, decisions) = trained_matcher.score_examples(test_examples, extractor, adjudicator);
    let (accepted_precision, accepted_recall, abstain_rate) = abstain_stats(&labels(test_examples), &decisions);

    Ok(HybridRun {
        name: "hybrid_gradient_boosting".to_string(),
        threshold: calibrated_threshold,
        match_threshold,
        non_match_threshold,
        raw_metrics,
        calibrated_metrics,
        raw_brier: brier_score(&labels(test_examples), &raw_test_scores),
        calibrated_brier: brier_score(&labels(test_examples), &calibrated_test_scores),
        accepted_precision,
        accepted_recall,
        abstain_rate,
        raw_scores: raw_test_scores,
        calibrated_scores: final_scores,
        decisions,
    })
}

pub fn run_feature_ablations(
    train_examples: &[PairExample],
    val_examples: &[PairExample],
    test_examples: &[PairExample],
    extractor: &PairFeatureExtractor,
) -> Result<Vec<AblationResult>> {
    let runs = vec![
        ("full", FULL_FEATURES.to_vec()),
        (
            "no_embedding",
            LEXICAL_FEATURES
                .iter()
                .chain(STRUCTURED_FEATURES.iter())
                .copied()
                .collect::<Vec<_>>(),
        ),
        (
            "no_structured",
            LEXICAL_FEATURES
                .iter()
                .copied()
                .chain(std::iter::once("embedding_cosine"))
                .collect::<Vec<_>>(),
        ),
    ];

    let mut results = Vec::new();
    for (name, feature_names) in runs {
        let run = run_model_with_features(train_examples, val_examples, test_examples, extractor, &feature_names)?;
        results.push(AblationResult {
            name: name.to_string(),
            metrics: run.calibrated_metrics,
        });
    }
    Ok(results)
}

pub fn train_hybrid_matcher(
    train_examples: &[PairExample],
    val_examples: &[PairExample],
    extractor: &PairFeatureExtractor,
    random_state: Option<u64>,
) -> Result<TrainedHybridMatcher> {
    let feature_names = FULL_FEATURES.to_vec();
    let train_matrix = extractor.vectorize_features(&feature_rows(train_examples), Some(&feature_names), true);
    let val_matrix = extractor.vectorize_features(&feature_rows(val_examples), Some(&feature_names), true);
    let model = LogisticModel::fit(&train_matrix, &labels(train_examples), random_state.unwrap_or(42));

    let raw_val_scores = model.predict_proba_batch(&val_matrix);
    let (calibrated_val_scores, calibrator) =
        fit_best_calibrator(&raw_val_scores, &labels(val_examples), random_state.unwrap_or(42));
    let calibrated_threshold = optimize_threshold(&labels(val_examples), &calibrated_val_scores);
    let (non_match_threshold, match_threshold) = optimize_abstain_band(
        val_examples,
        &calibrated_val_scores,
        calibrated_threshold.max(0.6),
        calibrated_threshold.min(0.4),
    );

    Ok(TrainedHybridMatcher {
        model,
        calibrator,
        feature_names,
        threshold: calibrated_threshold,
        match_threshold,
        non_match_threshold,
    })
}

fn run_model_with_features(
    train_examples: &[PairExample],
    val_examples: &[PairExample],
    test_examples: &[PairExample],
    extractor: &PairFeatureExtractor,
    feature_names: &[&'static str],
) -> Result<HybridRun> {
    let train_matrix = extractor.vectorize_features(&feature_rows(train_examples), Some(feature_names), true);
    let val_matrix = extractor.vectorize_features(&feature_rows(val_examples), Some(feature_names), true);
    let test_matrix = extractor.vectorize_features(&feature_rows(test_examples), Some(feature_names), true);
    let model = LogisticModel::fit(&train_matrix, &labels(train_examples), 42);
    let raw_val_scores = model.predict_proba_batch(&val_matrix);
    let raw_test_scores = model.predict_proba_batch(&test_matrix);
    let raw_threshold = optimize_threshold(&labels(val_examples), &raw_val_scores);
    let raw_metrics = summarize_predictions(&labels(test_examples), &raw_test_scores, raw_threshold);

    let (calibrated_val_scores, calibrator) = fit_best_calibrator(&raw_val_scores, &labels(val_examples), 42);
    let calibrated_test_scores = calibrator.transform_many(&raw_test_scores);
    let calibrated_threshold = optimize_threshold(&labels(val_examples), &calibrated_val_scores);
    let calibrated_metrics =
        summarize_predictions(&labels(test_examples), &calibrated_test_scores, calibrated_threshold);
    let (non_match_threshold, match_threshold) = optimize_abstain_band(
        val_examples,
        &calibrated_val_scores,
        calibrated_threshold.max(0.6),
        calibrated_threshold.min(0.4),
    );
    let matcher = TrainedHybridMatcher {
        model,
        calibrator,
        feature_names: feature_names.to_vec(),
        threshold: calibrated_threshold,
        match_threshold,
        non_match_threshold,
    };
    let (final_scores, decisions) = matcher.score_examples(test_examples, extractor, None);
    let (accepted_precision, accepted_recall, abstain_rate) = abstain_stats(&labels(test_examples), &decisions);

    Ok(HybridRun {
        name: "hybrid_gradient_boosting".to_string(),
        threshold: calibrated_threshold,
        match_threshold,
        non_match_threshold,
        raw_metrics,
        calibrated_metrics,
        raw_brier: brier_score(&labels(test_examples), &raw_test_scores),
        calibrated_brier: brier_score(&labels(test_examples), &calibrated_test_scores),
        accepted_precision,
        accepted_recall,
        abstain_rate,
        raw_scores: raw_test_scores,
        calibrated_scores: final_scores,
        decisions,
    })
}

fn contradiction_veto(example: &PairExample, score: f64, match_threshold: f64) -> bool {
    feature_value(example, "location_conflict") >= 1.0
        && feature_value(example, "org_overlap_count") == 0.0
        && feature_value(example, "shared_domain_count") == 0.0
        && feature_value(example, "topic_overlap_count") <= 1.0
        && (score >= match_threshold || feature_value(example, "name_similarity") >= 0.98)
}

fn abstain_stats(labels: &[i32], decisions: &[String]) -> (f64, f64, f64) {
    let match_indices: Vec<usize> = decisions
        .iter()
        .enumerate()
        .filter_map(|(index, decision)| (decision == "match").then_some(index))
        .collect();
    let abstain_count = decisions.iter().filter(|decision| decision.as_str() == "abstain").count();
    if match_indices.is_empty() {
        return (0.0, 0.0, abstain_count as f64 / decisions.len().max(1) as f64);
    }
    let true_positives = match_indices
        .iter()
        .filter(|index| labels[**index] == 1)
        .count() as f64;
    let accepted_precision = true_positives / match_indices.len() as f64;
    let positives = labels.iter().filter(|label| **label == 1).count().max(1) as f64;
    let accepted_recall = true_positives / positives;
    let abstain_rate = abstain_count as f64 / decisions.len().max(1) as f64;
    (accepted_precision, accepted_recall, abstain_rate)
}

fn optimize_abstain_band(
    examples: &[PairExample],
    calibrated_scores: &[f64],
    match_floor: f64,
    non_match_floor: f64,
) -> (f64, f64) {
    let labels = labels(examples);
    let mut best: Option<(f64, f64, f64, f64)> = None;
    let mut best_thresholds = (non_match_floor, match_floor);
    let lower_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
    let upper_grid = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85];

    for lower in lower_grid {
        for upper in upper_grid {
            if lower >= upper {
                continue;
            }
            let decisions: Vec<String> = calibrated_scores
                .iter()
                .map(|score| {
                    if *score >= upper {
                        "match".to_string()
                    } else if *score <= lower {
                        "non_match".to_string()
                    } else {
                        "abstain".to_string()
                    }
                })
                .collect();
            if !decisions.iter().any(|decision| decision == "match") {
                continue;
            }
            let (accepted_precision, accepted_recall, abstain_rate) = abstain_stats(&labels, &decisions);
            let objective = accepted_precision + 0.25 * accepted_recall - 0.1 * abstain_rate;
            let candidate = (objective, accepted_precision, accepted_recall, abstain_rate);
            if best.as_ref().map(|current| candidate > *current).unwrap_or(true) {
                best = Some(candidate);
                best_thresholds = (lower, upper);
            }
        }
    }

    if best.is_none() {
        return (non_match_floor, match_floor);
    }
    best_thresholds
}

fn fit_best_calibrator(raw_val_scores: &[f64], labels: &[i32], seed: u64) -> (Vec<f64>, Calibrator) {
    let identity_scores = raw_val_scores.to_vec();
    let identity_brier = brier_score(labels, &identity_scores);

    let isotonic = IsotonicCalibrator::fit(raw_val_scores, labels);
    let isotonic_scores = isotonic.transform_many(raw_val_scores);
    let isotonic_brier = brier_score(labels, &isotonic_scores);

    let sigmoid_model = LogisticModel::fit(
        &raw_val_scores.iter().map(|score| vec![*score]).collect::<Vec<_>>(),
        labels,
        seed,
    );
    let sigmoid_scores = sigmoid_model.predict_proba_batch(
        &raw_val_scores.iter().map(|score| vec![*score]).collect::<Vec<_>>(),
    );
    let sigmoid_brier = brier_score(labels, &sigmoid_scores);

    let mut candidates = vec![
        (identity_brier, identity_scores, Calibrator::Identity),
        (isotonic_brier, isotonic_scores, Calibrator::Isotonic(isotonic)),
        (sigmoid_brier, sigmoid_scores, Calibrator::Sigmoid(sigmoid_model)),
    ];
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (_, scores, calibrator) = candidates.remove(0);
    (scores, calibrator)
}

fn feature_rows(examples: &[PairExample]) -> Vec<BTreeMap<String, f64>> {
    examples.iter().map(|example| example.features.clone()).collect()
}

fn project_feature_rows(examples: &[PairExample], feature_names: &[&str]) -> Vec<BTreeMap<String, f64>> {
    examples
        .iter()
        .map(|example| {
            feature_names
                .iter()
                .map(|feature_name| ((*feature_name).to_string(), feature_value(example, feature_name)))
                .collect()
        })
        .collect()
}

fn labels(examples: &[PairExample]) -> Vec<i32> {
    examples.iter().map(|example| example.label).collect()
}

fn feature_value(example: &PairExample, feature_name: &str) -> f64 {
    *example.features.get(feature_name).unwrap_or(&0.0)
}

#[derive(Debug, Clone)]
pub enum Calibrator {
    Identity,
    Sigmoid(LogisticModel),
    Isotonic(IsotonicCalibrator),
}

impl Calibrator {
    pub fn transform_many(&self, scores: &[f64]) -> Vec<f64> {
        match self {
            Self::Identity => scores.to_vec(),
            Self::Sigmoid(model) => model.predict_proba_batch(
                &scores.iter().map(|score| vec![*score]).collect::<Vec<_>>(),
            ),
            Self::Isotonic(calibrator) => calibrator.transform_many(scores),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogisticModel {
    weights: Vec<f64>,
    bias: f64,
    means: Vec<f64>,
    stds: Vec<f64>,
}

impl LogisticModel {
    pub fn fit(features: &[Vec<f64>], labels: &[i32], _seed: u64) -> Self {
        if features.is_empty() {
            return Self {
                weights: vec![],
                bias: 0.0,
                means: vec![],
                stds: vec![],
            };
        }
        let columns = features[0].len();
        let mut means = vec![0.0; columns];
        let mut stds = vec![0.0; columns];
        for row in features {
            for (index, value) in row.iter().enumerate() {
                means[index] += *value;
            }
        }
        for mean in &mut means {
            *mean /= features.len() as f64;
        }
        for row in features {
            for (index, value) in row.iter().enumerate() {
                stds[index] += (value - means[index]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / features.len() as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0;
            }
        }
        let standardized = features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(index, value)| (value - means[index]) / stds[index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let positives = labels.iter().filter(|label| **label == 1).count().max(1) as f64;
        let negatives = labels.iter().filter(|label| **label == 0).count().max(1) as f64;
        let total = labels.len().max(1) as f64;
        let positive_weight = total / (2.0 * positives);
        let negative_weight = total / (2.0 * negatives);

        let mut weights = vec![0.0; columns];
        let mut bias = 0.0;
        let learning_rate = 0.08;
        let l2 = 0.01;
        for _ in 0..2500 {
            let mut grad_w = vec![0.0; columns];
            let mut grad_b = 0.0;
            for (row, label) in standardized.iter().zip(labels.iter().copied()) {
                let linear = dot(&weights, row) + bias;
                let prediction = sigmoid(linear);
                let target = label as f64;
                let sample_weight = if label == 1 { positive_weight } else { negative_weight };
                let error = (prediction - target) * sample_weight;
                for (index, value) in row.iter().enumerate() {
                    grad_w[index] += error * value;
                }
                grad_b += error;
            }
            for (index, weight) in weights.iter_mut().enumerate() {
                let grad = grad_w[index] / total + l2 * *weight;
                *weight -= learning_rate * grad;
            }
            bias -= learning_rate * grad_b / total;
        }

        Self {
            weights,
            bias,
            means,
            stds,
        }
    }

    pub fn predict_proba_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|row| self.predict_proba(row)).collect()
    }

    fn predict_proba(&self, row: &[f64]) -> f64 {
        if self.weights.is_empty() {
            return 0.5;
        }
        let standardized = row
            .iter()
            .enumerate()
            .map(|(index, value)| (value - self.means[index]) / self.stds[index])
            .collect::<Vec<_>>();
        sigmoid(dot(&self.weights, &standardized) + self.bias)
    }
}

#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    blocks: Vec<IsoBlock>,
}

#[derive(Debug, Clone)]
struct IsoBlock {
    min_score: f64,
    max_score: f64,
    avg: f64,
    count: usize,
}

impl IsotonicCalibrator {
    pub fn fit(scores: &[f64], labels: &[i32]) -> Self {
        let mut pairs: Vec<(f64, i32)> = scores.iter().copied().zip(labels.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut blocks: Vec<IsoBlock> = pairs
            .into_iter()
            .map(|(score, label)| IsoBlock {
                min_score: score,
                max_score: score,
                avg: label as f64,
                count: 1,
            })
            .collect();
        let mut index = 0usize;
        while index + 1 < blocks.len() {
            if blocks[index].avg > blocks[index + 1].avg {
                let left = blocks.remove(index);
                let right = blocks.remove(index);
                let total = left.count + right.count;
                let merged = IsoBlock {
                    min_score: left.min_score,
                    max_score: right.max_score,
                    avg: (left.avg * left.count as f64 + right.avg * right.count as f64) / total as f64,
                    count: total,
                };
                blocks.insert(index, merged);
                index = index.saturating_sub(1);
            } else {
                index += 1;
            }
        }
        Self { blocks }
    }

    pub fn transform_many(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|score| self.transform(*score)).collect()
    }

    fn transform(&self, score: f64) -> f64 {
        if self.blocks.is_empty() {
            return score;
        }
        if score <= self.blocks[0].max_score {
            return self.blocks[0].avg;
        }
        for window in self.blocks.windows(2) {
            let left = &window[0];
            let right = &window[1];
            if score <= right.max_score {
                let start = left.max_score;
                let end = right.max_score;
                if (end - start).abs() < f64::EPSILON {
                    return right.avg;
                }
                let ratio = ((score - start) / (end - start)).clamp(0.0, 1.0);
                return left.avg + ratio * (right.avg - left.avg);
            }
        }
        self.blocks.last().unwrap().avg
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn sigmoid(value: f64) -> f64 {
    let clipped = value.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-clipped).exp())
}
