use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSummary {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub average_precision: f64,
}

pub fn expected_calibration_error(labels: &[i32], scores: &[f64], n_bins: usize) -> f64 {
    if labels.is_empty() || scores.is_empty() || labels.len() != scores.len() || n_bins == 0 {
        return 0.0;
    }
    let mut ece = 0.0;
    for bin_index in 0..n_bins {
        let lower = bin_index as f64 / n_bins as f64;
        let upper = (bin_index + 1) as f64 / n_bins as f64;
        let bucket: Vec<(i32, f64)> = labels
            .iter()
            .copied()
            .zip(scores.iter().copied())
            .filter(|(_, score)| {
                if bin_index + 1 == n_bins {
                    *score >= lower && *score <= upper
                } else {
                    *score >= lower && *score < upper
                }
            })
            .collect();
        if bucket.is_empty() {
            continue;
        }
        let confidence = bucket.iter().map(|(_, score)| *score).sum::<f64>() / bucket.len() as f64;
        let accuracy = bucket.iter().map(|(label, _)| *label as f64).sum::<f64>() / bucket.len() as f64;
        ece += bucket.len() as f64 / scores.len() as f64 * (confidence - accuracy).abs();
    }
    ece
}

pub fn summarize_predictions(labels: &[i32], scores: &[f64], threshold: f64) -> MetricSummary {
    let predictions: Vec<i32> = scores.iter().map(|score| (*score >= threshold) as i32).collect();
    let tp = labels
        .iter()
        .zip(predictions.iter())
        .filter(|(label, pred)| **label == 1 && **pred == 1)
        .count() as f64;
    let fp = labels
        .iter()
        .zip(predictions.iter())
        .filter(|(label, pred)| **label == 0 && **pred == 1)
        .count() as f64;
    let fn_count = labels
        .iter()
        .zip(predictions.iter())
        .filter(|(label, pred)| **label == 1 && **pred == 0)
        .count() as f64;

    let precision = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
    let recall = if tp + fn_count == 0.0 { 0.0 } else { tp / (tp + fn_count) };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let average_precision = average_precision_score(labels, scores);
    MetricSummary {
        precision,
        recall,
        f1,
        average_precision,
    }
}

pub fn threshold_sweep(labels: &[i32], scores: &[f64], n_points: usize) -> Vec<serde_json::Value> {
    if labels.is_empty() || scores.is_empty() || n_points == 0 {
        return vec![];
    }
    let mut lo = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut hi = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (lo - hi).abs() < f64::EPSILON {
        lo = 0.0;
        hi = 1.0;
    }
    (0..n_points)
        .map(|index| {
            let threshold = if n_points == 1 {
                lo
            } else {
                lo + (hi - lo) * index as f64 / (n_points - 1) as f64
            };
            let summary = summarize_predictions(labels, scores, threshold);
            serde_json::json!({
                "threshold": ((threshold * 10000.0).round() / 10000.0),
                "precision": summary.precision,
                "recall": summary.recall,
                "f1": summary.f1,
            })
        })
        .collect()
}

pub fn optimize_threshold(labels: &[i32], scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.5;
    }
    let mut candidate_thresholds: Vec<f64> = scores.iter().cloned().map(|score| score.clamp(0.0, 1.0)).collect();
    candidate_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    candidate_thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    candidate_thresholds.extend([0.35, 0.5, 0.65, 0.8]);
    let mut best_threshold = 0.5;
    let mut best_f1 = -1.0;
    for threshold in candidate_thresholds {
        let summary = summarize_predictions(labels, scores, threshold);
        if summary.f1 > best_f1 {
            best_f1 = summary.f1;
            best_threshold = threshold;
        }
    }
    best_threshold
}

pub fn average_precision_score(labels: &[i32], scores: &[f64]) -> f64 {
    if labels.is_empty() || scores.is_empty() || labels.len() != scores.len() {
        return 0.0;
    }
    let positives = labels.iter().filter(|label| **label == 1).count();
    if positives == 0 || labels.iter().all(|label| *label == labels[0]) {
        return 0.0;
    }

    let mut pairs: Vec<(f64, i32)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_recall = 0.0;
    let mut ap = 0.0;
    for (_, label) in pairs {
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let recall = tp / positives as f64;
        let precision = tp / (tp + fp);
        ap += precision * (recall - prev_recall);
        prev_recall = recall;
    }
    ap
}

pub fn brier_score(labels: &[i32], scores: &[f64]) -> f64 {
    if labels.is_empty() || scores.is_empty() || labels.len() != scores.len() {
        return 0.0;
    }
    labels
        .iter()
        .zip(scores.iter())
        .map(|(label, score)| (*score - *label as f64).powi(2))
        .sum::<f64>()
        / labels.len() as f64
}
