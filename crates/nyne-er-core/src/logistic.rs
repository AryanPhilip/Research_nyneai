use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::util::sigmoid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LogisticFitOptions {
    pub iterations: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub balanced: bool,
}

impl Default for LogisticFitOptions {
    fn default() -> Self {
        Self {
            iterations: 5_000,
            learning_rate: 0.1,
            l2: 1e-4,
            balanced: true,
        }
    }
}

impl LogisticModel {
    pub fn fit(matrix: &Array2<f64>, labels: &[usize], options: LogisticFitOptions) -> Self {
        let n_features = matrix.ncols();
        let mut weights = vec![0.0; n_features];
        let mut bias = 0.0;

        let positive_count = labels.iter().filter(|label| **label == 1).count() as f64;
        let negative_count = labels.len() as f64 - positive_count;
        let pos_weight = if options.balanced && positive_count > 0.0 {
            labels.len() as f64 / (2.0 * positive_count)
        } else {
            1.0
        };
        let neg_weight = if options.balanced && negative_count > 0.0 {
            labels.len() as f64 / (2.0 * negative_count)
        } else {
            1.0
        };

        for _ in 0..options.iterations {
            let mut grad_w = vec![0.0; n_features];
            let mut grad_b = 0.0;
            for (row, label) in matrix.axis_iter(Axis(0)).zip(labels.iter().copied()) {
                let linear = row
                    .iter()
                    .zip(weights.iter())
                    .map(|(value, weight)| value * weight)
                    .sum::<f64>()
                    + bias;
                let prediction = sigmoid(linear);
                let target = label as f64;
                let weight = if label == 1 { pos_weight } else { neg_weight };
                let error = (prediction - target) * weight;
                for (index, value) in row.iter().enumerate() {
                    grad_w[index] += error * value;
                }
                grad_b += error;
            }

            let denom = labels.len() as f64;
            for (index, value) in grad_w.iter_mut().enumerate() {
                *value = (*value / denom) + (options.l2 * weights[index]);
                weights[index] -= options.learning_rate * *value;
            }
            bias -= options.learning_rate * (grad_b / denom);
        }

        Self { weights, bias }
    }

    pub fn predict_proba(&self, matrix: &Array2<f64>) -> Vec<f64> {
        matrix
            .axis_iter(Axis(0))
            .map(|row| {
                let linear = row
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(value, weight)| value * weight)
                    .sum::<f64>()
                    + self.bias;
                sigmoid(linear)
            })
            .collect()
    }

    pub fn predict_single(&self, values: &[f64]) -> f64 {
        let linear = values
            .iter()
            .zip(self.weights.iter())
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
            + self.bias;
        sigmoid(linear)
    }
}

pub fn matrix_from_scores(scores: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((scores.len(), 1), scores.to_vec()).expect("score matrix shape must be valid")
}

pub fn labels_to_array(labels: &[usize]) -> Array1<f64> {
    Array1::from(labels.iter().map(|label| *label as f64).collect::<Vec<_>>())
}
