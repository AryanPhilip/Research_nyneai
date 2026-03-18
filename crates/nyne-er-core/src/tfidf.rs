use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default)]
pub struct TfidfVectorizer {
    vocab: BTreeMap<String, usize>,
    idf: Vec<f64>,
}

impl TfidfVectorizer {
    pub fn fit(texts: &[String]) -> Self {
        let mut doc_freq: BTreeMap<String, usize> = BTreeMap::new();
        for text in texts {
            let seen = tokenize_ngrams(text);
            for token in seen {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }

        let mut vocab = BTreeMap::new();
        let mut idf = Vec::new();
        let n_docs = texts.len() as f64;
        for (index, (token, freq)) in doc_freq.into_iter().enumerate() {
            vocab.insert(token, index);
            idf.push(((1.0 + n_docs) / (1.0 + freq as f64)).ln() + 1.0);
        }
        Self { vocab, idf }
    }

    pub fn fit_transform(texts: &[String]) -> (Self, Vec<Vec<f64>>) {
        let vectorizer = Self::fit(texts);
        let matrix = texts.iter().map(|text| vectorizer.transform(text)).collect();
        (vectorizer, matrix)
    }

    pub fn transform(&self, text: &str) -> Vec<f64> {
        let mut vector = vec![0.0; self.vocab.len()];
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for token in tokenize_ngrams_multiset(text) {
            if let Some(index) = self.vocab.get(&token) {
                *counts.entry(*index).or_insert(0) += 1;
            }
        }
        for (index, count) in counts {
            vector[index] = count as f64 * self.idf[index];
        }
        vector
    }
}

pub fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (l, r) in left.iter().zip(right.iter()) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

fn tokenize_ngrams(text: &str) -> BTreeSet<String> {
    tokenize(text).into_iter().collect()
}

fn tokenize_ngrams_multiset(text: &str) -> Vec<String> {
    tokenize(text)
}

fn tokenize(text: &str) -> Vec<String> {
    let tokens: Vec<String> = text
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect();
    let mut features = tokens.clone();
    for window in tokens.windows(2) {
        features.push(format!("{} {}", window[0], window[1]));
    }
    features
}

