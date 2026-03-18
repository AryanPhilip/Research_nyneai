use std::collections::HashSet;

use url::Url;

pub fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn normalize_text(value: &str) -> String {
    collapse_whitespace(&value.to_lowercase())
}

pub fn token_set(values: &[String]) -> HashSet<String> {
    let mut tokens = HashSet::new();
    for value in values {
        for token in normalize_text(value).split_whitespace() {
            if !token.is_empty() {
                tokens.insert(token.to_string());
            }
        }
    }
    tokens
}

pub fn jaccard(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 0.0;
    }
    let union = left.union(right).count();
    if union == 0 {
        return 0.0;
    }
    left.intersection(right).count() as f64 / union as f64
}

fn longest_common_substring(
    left: &[char],
    right: &[char],
    left_offset: usize,
    right_offset: usize,
) -> Option<(usize, usize, usize)> {
    let mut best_left = 0usize;
    let mut best_right = 0usize;
    let mut best_len = 0usize;
    let mut table = vec![0usize; right.len() + 1];

    for (i, lch) in left.iter().enumerate() {
        for j in (0..right.len()).rev() {
            if *lch == right[j] {
                table[j + 1] = table[j] + 1;
                if table[j + 1] > best_len {
                    best_len = table[j + 1];
                    best_left = i + 1 - best_len;
                    best_right = j + 1 - best_len;
                }
            } else {
                table[j + 1] = 0;
            }
        }
    }

    if best_len == 0 {
        None
    } else {
        Some((left_offset + best_left, right_offset + best_right, best_len))
    }
}

fn matching_blocks(left: &[char], right: &[char], left_offset: usize, right_offset: usize) -> usize {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let Some((left_start, right_start, length)) =
        longest_common_substring(left, right, left_offset, right_offset)
    else {
        return 0;
    };

    let left_local = left_start - left_offset;
    let right_local = right_start - right_offset;

    let before = matching_blocks(
        &left[..left_local],
        &right[..right_local],
        left_offset,
        right_offset,
    );
    let after = matching_blocks(
        &left[left_local + length..],
        &right[right_local + length..],
        left_start + length,
        right_start + length,
    );
    before + length + after
}

pub fn sequence_similarity(left: &str, right: &str) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let left_chars: Vec<char> = normalize_text(left).chars().collect();
    let right_chars: Vec<char> = normalize_text(right).chars().collect();
    if left_chars.is_empty() || right_chars.is_empty() {
        return 0.0;
    }
    let matches = matching_blocks(&left_chars, &right_chars, 0, 0);
    (2.0 * matches as f64) / (left_chars.len() as f64 + right_chars.len() as f64)
}

pub fn sigmoid(value: f64) -> f64 {
    let clamped = value.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-clamped).exp())
}

pub fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

pub fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    let left_norm = l2_norm(left);
    let right_norm = l2_norm(right);
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot(left, right) / (left_norm * right_norm)
}

pub fn domain_without_www(url: &Url) -> Option<String> {
    url.domain().map(|domain| domain.trim_start_matches("www.").to_string())
}
