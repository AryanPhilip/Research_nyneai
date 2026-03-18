pub fn normalize_text(value: &str) -> String {
    value.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn sequence_similarity(left: &str, right: &str) -> f64 {
    let left = normalize_text(left);
    let right = normalize_text(right);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let left_chars: Vec<char> = left.chars().collect();
    let right_chars: Vec<char> = right.chars().collect();
    let matches = gestalt_match_count(&left_chars, &right_chars) as f64;
    (2.0 * matches) / (left_chars.len() as f64 + right_chars.len() as f64)
}

fn gestalt_match_count(left: &[char], right: &[char]) -> usize {
    let Some((left_start, right_start, size)) = longest_common_substring(left, right) else {
        return 0;
    };
    size
        + gestalt_match_count(&left[..left_start], &right[..right_start])
        + gestalt_match_count(&left[left_start + size..], &right[right_start + size..])
}

fn longest_common_substring(left: &[char], right: &[char]) -> Option<(usize, usize, usize)> {
    let mut best = (0usize, 0usize, 0usize);
    let mut table = vec![0usize; right.len() + 1];
    for (left_index, left_char) in left.iter().enumerate() {
        let mut prev = 0usize;
        for (right_index, right_char) in right.iter().enumerate() {
            let next_prev = table[right_index + 1];
            if left_char == right_char {
                table[right_index + 1] = prev + 1;
                if table[right_index + 1] > best.2 {
                    best = (
                        left_index + 1 - table[right_index + 1],
                        right_index + 1 - table[right_index + 1],
                        table[right_index + 1],
                    );
                }
            } else {
                table[right_index + 1] = 0;
            }
            prev = next_prev;
        }
    }
    (best.2 > 0).then_some(best)
}

