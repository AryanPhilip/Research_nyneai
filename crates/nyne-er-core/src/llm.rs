use crate::schemas::ProfileRecord;
use anyhow::Result;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::env;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMResult {
    pub verdict: String,
    pub confidence: f64,
    pub reasoning: String,
}

pub fn llm_available() -> bool {
    api_key().is_some()
}

pub fn adjudicate_pair(
    left: &ProfileRecord,
    right: &ProfileRecord,
    features: &BTreeMap<String, f64>,
    model_score: f64,
    model_decision: &str,
) -> Option<LLMResult> {
    let raw = call_llm(&build_adjudication_prompt(
        left,
        right,
        features,
        model_score,
        model_decision,
    ))?;
    parse_llm_result(&raw, model_score, model_decision)
}

pub fn explain_pair(
    left: &ProfileRecord,
    right: &ProfileRecord,
    features: &BTreeMap<String, f64>,
    decision: &str,
    score: f64,
) -> Option<String> {
    call_llm(&build_explanation_prompt(left, right, features, decision, score))
}

fn api_key() -> Option<(&'static str, String)> {
    env::var("OPENAI_API_KEY")
        .ok()
        .map(|key| ("openai", key))
        .or_else(|| env::var("ANTHROPIC_API_KEY").ok().map(|key| ("anthropic", key)))
}

fn call_llm(prompt: &str) -> Option<String> {
    let (provider, key) = api_key()?;
    let client = Client::builder().build().ok()?;
    match provider {
        "openai" => call_openai(&client, prompt, &key).ok(),
        "anthropic" => call_anthropic(&client, prompt, &key).ok(),
        _ => None,
    }
}

fn call_openai(client: &Client, prompt: &str, api_key: &str) -> Result<String> {
    let response: serde_json::Value = client
        .post("https://api.openai.com/v1/chat/completions")
        .header(AUTHORIZATION, format!("Bearer {api_key}"))
        .header(CONTENT_TYPE, "application/json")
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 300,
        }))
        .send()?
        .json()?;
    Ok(response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_string())
}

fn call_anthropic(client: &Client, prompt: &str, api_key: &str) -> Result<String> {
    let response: serde_json::Value = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header(CONTENT_TYPE, "application/json")
        .json(&json!({
            "model": "claude-3-5-haiku-latest",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }))
        .send()?
        .json()?;
    Ok(response["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_string())
}

fn parse_llm_result(raw: &str, model_score: f64, model_decision: &str) -> Option<LLMResult> {
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str::<serde_json::Value>(cleaned)
        .ok()
        .map(|value| LLMResult {
            verdict: value["verdict"]
                .as_str()
                .unwrap_or(model_decision)
                .to_string(),
            confidence: value["confidence"].as_f64().unwrap_or(model_score),
            reasoning: value["reasoning"]
                .as_str()
                .unwrap_or("No reasoning provided.")
                .to_string(),
        })
        .or_else(|| {
            Some(LLMResult {
                verdict: model_decision.to_string(),
                confidence: model_score,
                reasoning: raw.chars().take(500).collect(),
            })
        })
}

fn build_profile_summary(profile: &ProfileRecord) -> String {
    let orgs = if profile.organizations.is_empty() {
        "none".to_string()
    } else {
        profile
            .organizations
            .iter()
            .take(4)
            .map(|org| org.name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let locations = if profile.locations.is_empty() {
        "none".to_string()
    } else {
        profile
            .locations
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    };
    let topics = if profile.topics.is_empty() {
        "none".to_string()
    } else {
        profile
            .topics
            .iter()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "Name: {}\nSource: {}\nHeadline: {}\nOrgs: {}\nLocations: {}\nTopics: {}\nBio excerpt: {}",
        profile.display_name,
        profile.source_type.as_str(),
        profile.headline.clone().unwrap_or_else(|| "none".to_string()),
        orgs,
        locations,
        topics,
        profile.bio_text.chars().take(300).collect::<String>()
    )
}

fn build_adjudication_prompt(
    left: &ProfileRecord,
    right: &ProfileRecord,
    features: &BTreeMap<String, f64>,
    model_score: f64,
    model_decision: &str,
) -> String {
    let feature_lines = features
        .iter()
        .map(|(key, value)| format!("  {key}: {value:.4}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are an entity resolution expert. Two online profiles have been compared by a statistical model. Determine whether they refer to the same real person.\n\n--- Profile A ---\n{}\n\n--- Profile B ---\n{}\n\n--- Model Features ---\n{}\n\nModel score: {:.4} (decision: {})\n\nRespond with ONLY a JSON object:\n{{\"verdict\": \"match\" or \"non_match\", \"confidence\": 0.0-1.0, \"reasoning\": \"1-2 sentences\"}}",
        build_profile_summary(left),
        build_profile_summary(right),
        feature_lines,
        model_score,
        model_decision
    )
}

fn build_explanation_prompt(
    left: &ProfileRecord,
    right: &ProfileRecord,
    features: &BTreeMap<String, f64>,
    decision: &str,
    score: f64,
) -> String {
    let feature_lines = features
        .iter()
        .map(|(key, value)| format!("  {key}: {value:.4}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are an entity resolution expert. Explain in 2-3 clear sentences why these two profiles were classified as '{}' (score: {:.3}). Be specific about which evidence supports or contradicts the match.\n\n--- Profile A ---\n{}\n\n--- Profile B ---\n{}\n\n--- Feature Scores ---\n{}\n\nRespond with ONLY a plain text explanation (no JSON).",
        decision,
        score,
        build_profile_summary(left),
        build_profile_summary(right),
        feature_lines
    )
}
