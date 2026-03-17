"""LLM-based adjudication and explanation for entity resolution pairs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from nyne_er_lab.features.dataset import PairExample
from nyne_er_lab.schemas import ProfileRecord


@dataclass
class LLMResult:
    verdict: str  # "match" | "non_match"
    confidence: float
    reasoning: str


def _profile_summary(profile: ProfileRecord) -> str:
    orgs = ", ".join(o.name for o in profile.organizations[:4]) if profile.organizations else "none"
    locs = ", ".join(profile.locations[:3]) if profile.locations else "none"
    topics = ", ".join(profile.topics[:6]) if profile.topics else "none"
    return (
        f"Name: {profile.display_name}\n"
        f"Source: {profile.source_type}\n"
        f"Headline: {profile.headline or 'none'}\n"
        f"Orgs: {orgs}\n"
        f"Locations: {locs}\n"
        f"Topics: {topics}\n"
        f"Bio excerpt: {(profile.bio_text or '')[:300]}"
    )


def _build_adjudication_prompt(
    left: ProfileRecord,
    right: ProfileRecord,
    features: dict[str, float],
    model_score: float,
    model_decision: str,
) -> str:
    feature_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in features.items())
    return (
        "You are an entity resolution expert. Two online profiles have been compared "
        "by a statistical model. Determine whether they refer to the same real person.\n\n"
        f"--- Profile A ---\n{_profile_summary(left)}\n\n"
        f"--- Profile B ---\n{_profile_summary(right)}\n\n"
        f"--- Model Features ---\n{feature_lines}\n\n"
        f"Model score: {model_score:.4f} (decision: {model_decision})\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"verdict": "match" or "non_match", "confidence": 0.0-1.0, "reasoning": "1-2 sentences"}'
    )


def _build_explanation_prompt(
    left: ProfileRecord,
    right: ProfileRecord,
    features: dict[str, float],
    decision: str,
    score: float,
) -> str:
    feature_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in features.items())
    return (
        "You are an entity resolution expert. Explain in 2-3 clear sentences why "
        f"these two profiles were classified as '{decision}' (score: {score:.3f}). "
        "Be specific about which evidence supports or contradicts the match.\n\n"
        f"--- Profile A ---\n{_profile_summary(left)}\n\n"
        f"--- Profile B ---\n{_profile_summary(right)}\n\n"
        f"--- Feature Scores ---\n{feature_lines}\n\n"
        "Respond with ONLY a plain text explanation (no JSON)."
    )


def _get_api_key() -> tuple[str, str] | None:
    """Return (provider, key) or None."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return ("openai", key)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return ("anthropic", key)
    return None


def _call_openai(prompt: str, api_key: str) -> str:
    import httpx

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 300,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt: str, api_key: str) -> str:
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-3-5-haiku-latest",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def _call_llm(prompt: str) -> str | None:
    creds = _get_api_key()
    if creds is None:
        return None
    provider, key = creds
    try:
        if provider == "openai":
            return _call_openai(prompt, key)
        return _call_anthropic(prompt, key)
    except Exception:
        return None


def llm_available() -> bool:
    return _get_api_key() is not None


def adjudicate_pair(
    left: ProfileRecord,
    right: ProfileRecord,
    features: dict[str, float],
    model_score: float,
    model_decision: str,
) -> LLMResult | None:
    prompt = _build_adjudication_prompt(left, right, features, model_score, model_decision)
    raw = _call_llm(prompt)
    if raw is None:
        return None
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
        return LLMResult(
            verdict=data.get("verdict", model_decision),
            confidence=float(data.get("confidence", model_score)),
            reasoning=data.get("reasoning", "No reasoning provided."),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return LLMResult(verdict=model_decision, confidence=model_score, reasoning=raw[:500])


def explain_pair(
    left: ProfileRecord,
    right: ProfileRecord,
    features: dict[str, float],
    decision: str,
    score: float,
) -> str | None:
    prompt = _build_explanation_prompt(left, right, features, decision, score)
    return _call_llm(prompt)
