"""Build demo and report artifacts from the entity-resolution pipeline."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from nyne_er_lab.cluster import resolve_identities
from nyne_er_lab.datasets import load_benchmark_profiles, load_raw_pages
from nyne_er_lab.eval import bcubed_f1
from nyne_er_lab.features import (
    PairFeatureExtractor,
    assign_profile_splits,
    build_examples_for_profiles,
    build_pair_examples,
    build_split_candidates,
    profiles_for_split,
)
from nyne_er_lab.ingest import parse_raw_pages
from nyne_er_lab.models import (
    run_embedding_baseline,
    run_feature_ablations,
    run_hybrid_matcher,
    run_lexical_baseline,
    run_name_baseline,
    train_hybrid_matcher,
)


def _metric_payload(run_name: str, f1: float, precision: float, recall: float, average_precision: float) -> dict[str, float | str]:
    return {
        "name": run_name,
        "f1": round(f1, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "average_precision": round(average_precision, 3),
    }


def _benchmark_context() -> dict[str, object]:
    profiles = load_benchmark_profiles()
    split_map = assign_profile_splits(profiles)
    extractor = PairFeatureExtractor().fit(profiles_for_split(profiles, split_map, "train"))

    train_examples = build_pair_examples(
        profiles,
        build_split_candidates(profiles, split_map, "train"),
        extractor,
        split_map,
        "train",
    )
    val_examples = build_pair_examples(
        profiles,
        build_split_candidates(profiles, split_map, "val"),
        extractor,
        split_map,
        "val",
    )
    test_examples = build_pair_examples(
        profiles,
        build_split_candidates(profiles, split_map, "test"),
        extractor,
        split_map,
        "test",
    )

    name_run = run_name_baseline(val_examples, test_examples)
    embedding_run = run_embedding_baseline(val_examples, test_examples)
    lexical_run = run_lexical_baseline(train_examples, val_examples, test_examples, extractor)
    hybrid_run = run_hybrid_matcher(train_examples, val_examples, test_examples, extractor)
    matcher = train_hybrid_matcher(train_examples, val_examples, extractor)
    identities, resolved_pairs = resolve_identities(
        profiles,
        build_examples_for_profiles(profiles, extractor),
        matcher,
        extractor=extractor,
    )
    ablations = run_feature_ablations(train_examples, val_examples, test_examples, extractor)

    return {
        "profiles": profiles,
        "extractor": extractor,
        "matcher": matcher,
        "metrics": [
            _metric_payload("fuzzy_name", name_run.metrics.f1, name_run.metrics.precision, name_run.metrics.recall, name_run.metrics.average_precision),
            _metric_payload("embedding_only", embedding_run.metrics.f1, embedding_run.metrics.precision, embedding_run.metrics.recall, embedding_run.metrics.average_precision),
            _metric_payload("lexical_baseline", lexical_run.metrics.f1, lexical_run.metrics.precision, lexical_run.metrics.recall, lexical_run.metrics.average_precision),
            _metric_payload("hybrid", hybrid_run.calibrated_metrics.f1, hybrid_run.calibrated_metrics.precision, hybrid_run.calibrated_metrics.recall, hybrid_run.calibrated_metrics.average_precision),
        ],
        "hybrid": hybrid_run,
        "cluster_f1": round(bcubed_f1(profiles, identities), 3),
        "ablations": [
            {"name": ablation.name, "f1": round(ablation.metrics.f1, 3), "average_precision": round(ablation.metrics.average_precision, 3)}
            for ablation in ablations
        ],
        "confusion_slices": {
            "test_examples": len(test_examples),
            "forced_precision": round(hybrid_run.calibrated_metrics.precision, 3),
            "accepted_precision": round(hybrid_run.accepted_precision, 3),
            "accepted_recall": round(hybrid_run.accepted_recall, 3),
            "abstain_rate": round(hybrid_run.abstain_rate, 3),
        },
        "resolved_pairs": resolved_pairs,
        "identities": identities,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "test_examples": test_examples,
        "split_map": split_map,
    }


def _demo_cases(extractor, matcher) -> list[dict[str, object]]:
    raw_profiles = parse_raw_pages(load_raw_pages())
    raw_examples = build_examples_for_profiles(raw_profiles, extractor)
    identities, resolved_pairs = resolve_identities(raw_profiles, raw_examples, matcher, extractor=extractor)
    profile_lookup = {profile.profile_id: profile for profile in raw_profiles}

    match_cases = []
    seen_match_titles = set()
    for pair in resolved_pairs:
        if pair.decision != "match":
            continue
        left = profile_lookup[pair.left_profile_id]
        title_key = left.display_name
        if title_key in seen_match_titles:
            continue
        seen_match_titles.add(title_key)
        match_cases.append(pair)
        if len(match_cases) == 3:
            break
    abstain_cases = [pair for pair in resolved_pairs if pair.decision == "abstain"][:2]
    non_match_cases = [pair for pair in resolved_pairs if pair.decision == "non_match"][:2]
    curated_pairs = match_cases + abstain_cases + non_match_cases

    cases: list[dict[str, object]] = []
    for pair in curated_pairs:
        left = profile_lookup[pair.left_profile_id]
        right = profile_lookup[pair.right_profile_id]
        cases.append(
            {
                "title": f"{left.display_name} vs {right.display_name}",
                "decision": pair.decision,
                "score": round(pair.score, 3),
                "profiles": [left.profile_id, right.profile_id],
                "left_source": left.source_type,
                "right_source": right.source_type,
                "explanation": pair.evidence_card.final_explanation,
                "reason_codes": pair.evidence_card.reason_codes,
            }
        )

    return cases


def _render_metric_rows(metrics: list[dict[str, object]]) -> str:
    rows = []
    for metric in metrics:
        width = int(float(metric["f1"]) * 100)
        rows.append(
            f"""
            <tr>
              <td>{escape(str(metric['name']))}</td>
              <td>{metric['precision']}</td>
              <td>{metric['recall']}</td>
              <td>{metric['f1']}</td>
              <td>{metric['average_precision']}</td>
              <td><div class="bar"><span style="width:{width}%"></span></div></td>
            </tr>
            """
        )
    return "\n".join(rows)


def _render_case_cards(cases: list[dict[str, object]]) -> str:
    cards = []
    for case in cases:
        cards.append(
            f"""
            <article class="case {escape(str(case['decision']))}">
              <h3>{escape(str(case['title']))}</h3>
              <p><strong>Decision:</strong> {escape(str(case['decision']))} at {case['score']}</p>
              <p><strong>Sources:</strong> {escape(str(case['left_source']))} + {escape(str(case['right_source']))}</p>
              <p>{escape(str(case['explanation']))}</p>
              <p><strong>Reason codes:</strong> {escape(', '.join(case['reason_codes']))}</p>
            </article>
            """
        )
    return "\n".join(cards)


def _render_identity_cards(identities) -> str:
    cards = []
    for identity in identities[:6]:
        cards.append(
            f"""
            <article class="identity">
              <h3>{escape(identity.canonical_name)}</h3>
              <p><strong>Members:</strong> {len(identity.member_profile_ids)} | <strong>Confidence:</strong> {identity.confidence_band}</p>
              <p>{escape(identity.summary)}</p>
              <p><strong>Profiles:</strong> {escape(', '.join(identity.member_profile_ids))}</p>
            </article>
            """
        )
    return "\n".join(cards)


def _render_demo_html(metrics: list[dict[str, object]], confusion_slices: dict[str, object], cases: list[dict[str, object]], identities) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Nyne Entity Resolution Lab</title>
    <style>
      :root {{
        --bg: #f6f3ea;
        --panel: #fffdf7;
        --ink: #191716;
        --muted: #6c625a;
        --accent: #0e6ba8;
        --good: #2a7f62;
        --warn: #b57f00;
        --bad: #a63d40;
        --border: #ddd4c7;
      }}
      body {{
        margin: 0;
        font-family: Georgia, 'Iowan Old Style', serif;
        color: var(--ink);
        background: radial-gradient(circle at top left, #fff7df, var(--bg) 45%), var(--bg);
      }}
      main {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 40px 20px 80px;
      }}
      h1, h2, h3 {{
        margin: 0 0 12px;
      }}
      p {{
        line-height: 1.55;
      }}
      .hero {{
        padding: 28px;
        border: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(14,107,168,0.08), rgba(255,255,255,0.9));
        border-radius: 18px;
        margin-bottom: 24px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 16px;
      }}
      .panel, .case, .identity {{
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 16px;
        padding: 18px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        text-align: left;
        padding: 10px 8px;
        border-bottom: 1px solid var(--border);
      }}
      .bar {{
        width: 100%;
        height: 10px;
        background: #eee5d6;
        border-radius: 999px;
        overflow: hidden;
      }}
      .bar span {{
        display: block;
        height: 100%;
        background: linear-gradient(90deg, #0e6ba8, #2a7f62);
      }}
      .match {{ border-left: 6px solid var(--good); }}
      .abstain {{ border-left: 6px solid var(--warn); }}
      .non_match {{ border-left: 6px solid var(--bad); }}
      .eyebrow {{
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 12px;
        color: var(--muted);
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <p class="eyebrow">Nyne-Oriented ML Proof of Work</p>
        <h1>Interpretable Public-Web Entity Resolution Lab</h1>
        <p>This demo packages a benchmarked pairwise matcher, conservative abstention policy, identity clustering layer, and explanation cards into one reproducible artifact set.</p>
      </section>

      <section class="panel">
        <h2>Benchmark Snapshot</h2>
        <table>
          <thead>
            <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>AP</th><th>Visual</th></tr>
          </thead>
          <tbody>
            {_render_metric_rows(metrics)}
          </tbody>
        </table>
      </section>

      <section class="grid" style="margin-top: 18px;">
        <div class="panel">
          <h2>Calibration</h2>
          <p>Accepted precision: <strong>{confusion_slices['accepted_precision']}</strong></p>
          <p>Accepted recall: <strong>{confusion_slices['accepted_recall']}</strong></p>
          <p>Abstain rate: <strong>{confusion_slices['abstain_rate']}</strong></p>
        </div>
        <div class="panel">
          <h2>Confusion Slice</h2>
          <p>Held-out test examples: <strong>{confusion_slices['test_examples']}</strong></p>
          <p>Forced precision: <strong>{confusion_slices['forced_precision']}</strong></p>
          <p>This project prefers abstention over aggressive same-name merges.</p>
        </div>
      </section>

      <section style="margin-top: 24px;">
        <h2>Curated Cases</h2>
        <div class="grid">
          {_render_case_cards(cases)}
        </div>
      </section>

      <section style="margin-top: 24px;">
        <h2>Resolved Identity Gallery</h2>
        <div class="grid">
          {_render_identity_cards(identities)}
        </div>
      </section>
    </main>
  </body>
</html>
"""


def _render_blog_markdown(metrics: list[dict[str, object]], cluster_f1: float, ablations: list[dict[str, object]], cases: list[dict[str, object]]) -> str:
    metric_lines = "\n".join(
        f"| {metric['name']} | {metric['precision']} | {metric['recall']} | {metric['f1']} | {metric['average_precision']} |"
        for metric in metrics
    )
    ablation_lines = "\n".join(
        f"| {ablation['name']} | {ablation['f1']} | {ablation['average_precision']} |"
        for ablation in ablations
    )
    case_lines = "\n".join(
        f"- `{case['decision']}`: {case['title']} ({case['score']}) with reason codes `{', '.join(case['reason_codes'])}`"
        for case in cases
    )
    return f"""# Interpretable Public-Web Entity Resolution Lab

## Problem
AI agents need more than a name string. They need a way to connect fragmented public professional profiles into a unified identity while avoiding false merges on same-name collisions.

## Benchmark
The benchmark combines curated public-profile-style records for AI/ML builders with hard negatives, including exact-name collisions. Splits are person-disjoint.

| Model | Precision | Recall | F1 | AP |
| --- | --- | --- | --- | --- |
{metric_lines}

Cluster B-cubed F1: **{cluster_f1}**

## System
1. Parse public-profile snapshots into typed `ProfileRecord`s.
2. Generate candidate pairs with rule-based blocking plus TF-IDF neighbor retrieval.
3. Score pairs with a full-feature logistic matcher.
4. Calibrate scores and abstain on contradictory same-name collisions.
5. Cluster accepted matches into canonical identities and emit explanation cards.

## Calibration and Abstention
The calibrated matcher improves confidence quality and preserves a conservative abstention path for exact-name collisions with conflicting org, domain, topic, and location evidence.

## Ablations
| Ablation | F1 | AP |
| --- | --- | --- |
{ablation_lines}

The structured feature block is doing meaningful work. Removing it collapses performance on held-out same-name collisions.

## Curated Cases
{case_lines}

## Failure Modes
- Cross-domain topic overlap still creates noisy candidates in blocking.
- The current corpus is intentionally small and curated; scaling would need richer raw-page coverage and more adversarial negatives.
- Embedding quality is limited by a local TF-IDF representation; a production system would swap in a stronger embedder while preserving the same evaluation harness.

## What I Would Build Next
- Expand the public-page corpus beyond the current seed set.
- Add richer temporal contradiction handling and alias discovery.
- Replace TF-IDF retrieval with a stronger semantic embedder while keeping the same benchmark and abstention policy.
"""


def build_demo_artifacts(output_dir: str | Path = "reports") -> dict[str, str]:
    """Generate reproducible demo and report artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark = _benchmark_context()
    cases = _demo_cases(benchmark["extractor"], benchmark["matcher"])

    metrics_json_path = output_path / "benchmark_metrics.json"
    demo_html_path = output_path / "demo.html"
    blog_md_path = output_path / "blog.md"

    metrics_payload = {
        "models": benchmark["metrics"],
        "cluster_f1": benchmark["cluster_f1"],
        "ablations": benchmark["ablations"],
        "confusion_slices": benchmark["confusion_slices"],
        "cases": cases,
    }
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2))
    demo_html_path.write_text(
        _render_demo_html(
            benchmark["metrics"],
            benchmark["confusion_slices"],
            cases,
            benchmark["identities"],
        )
    )
    blog_md_path.write_text(
        _render_blog_markdown(
            benchmark["metrics"],
            benchmark["cluster_f1"],
            benchmark["ablations"],
            cases,
        )
    )

    return {
        "metrics_json": str(metrics_json_path),
        "demo_html": str(demo_html_path),
        "blog_md": str(blog_md_path),
    }
