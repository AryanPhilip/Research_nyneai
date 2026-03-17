"""Build demo and report artifacts from the honest benchmark pipeline."""

from __future__ import annotations

import json
import base64
from html import escape
from pathlib import Path

from nyne_er_lab.eval.benchmark import run_benchmark


def _benchmark_context() -> dict[str, object]:
    report = run_benchmark("real_curated_core", protocol="grouped_cv", seeds=[7, 11, 17])
    return {
        "report": report,
        "profiles": report.profiles,
        "extractor": report.extractor,
        "matcher": report.matcher,
        "metrics": report.model_metrics,
        "headline_metrics": report.headline_metrics,
        "hybrid": next(metric for metric in report.model_metrics if metric["name"] == "hybrid"),
        "cluster_f1": report.cluster_metrics["bcubed_f1"],
        "ablations": report.ablations,
        "resolved_pairs": report.resolved_pairs,
        "identities": report.identities,
        "train_examples": report.train_examples,
        "val_examples": report.val_examples,
        "test_examples": report.test_examples,
        "blocking_stats": report.blocking_stats,
        "failure_slices": report.failure_slices,
        "top_errors": report.top_errors,
        "search_trace": report.open_world_retrieval,
    }


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


def _render_error_cards(errors: list[dict[str, object]]) -> str:
    cards = []
    for error in errors[:8]:
        cards.append(
            f"""
            <article class="case {escape(str(error['decision']))}">
              <p class="eyebrow">{escape(str(error['bucket']))}</p>
              <h3>{escape(str(error['left_name']))} vs {escape(str(error['right_name']))}</h3>
              <p><strong>Decision:</strong> {escape(str(error['decision']))} at {error['score']}</p>
              <p>{escape(str(error['explanation']))}</p>
              <p><strong>Counterfactuals:</strong> {escape(' | '.join(error['counterfactuals']))}</p>
            </article>
            """
        )
    return "\n".join(cards)


def _render_slice_cards(slices: list[dict[str, object]]) -> str:
    cards = []
    for item in slices:
        cards.append(
            f"""
            <article class="identity">
              <h3>{escape(str(item['name']))}</h3>
              <p>{escape(str(item['description']))}</p>
              <p><strong>Count:</strong> {item['count']} | <strong>Precision:</strong> {item['precision']} | <strong>Recall:</strong> {item['recall']} | <strong>F1:</strong> {item['f1']}</p>
            </article>
            """
        )
    return "\n".join(cards)


def _render_trace_cards(trace: dict[str, object]) -> str:
    cards = []
    for item in trace.get("traces", [])[:4]:
        candidates = "<br>".join(
            f"{escape(candidate['display_name'])} ({escape(candidate['source_type'])}) · score={candidate['score']} · same_person={candidate['same_person']}"
            for candidate in item["top_candidates"]
        )
        cards.append(
            f"""
            <article class="panel">
              <h3>{escape(str(item['query_name']))}</h3>
              <p><strong>Top candidates</strong><br>{candidates}</p>
            </article>
            """
        )
    return "\n".join(cards)


def _render_demo_html(report) -> str:
    metrics = report.model_metrics
    headline = report.headline_metrics
    leakage_failures = [item for item in report.leakage_checks if not item["passed"]]
    
    # Load and encode logo
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Entity Resolution Lab logo" style="width: 100%; height: auto; border-radius: 12px;">'
    else:
        logo_html = ""
        
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Entity Resolution Lab</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      
      :root {{
        --bg: #FFFFFF;
        --panel: #F9FAFB;
        --ink: #222425;
        --muted: #5F6368;
        --accent: #3B82F6;
        --good: #10B981;
        --warn: #F59E0B;
        --bad: #EF4444;
        --border: #E5E7EB;
      }}
      body {{
        margin: 0;
        color: var(--ink);
        background: var(--bg);
        font-family: 'Inter', -apple-system, sans-serif;
      }}
      main {{ max-width: 1120px; margin: 0 auto; padding: 40px 20px 80px; }}
      .hero {{
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--panel);
        padding: 28px;
        margin-bottom: 22px;
      }}
      .hero-grid {{ display: grid; grid-template-columns: 120px 1fr; gap: 20px; align-items: center; }}
      .logo svg {{ width: 100%; height: auto; }}
      .eyebrow {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; color: var(--muted); }}
      .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
      .panel, .case, .identity {{
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 12px;
        padding: 18px;
      }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border); }}
      .bar {{ width: 100%; height: 10px; background: var(--border); border-radius: 999px; overflow: hidden; }}
      .bar span {{ display: block; height: 100%; background: var(--accent); }}
      .match {{ border-left: 2px solid var(--good); }}
      .abstain {{ border-left: 2px solid var(--warn); }}
      .non_match {{ border-left: 2px solid var(--bad); }}
      h1, h2, h3 {{ margin: 0 0 12px; font-weight: 700; color: var(--ink); }}
      p {{ line-height: 1.55; color: var(--ink); }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <div class="hero-grid">
          <div class="logo">
            {logo_html}
          </div>
          <div>
            <p class="eyebrow">Entity Resolution Lab</p>
            <h1>Public-Web Identity Resolution</h1>
            <p>Founder-facing benchmark and demo layer for explainable identity resolution. Headline metrics come only from the curated real-profile benchmark; stress suites are reported separately.</p>
          </div>
        </div>
      </section>

      <section class="grid" style="margin-bottom:16px;">
        <article class="panel"><h2>Headline F1</h2><p><strong>{headline['f1']}</strong></p></article>
        <article class="panel"><h2>Accepted Precision</h2><p><strong>{headline['accepted_precision']}</strong></p></article>
        <article class="panel"><h2>Mean ECE</h2><p><strong>{headline['mean_ece']}</strong></p></article>
        <article class="panel"><h2>Leakage Checks</h2><p><strong>{len(report.leakage_checks) - len(leakage_failures)}/{len(report.leakage_checks)}</strong></p></article>
      </section>

      <section class="panel">
        <h2>Benchmark</h2>
        <table>
          <thead>
            <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>AP</th><th>Visual</th></tr>
          </thead>
          <tbody>{_render_metric_rows(metrics)}</tbody>
        </table>
      </section>

      <section class="grid" style="margin-top:16px;">
        <article class="panel">
          <h2>Dataset Tracks</h2>
          <p><strong>{report.dataset_summary['headline_dataset']}</strong>: {escape(str(report.dataset_summary['headline_description']))}</p>
          <p><strong>Stress suites</strong>: hard-negative bank and synthetic stress are isolated from headline metrics.</p>
        </article>
        <article class="panel">
          <h2>Open-World Retrieval</h2>
          <p>Recall@1: <strong>{report.open_world_retrieval['recall_at_1']}</strong></p>
          <p>Recall@3: <strong>{report.open_world_retrieval['recall_at_3']}</strong></p>
          <p>Recall@5: <strong>{report.open_world_retrieval['recall_at_5']}</strong></p>
        </article>
        <article class="panel">
          <h2>Stress Metrics</h2>
          <p>Hard negatives F1: <strong>{report.stress_metrics['hard_negative_bank']['f1']}</strong></p>
          <p>Synthetic stress F1: <strong>{report.stress_metrics['synthetic_stress']['f1']}</strong></p>
        </article>
      </section>

      <section style="margin-top:18px;">
        <h2>Failure Gallery</h2>
        <div class="grid">{_render_error_cards(report.top_errors)}</div>
      </section>

      <section style="margin-top:18px;">
        <h2>Failure Slices</h2>
        <div class="grid">{_render_slice_cards(report.failure_slices)}</div>
      </section>

      <section style="margin-top:18px;">
        <h2>Search Trace</h2>
        <div class="grid">{_render_trace_cards(report.open_world_retrieval)}</div>
      </section>
    </main>
  </body>
</html>
"""


def _render_blog(report) -> str:
    failures = [item for item in report.leakage_checks if not item["passed"]]
    leakage_lines = "\n".join(f"- {'PASS' if item['passed'] else 'FAIL'}: **{item['name']}** — {item['detail']}" for item in report.leakage_checks)
    slice_lines = "\n".join(
        f"- **{item['name']}**: count={item['count']}, precision={item['precision']}, recall={item['recall']}, f1={item['f1']}"
        for item in report.failure_slices
    )
    error_lines = "\n".join(
        f"- **{item['bucket']}**: {item['left_name']} vs {item['right_name']} at {item['score']} — {item['explanation']}"
        for item in report.top_errors[:6]
    )
    return f"""# Interpretable Public-Web Entity Resolution Lab

## Benchmark

Headline dataset: `{report.dataset_name}`. The benchmark now separates:

- `real_curated_core` for headline metrics
- `hard_negative_bank` for same-name and adjacent-domain stress
- `synthetic_stress` for robustness only

Hybrid average metrics across grouped seeds:

- Precision: **{report.headline_metrics['precision']}**
- Recall: **{report.headline_metrics['recall']}**
- F1: **{report.headline_metrics['f1']}**
- Average precision: **{report.headline_metrics['average_precision']}**
- Mean Brier: **{report.headline_metrics['mean_brier']}**
- Mean ECE: **{report.headline_metrics['mean_ece']}**

## Honest Evaluation

The evaluation now uses grouped identity splits, train-only TF-IDF fitting, validation-only threshold selection, and explicit leakage diagnostics.

{leakage_lines}

Leakage failures: **{len(failures)}**

## Stress Suites

- Hard negative bank: precision={report.stress_metrics['hard_negative_bank']['precision']}, recall={report.stress_metrics['hard_negative_bank']['recall']}, f1={report.stress_metrics['hard_negative_bank']['f1']}
- Synthetic stress: precision={report.stress_metrics['synthetic_stress']['precision']}, recall={report.stress_metrics['synthetic_stress']['recall']}, f1={report.stress_metrics['synthetic_stress']['f1']}
- Open-world retrieval: recall@1={report.open_world_retrieval['recall_at_1']}, recall@3={report.open_world_retrieval['recall_at_3']}, recall@5={report.open_world_retrieval['recall_at_5']}

## Failure Modes

{slice_lines}

## Top Errors

{error_lines}

## Product Demo Features

- Evidence Ledger with supporting and contradicting signals per decision
- Counterfactual panel describing what is missing or conflicting
- Identity graph for resolved profile clusters
- Alias stress and what-if scoring in the app
- Open-world search trace for live resolution flows

## Logo Direction

Primary mark: three input nodes resolving into one canonical node, with one broken edge reconnecting to signal evidence-based identity stitching. Palette: warm paper, deep slate, electric teal, muted gold.
"""


def build_demo_artifacts(output_dir: str | Path = "reports") -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = run_benchmark("real_curated_core", protocol="grouped_cv", seeds=[7, 11, 17])
    metrics_payload = report.to_payload()
    metrics_path = output_path / "benchmark_metrics.json"
    demo_path = output_path / "demo.html"
    blog_path = output_path / "blog.md"

    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    demo_path.write_text(_render_demo_html(report))
    blog_path.write_text(_render_blog(report))

    return {
        "metrics_json": str(metrics_path),
        "demo_html": str(demo_path),
        "blog_md": str(blog_path),
    }
