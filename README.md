# Entity Resolution Lab

A benchmark-first entity resolution system for matching fragmented public professional identities across the open web. Given scattered online profiles (GitHub, personal sites, conference bios, company pages, podcast appearances, Hugging Face), the system decides when they belong to the same person, when they do not, and when it should abstain.

## What It Does

- Ingests and normalizes public profile records across six source types
- Generates candidate pairs via rule-based blocking (fuzzy name, alias/initial, domain overlap, org/title overlap) plus TF-IDF embedding fallback
- Scores pairs with a calibrated hybrid matcher over 15 lexical, structured, contradiction, and semantic features
- Produces evidence cards, conservative abstention decisions, and clustered canonical identities
- Separates headline evaluation from harder stress suites to prevent inflated metrics
- Ships an interactive Streamlit dashboard, a static HTML report, JSON benchmark artifacts, and a blog-style writeup

## Architecture

The system follows a six-stage pipeline:

```
RawProfilePage -> parse_raw_page() -> ProfileRecord
  -> compose_normalized_text() -> normalized text
  -> generate_block_candidates() -> BlockCandidate pairs
  -> PairFeatureExtractor.featurize_pair() -> PairExample (15 features)
  -> TrainedHybridMatcher.score_examples() -> scores + decisions
  -> resolve_identities() -> CanonicalIdentity clusters + EvidenceCards
```

Key design decisions:

- **Person-disjoint splits** -- Train/val/test split on canonical person ID, not individual profiles, to prevent information leakage.
- **Explainable blocking** -- All candidate pairs carry traceable rule-based reasons that propagate through to evidence card outputs.
- **Calibrated abstention** -- Isotonic/sigmoid calibration with three-way decisions (match / non-match / abstain) and a contradiction veto for high-confidence name matches lacking corroborating structured evidence.
- **Feature importance** -- Ablation studies show removing structured features (org, location, temporal) degrades F1 from 1.0 to 0.667, confirming their role in same-name collision handling.

## Dataset Tiers

The benchmark is intentionally split into three tiers to prevent easy overfitting:

| Tier | Description | Size |
|------|-------------|------|
| `real_curated_core` | Curated public AI/ML-builder profiles | 56 profiles / 16 identities |
| `hard_negative_bank` | Same-name and adjacent-domain distractors | 65 profiles |
| `synthetic_stress` | Generated perturbations for robustness checks | 88 profiles |

## Current Headline Metrics

From `real_curated_core` with grouped multi-seed evaluation:

| Metric | Value |
|--------|-------|
| Precision | 1.000 |
| Recall | 0.839 |
| F1 | 0.901 |
| Average Precision | 1.000 |
| Mean Brier Score | 0.018 |
| Mean ECE | 0.039 |
| Accepted Precision (post-abstention) | 1.000 |
| Accepted Recall (post-abstention) | 0.800 |
| Abstain Rate | 9.1% |

Stress-suite snapshots: hard-negative F1 0.937, synthetic-stress F1 0.888. Leakage checks pass cleanly.

## Interactive Dashboard

The Streamlit dashboard provides ten tabs for deep exploration:

- **Benchmark** -- Headline metrics, model comparison, score distributions, calibration curves, and a 3-way confusion matrix heatmap
- **Identity Graph** -- Force-directed network visualization of resolved identities with interactive filtering
- **Failure Lab** -- Sorted error gallery with Splink-style waterfall charts showing per-feature score contributions
- **Evidence Ledger** -- Full evidence card viewer with waterfall breakdowns and optional LLM-powered natural language explanations
- **Pipeline Inspector** -- Global feature weight distribution, PCA score landscape, and blocking effectiveness analysis
- **Alias Stress** -- What-if comparison mode with optional LLM adjudication for ambiguous pairs
- **Data Profile** -- Field completeness heatmaps by source type, source distribution summaries, and per-identity complementarity views
- **Model Arena** -- Head-to-head comparison of all models (name-only, embedding-only, lexical LR, hybrid) with agreement heatmaps and disagreement exploration
- **Resolve** -- Paste raw profile text or URLs for live resolution with search trace visualization
- **Search Trace** -- Open-world candidate retrieval debugging with blocking reason inspection

### AI-Powered Features

When an OpenAI or Anthropic API key is available (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`), additional capabilities are enabled:

- **LLM Adjudication** -- Ask an LLM to independently assess ambiguous match decisions
- **Natural Language Explanations** -- Generate human-readable explanations of why two profiles were matched or rejected

These features degrade gracefully when no API key is configured.

## Getting Started

Install dependencies:

```bash
pip install -e ".[dev]"
```

For LLM features (optional):

```bash
pip install -e ".[dev,llm]"
```

Run tests:

```bash
pytest -q
```

Generate reports:

```bash
python -m nyne_er_lab --output-dir reports
```

Launch the interactive dashboard:

```bash
streamlit run src/nyne_er_lab/app.py
```

## Output Artifacts

| Artifact | Description |
|----------|-------------|
| `reports/benchmark_metrics.json` | Machine-readable benchmark report with headline metrics, CV summary, stress metrics, leakage checks, thresholds, and top errors |
| `reports/demo.html` | Static HTML demo with interactive Plotly visualizations |
| `reports/blog.md` | Technical writeup of the benchmark methodology and failure modes |

## Project Structure

```
src/nyne_er_lab/
  schemas.py           Pydantic data contracts used at all module boundaries
  datasets.py          Dataset registry and tiered benchmark loading
  ingest/              HTML parsing (6 source types) and text normalization
  blocking/            Rule-based candidate generation with TF-IDF fallback
  features/            15-feature pairwise extraction with train-only TF-IDF
  models/              Baselines + calibrated hybrid matcher with abstention
  models/llm_adjudicator.py   LLM integration for adjudication and explanation
  eval/                Metrics, person-disjoint splits, benchmark runner
  cluster/             Union-find clustering and evidence card assembly
  demo/                HTML report builder and static assets
  app.py               Streamlit interactive dashboard
  app_data.py          Dashboard data layer
data/
  fixtures/            Seed profile JSON fixtures with canonical person IDs
  raw_pages/           Source HTML snapshots
tests/                 Pipeline stage tests with quantitative assertions
reports/               Generated benchmark artifacts
```

## Limitations

The headline dataset is curated and relatively small. Same-name collisions are mostly pushed into the hard-negative stress suite, which means the headline benchmark is cleaner than a real open-world deployment. The next meaningful improvement is a larger real curated corpus with harder same-name collisions inside the headline evaluation set, not more model complexity.
