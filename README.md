# Nyne ER Lab

Public-web identity resolution lab for matching fragmented professional profiles across the open internet.

This project is a founder-facing proof of work for the kind of entity resolution problem Nyne cares about: given scattered public profiles, can a system decide when they belong to the same person, when they do not, and when it should abstain?

## What It Does

- Ingests normalized public-profile records across sources like GitHub, personal sites, conference bios, company pages, podcast bios, and Hugging Face
- Generates candidate pairs with rule-based blocking plus TF-IDF neighbor fallback
- Scores pairs with a calibrated hybrid matcher over lexical, structured, contradiction, and semantic features
- Produces evidence cards, conservative abstention, and clustered canonical identities
- Separates headline evaluation from harder stress suites so the benchmark is more honest
- Ships a Streamlit demo, generated HTML report, JSON benchmark artifact, and blog-style writeup

## Why This Version Is Better

The earlier version was too easy to overtrust because synthetic examples and easy split behavior could inflate the apparent model quality.

This version fixes that by:

- using explicit dataset tiers instead of one mixed benchmark
- fitting TF-IDF features on train profiles only during scored evaluation
- reporting grouped multi-seed benchmark results
- running leakage checks
- keeping hard negatives and synthetic stress cases separate from headline claims
- exposing failure slices, top errors, and open-world retrieval traces

## Dataset Tiers

The benchmark is intentionally split into three named datasets:

- `real_curated_core`
  Headline benchmark of curated public AI/ML-builder profiles only.
- `hard_negative_bank`
  Same-name and adjacent-domain distractors used to stress false-merge behavior.
- `synthetic_stress`
  Generated perturbations used for robustness checks only, not headline claims.

Current corpus sizes:

- `real_curated_core`: 56 profiles / 16 identities
- `hard_negative_bank`: 65 profiles
- `synthetic_stress`: 88 profiles

## Current Headline Metrics

From `real_curated_core` with grouped multi-seed evaluation:

- Precision: `1.000`
- Recall: `0.839`
- F1: `0.901`
- Average precision: `1.000`
- Mean Brier: `0.0179`
- Mean ECE: `0.0386`
- Accepted precision after abstention: `1.000`
- Accepted recall after abstention: `0.800`
- Abstain rate: `0.091`

Stress-suite snapshots:

- `hard_negative_bank` F1: `0.937`
- `synthetic_stress` F1: `0.888`

Leakage checks currently pass cleanly.

## Important Caveat

The project is much more credible now, but it is still not a production-scale benchmark.

The headline dataset is curated and relatively small. Same-name collisions are mostly pushed into the hard-negative stress suite, which means the headline benchmark is cleaner than a real open-world deployment. That is acceptable for a proof-of-work demo, but not enough to claim production readiness.

## Running It

Install dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Generate reports:

```bash
PYTHONPATH=src python3 -m nyne_er_lab --output-dir reports
```

Run the Streamlit app:

```bash
streamlit run src/nyne_er_lab/app.py
```

## Main Outputs

- [reports/benchmark_metrics.json](/Users/aryanphilip/Developer/Research_nyneai/reports/benchmark_metrics.json)
  Machine-readable benchmark report with headline metrics, CV summary, stress metrics, leakage checks, thresholds, and top errors.
- [reports/demo.html](/Users/aryanphilip/Developer/Research_nyneai/reports/demo.html)
  Static founder-facing demo artifact.
- [reports/blog.md](/Users/aryanphilip/Developer/Research_nyneai/reports/blog.md)
  Short technical writeup of the benchmark and failure modes.

## Project Structure

- [src/nyne_er_lab/eval/benchmark.py](/Users/aryanphilip/Developer/Research_nyneai/src/nyne_er_lab/eval/benchmark.py)
  Honest benchmark runner and report assembly.
- [src/nyne_er_lab/datasets.py](/Users/aryanphilip/Developer/Research_nyneai/src/nyne_er_lab/datasets.py)
  Dataset registry and tiered benchmark loading.
- [src/nyne_er_lab/features/extractor.py](/Users/aryanphilip/Developer/Research_nyneai/src/nyne_er_lab/features/extractor.py)
  Pairwise feature extraction with train-only TF-IDF fitting.
- [src/nyne_er_lab/models/hybrid.py](/Users/aryanphilip/Developer/Research_nyneai/src/nyne_er_lab/models/hybrid.py)
  Calibrated hybrid matcher with abstention.
- [src/nyne_er_lab/cluster/resolver.py](/Users/aryanphilip/Developer/Research_nyneai/src/nyne_er_lab/cluster/resolver.py)
  Evidence cards and canonical identity assembly.
- [src/nyne_er_lab/app.py](/Users/aryanphilip/Developer/Research_nyneai/src/nyne_er_lab/app.py)
  Interactive demo UI.

## Demo Features

- Benchmark dashboard with leakage checks and stress-suite metrics
- Identity graph visualization
- Failure gallery and evidence ledger
- Alias stress / what-if comparison mode
- Live resolver for pasted profile text or URLs
- Search trace view for open-world candidate retrieval

## Bottom Line

Yes, the project looks good now.

It reads like a real systems + ML proof of work instead of a toy benchmark. The strongest parts are the dataset separation, abstention, evidence cards, and the fact that the suspicious perfect score is gone.

If you want to make it stronger after this, the next real step is not more model complexity. It is a larger real curated corpus with harder same-name collisions inside the headline evaluation set.
