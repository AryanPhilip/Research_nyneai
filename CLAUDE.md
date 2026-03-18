# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Entity Resolution Lab -- a benchmark-first entity resolution system for matching fragmented public professional identities (e.g., GitHub profiles, conference bios, personal sites) across the open web. It resolves multiple online profiles of the same person into unified canonical identities while avoiding false merges on same-name collisions.

## Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file / specific test
pytest tests/test_blocking.py
pytest tests/test_hybrid.py::test_hybrid_beats_baselines

# Run the full end-to-end demo pipeline (generates reports/)
python -m nyne_er_lab

# Launch Streamlit demo (if applicable)
streamlit run src/nyne_er_lab/demo/builder.py
```

## Architecture

The system is a six-stage pipeline, all orchestrated by `demo/builder.py:build_demo_artifacts()`:

```
RawProfilePage → parse_raw_page() → ProfileRecord
    → compose_normalized_text() → normalized text
    → generate_block_candidates() → BlockCandidate pairs
    → PairFeatureExtractor.featurize_pair() → PairExample (15 features)
    → TrainedHybridMatcher.score_examples() → scores + decisions
    → resolve_identities() → CanonicalIdentity clusters + EvidenceCards
```

### Key modules under `src/nyne_er_lab/`

- **`schemas.py`** — Pydantic data contracts used throughout: `RawProfilePage`, `ProfileRecord`, `CandidatePair`, `EvidenceCard`, `CanonicalIdentity`. All inter-module boundaries use these typed models.
- **`datasets.py`** — Loads the 23 seed profile fixtures from `data/` with canonical person ID labels.
- **`ingest/`** — `parsers.py` extracts structured fields from HTML using source-specific CSS selectors (6 source types: GitHub, personal site, conference bio, company profile, podcast guest, HuggingFace). `normalize.py` builds lowercased deduplicated text for downstream retrieval.
- **`blocking/blocker.py`** — Candidate generation via rule-based blocking (fuzzy name, alias/initial, domain overlap, org/title overlap) plus TF-IDF embedding fallback. Every pair carries explainable blocking reasons.
- **`features/`** — `extractor.py:PairFeatureExtractor` computes 15 normalized features (lexical, structured, contradiction, semantic). `dataset.py` assembles person-disjoint train/val/test splits.
- **`models/`** — `baselines.py` has three baselines (name-only, embedding-only, lexical logistic regression). `hybrid.py:TrainedHybridMatcher` is a calibrated logistic regression on all 15 features with isotonic calibration and a conservative abstention band + contradiction veto.
- **`eval/`** — `metrics.py` for pairwise precision/recall/F1/AP with threshold optimization. `clusters.py` for B-cubed F1. `splits.py` for person-disjoint split assignment.
- **`cluster/resolver.py`** — Union-find clustering of match decisions into `CanonicalIdentity` objects with `EvidenceCard` explanations per pair.
- **`demo/builder.py`** — End-to-end orchestrator that trains all models, clusters identities, and outputs `benchmark_metrics.json`, `demo.html`, and `blog.md` into `reports/`.

### Important design decisions

- **Person-disjoint splits**: Train/val/test split on `canonical_person_id`, not individual profiles, to prevent information leakage.
- **Explainable blocking**: All candidate pairs include traceable rule-based reasons; these propagate through to `EvidenceCard` outputs.
- **Calibrated abstention**: The hybrid matcher uses isotonic/sigmoid calibration and a three-way decision (match/non_match/abstain) with a contradiction veto for high-confidence name matches that lack corroborating structured evidence.
- **Structured features are critical**: Feature ablations show removing structured features (org, location, temporal) degrades F1 from 1.0 to 0.667 — essential for same-name collision handling.

## Data

- `data/` contains seed profile fixtures (JSON) and raw HTML page snapshots used as ground truth.
- Benchmark profiles are public AI/ML builders with intentional hard negatives (e.g., "Chip Huynh" vs "Chip Huyen").

## Testing

Tests are in `tests/` and cover each pipeline stage with assertions on blocking recall (>=95%), volume ratio (<75%), model ordering (hybrid beats baselines), calibration quality (Brier score), cluster separation (B-cubed F1 >= 0.8), and artifact generation.
