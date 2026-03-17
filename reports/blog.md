# Interpretable Public-Web Entity Resolution Lab

## Benchmark

Headline dataset: `real_curated_core`. The benchmark now separates:

- `real_curated_core` for headline metrics
- `hard_negative_bank` for same-name and adjacent-domain stress
- `synthetic_stress` for robustness only

Hybrid average metrics across grouped seeds:

- Precision: **1.0**
- Recall: **0.839**
- F1: **0.901**
- Average precision: **1.0**
- Mean Brier: **0.0179**
- Mean ECE: **0.0386**

## Honest Evaluation

The evaluation now uses grouped identity splits, train-only TF-IDF fitting, validation-only threshold selection, and explicit leakage diagnostics.

- PASS: **canonical_person_disjoint** — Canonical identities are disjoint across splits.
- PASS: **duplicate_urls_across_train_test** — No duplicated URLs across train/test.
- PASS: **near_duplicate_raw_text_across_train_test** — No near-duplicate raw text across train/test.
- PASS: **synthetic_contamination** — Headline dataset contains no synthetic profiles.
- PASS: **train_only_feature_fitting** — TF-IDF fit on train profiles only.
- PASS: **threshold_selected_on_validation_only** — Baseline and hybrid thresholds are chosen on validation splits only.

Leakage failures: **0**

## Stress Suites

- Hard negative bank: precision=0.944, recall=0.931, f1=0.937
- Synthetic stress: precision=0.85, recall=0.93, f1=0.888
- Open-world retrieval: recall@1=1.0, recall@3=1.0, recall@5=1.0

## Failure Modes

- **location_conflict**: count=14, precision=1.0, recall=0.9, f1=0.947
- **sparse_overlap**: count=1, precision=0.0, recall=0.0, f1=0.0
- **cross_source_true_match**: count=15, precision=1.0, recall=0.8, f1=0.889

## Top Errors

- **false_negative**: Harrison Chase vs Harrison Chase at 0.333 — Decision=non_match at confidence 0.33. Supporting signals: high_name_similarity, shared_org, shared_topics.
- **abstain**: Harrison Chase vs Harrison Chase at 0.5 — Decision=abstain at confidence 0.50. Supporting signals: high_name_similarity, shared_org, shared_topics. Contradictions: location_conflict.
- **abstain**: Harrison Chase vs Harrison Chase at 0.5 — Decision=abstain at confidence 0.50. Supporting signals: high_name_similarity, shared_org, shared_topics.

## Product Demo Features

- Evidence Ledger with supporting and contradicting signals per decision
- Counterfactual panel describing what is missing or conflicting
- Identity graph for resolved profile clusters
- Alias stress and what-if scoring in the app
- Open-world search trace for live resolution flows

## Logo Direction

Primary mark: three input nodes resolving into one canonical node, with one broken edge reconnecting to signal evidence-based identity stitching. Palette: warm paper, deep slate, electric teal, muted gold.
