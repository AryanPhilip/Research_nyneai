# Interpretable Public-Web Entity Resolution Lab

## Problem
AI agents need more than a name string. They need a way to connect fragmented public professional profiles into a unified identity while avoiding false merges on same-name collisions.

## Benchmark
The benchmark combines curated public-profile-style records for AI/ML builders with hard negatives, including exact-name collisions. Splits are person-disjoint.

| Model | Precision | Recall | F1 | AP |
| --- | --- | --- | --- | --- |
| fuzzy_name | 0.4 | 1.0 | 0.571 | 0.4 |
| embedding_only | 0.333 | 1.0 | 0.5 | 0.333 |
| lexical_baseline | 0.545 | 1.0 | 0.706 | 0.915 |
| hybrid | 1.0 | 1.0 | 1.0 | 1.0 |

Cluster B-cubed F1: **0.906**

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
| full | 1.0 | 1.0 |
| no_embedding | 1.0 | 1.0 |
| no_structured | 0.706 | 0.773 |

The structured feature block is doing meaningful work. Removing it collapses performance on held-out same-name collisions.

## Curated Cases
- `match`: Andrej Karpathy vs Andrej Karpathy (1.0) with reason codes `high_name_similarity, shared_domain, shared_org, shared_topics`
- `match`: Chip Huyen vs Chip Huyen (0.722) with reason codes `high_name_similarity, shared_domain, shared_topics`
- `match`: Jay Alammar vs Jay Alammar (1.0) with reason codes `high_name_similarity, shared_domain, shared_org, shared_topics`
- `non_match`: Andrej Karpathy vs Chip Huyen (0.0) with reason codes `shared_topics, location_conflict`
- `non_match`: Andrej Karpathy vs Chip Huyen (0.0) with reason codes `shared_topics`

## Failure Modes
- Cross-domain topic overlap still creates noisy candidates in blocking.
- The current corpus is intentionally small and curated; scaling would need richer raw-page coverage and more adversarial negatives.
- Embedding quality is limited by a local TF-IDF representation; a production system would swap in a stronger embedder while preserving the same evaluation harness.

## What I Would Build Next
- Expand the public-page corpus beyond the current seed set.
- Add richer temporal contradiction handling and alias discovery.
- Replace TF-IDF retrieval with a stronger semantic embedder while keeping the same benchmark and abstention policy.
