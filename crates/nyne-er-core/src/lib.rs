pub mod benchmark;
pub mod blocking;
pub mod cluster;
pub mod datasets;
pub mod features;
pub mod ingest;
pub mod live;
pub mod llm;
pub mod metrics;
pub mod models;
pub mod schemas;
pub mod search;
pub mod similarity;
pub mod tfidf;
pub mod util;

pub use benchmark::{run_benchmark, BenchmarkReport};
pub use blocking::{
    blocking_recall, candidate_volume_ratio, domain_overlap_match, embedding_neighbor_candidates,
    exact_or_fuzzy_name_match, generate_block_candidates, gold_positive_pairs,
    org_or_title_overlap_match, BlockCandidate,
};
pub use cluster::{ResolvedPair, bcubed_f1, generate_evidence_card, resolve_identities};
pub use datasets::{
    available_datasets, load_benchmark_profiles, load_dataset, load_raw_pages, load_seed_profiles,
    DatasetBundle,
};
pub use features::{
    assert_person_disjoint, assign_profile_splits, build_examples_for_profiles, build_pair_examples,
    build_split_candidates, profiles_for_split, summarize_split_assignments, PairExample,
    PairFeatureExtractor, SplitName, FEATURE_ORDER,
};
pub use ingest::{compose_normalized_text, parse_raw_page, parse_raw_pages};
pub use live::{profile_from_text, profile_from_url, resolve_live_profile, LiveResult, TextProfileInput};
pub use llm::{adjudicate_pair, explain_pair, llm_available, LLMResult};
pub use metrics::{
    average_precision_score, brier_score, expected_calibration_error, optimize_threshold,
    summarize_predictions, threshold_sweep, MetricSummary,
};
pub use models::{
    run_embedding_baseline, run_feature_ablations, run_hybrid_matcher, run_lexical_baseline,
    run_name_baseline, train_hybrid_matcher, AblationResult, BaselineRun, HybridRun,
    LlmAdjudicator, TrainedHybridMatcher,
};
pub use schemas::{
    CandidatePair, CanonicalIdentity, ConfidenceBand, DecisionType, EducationClaim, EvidenceCard,
    EvidenceSignal, OrganizationClaim, ProfileRecord, RawProfilePage, SourceType, TextSpan,
};
pub use search::{search_and_build_profiles, search_person, SearchHit};
