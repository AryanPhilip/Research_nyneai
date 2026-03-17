"""Feature extraction for pairwise entity-matching models."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nyne_er_lab.ingest import compose_normalized_text
from nyne_er_lab.schemas import ProfileRecord


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _token_set(values: list[str]) -> set[str]:
    return {
        token
        for value in values
        for token in _normalize_text(value).split()
        if token
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _sequence_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, _normalize_text(left), _normalize_text(right)).ratio()


def _domains(profile: ProfileRecord) -> set[str]:
    urls = [str(link) for link in profile.outbound_links]
    if profile.source_type in {"personal_site", "huggingface"}:
        urls.append(str(profile.url))
    return {
        urlparse(url).netloc.replace("www.", "")
        for url in urls
        if urlparse(url).netloc
    }


def _org_names(profile: ProfileRecord) -> list[str]:
    return [org.name for org in profile.organizations]


def _topic_values(profile: ProfileRecord) -> list[str]:
    values = list(profile.topics)
    if profile.headline:
        values.append(profile.headline)
    return values


def _year_values(profile: ProfileRecord) -> set[int]:
    values: set[int] = set()
    for timestamp in profile.timestamps:
        try:
            values.add(int(timestamp[:4]))
        except ValueError:
            continue
    for org in profile.organizations:
        if org.start_year:
            values.add(org.start_year)
        if org.end_year:
            values.add(org.end_year)
    return values


@dataclass
class PairFeatureExtractor:
    """Shared featurizer reused by baselines and hybrid models."""

    vectorizer: TfidfVectorizer | None = None
    profile_vectors: dict[str, np.ndarray] | None = None
    fit_profile_ids: set[str] | None = None
    feature_order: tuple[str, ...] = (
        "name_similarity",
        "alias_similarity",
        "headline_similarity",
        "bio_similarity",
        "shared_domain_count",
        "org_overlap_count",
        "org_jaccard",
        "topic_overlap_count",
        "topic_jaccard",
        "location_overlap",
        "temporal_overlap_count",
        "temporal_distance",
        "location_conflict",
        "same_source_type",
        "embedding_cosine",
    )

    def fit(self, profiles: list[ProfileRecord]) -> "PairFeatureExtractor":
        texts = [compose_normalized_text(profile) for profile in profiles]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = self.vectorizer.fit_transform(texts)
        dense = matrix.toarray()
        self.profile_vectors = {
            profile.profile_id: dense[index]
            for index, profile in enumerate(profiles)
        }
        self.fit_profile_ids = {profile.profile_id for profile in profiles}
        return self

    def _profile_vector(self, profile: ProfileRecord) -> np.ndarray | None:
        if self.profile_vectors is None:
            return None
        cached = self.profile_vectors.get(profile.profile_id)
        if cached is not None:
            return cached
        if self.vectorizer is None:
            return None
        vector = self.vectorizer.transform([compose_normalized_text(profile)]).toarray()[0]
        self.profile_vectors[profile.profile_id] = vector
        return vector

    def _embedding_cosine(self, left: ProfileRecord, right: ProfileRecord) -> float:
        if not self.profile_vectors:
            return 0.0

        left_vec = self._profile_vector(left)
        right_vec = self._profile_vector(right)
        if left_vec is None or right_vec is None:
            return 0.0
        left_norm = np.linalg.norm(left_vec)
        right_norm = np.linalg.norm(right_vec)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))

    def featurize_pair(self, left: ProfileRecord, right: ProfileRecord) -> dict[str, float]:
        left_aliases = [left.display_name, *left.aliases]
        right_aliases = [right.display_name, *right.aliases]
        left_orgs = _token_set(_org_names(left))
        right_orgs = _token_set(_org_names(right))
        left_topics = _token_set(_topic_values(left))
        right_topics = _token_set(_topic_values(right))
        left_locations = _token_set(left.locations)
        right_locations = _token_set(right.locations)
        left_years = _year_values(left)
        right_years = _year_values(right)

        temporal_distance = 0.0
        if left_years and right_years:
            temporal_distance = float(abs(min(left_years) - min(right_years)))

        features = {
            "name_similarity": _sequence_similarity(left.display_name, right.display_name),
            "alias_similarity": max(
                _sequence_similarity(left_alias, right_alias)
                for left_alias in left_aliases
                for right_alias in right_aliases
            ),
            "headline_similarity": _sequence_similarity(left.headline or "", right.headline or ""),
            "bio_similarity": _sequence_similarity(left.bio_text, right.bio_text),
            "shared_domain_count": float(len(_domains(left) & _domains(right))),
            "org_overlap_count": float(len(left_orgs & right_orgs)),
            "org_jaccard": _jaccard(left_orgs, right_orgs),
            "topic_overlap_count": float(len(left_topics & right_topics)),
            "topic_jaccard": _jaccard(left_topics, right_topics),
            "location_overlap": float(bool(left_locations & right_locations)),
            "temporal_overlap_count": float(len(left_years & right_years)),
            "temporal_distance": temporal_distance,
            "location_conflict": float(bool(left_locations and right_locations and not (left_locations & right_locations))),
            "same_source_type": float(left.source_type == right.source_type),
            "embedding_cosine": self._embedding_cosine(left, right),
        }
        return features

    def vectorize_features(
        self,
        feature_rows: list[dict[str, float]],
        *,
        include_embedding: bool = True,
        feature_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        order = list(feature_names or self.feature_order)
        if not include_embedding:
            order = [feature_name for feature_name in order if feature_name != "embedding_cosine"]
        return np.asarray([[row.get(feature_name, 0.0) for feature_name in order] for row in feature_rows], dtype=float)
