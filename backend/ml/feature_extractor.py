"""
Feature Extractor for Mental Stress NLP Pipeline

Implements:
- TF-IDF vectorization (sklearn)
- LDA topic modeling (sklearn)
- Combined feature matrix for ensemble models
- Serialization / deserialization utilities
"""

import os
import pickle
import logging
from typing import Optional

import numpy as np
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Default paths (relative to backend/)
DEFAULT_TFIDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "saved_models", "tfidf_vectorizer.pkl"
)
DEFAULT_LDA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "saved_models", "lda_model.pkl"
)


class FeatureExtractor:
    """
    Unified feature extraction combining TF-IDF and LDA topic features.

    Attributes:
        tfidf: Fitted TfidfVectorizer
        lda: Fitted LatentDirichletAllocation model
        n_topics: Number of LDA topics
        max_features: TF-IDF vocabulary size cap
    """

    def __init__(self, max_features: int = 10_000, n_topics: int = 20):
        self.max_features = max_features
        self.n_topics = n_topics
        self._is_fitted = False

        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),       # Unigrams + bigrams
            sublinear_tf=True,         # Apply log normalization
            min_df=2,                  # Ignore very rare terms
            max_df=0.95,               # Ignore very common terms
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # Words only
        )

        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=20,
            learning_method="online",
            learning_offset=50.0,
            random_state=42,
            n_jobs=1,  # Set to 1 to avoid PicklingError in background runs
        )

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> "FeatureExtractor":
        """
        Fit TF-IDF and LDA on a corpus of preprocessed texts.

        Args:
            texts: List of preprocessed text strings

        Returns:
            self (for chaining)
        """
        logger.info("Fitting TF-IDF vectorizer (max_features=%d)...", self.max_features)
        tfidf_matrix = self.tfidf.fit_transform(texts)

        logger.info("Fitting LDA model (n_topics=%d)...", self.n_topics)
        self.lda.fit(tfidf_matrix)

        self._is_fitted = True
        logger.info("FeatureExtractor fitted successfully.")
        return self

    def transform_tfidf(self, texts: list[str]) -> csr_matrix:
        """Return TF-IDF sparse matrix for given texts."""
        self._check_fitted()
        return self.tfidf.transform(texts)

    def transform_lda(self, texts: list[str]) -> np.ndarray:
        """
        Return LDA topic distribution matrix (dense).

        Shape: (n_samples, n_topics)
        """
        self._check_fitted()
        tfidf_matrix = self.tfidf.transform(texts)
        return self.lda.transform(tfidf_matrix)

    def transform_combined(self, texts: list[str]) -> csr_matrix:
        """
        Return combined TF-IDF + LDA feature matrix (sparse).

        Used by Random Forest and LSTM (after densification).
        Shape: (n_samples, max_features + n_topics)
        """
        self._check_fitted()
        tfidf_mat = self.transform_tfidf(texts)
        lda_mat = csr_matrix(self.transform_lda(texts))
        return hstack([tfidf_mat, lda_mat])

    def transform(self, texts: list[str], mode: str = "tfidf") -> np.ndarray:
        """
        Unified transform interface.

        Args:
            texts: Preprocessed text strings
            mode: One of 'tfidf', 'lda', 'combined'

        Returns:
            Feature matrix (sparse or dense depending on mode)
        """
        modes = {
            "tfidf": self.transform_tfidf,
            "lda": self.transform_lda,
            "combined": self.transform_combined,
        }
        if mode not in modes:
            raise ValueError(f"mode must be one of {list(modes.keys())}")
        return modes[mode](texts)

    # ------------------------------------------------------------------
    # Topic inspection
    # ------------------------------------------------------------------

    def get_top_words_per_topic(self, n_words: int = 10) -> dict[int, list[str]]:
        """Return top n_words for each LDA topic."""
        self._check_fitted()
        feature_names = self.tfidf.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            topics[topic_idx] = [feature_names[i] for i in top_indices]
        return topics

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(
        self,
        tfidf_path: str = DEFAULT_TFIDF_PATH,
        lda_path: str = DEFAULT_LDA_PATH,
    ) -> None:
        """Persist TF-IDF and LDA models to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)
        with open(tfidf_path, "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(lda_path, "wb") as f:
            pickle.dump(self.lda, f)
        logger.info("FeatureExtractor saved: %s, %s", tfidf_path, lda_path)

    @classmethod
    def load(
        cls,
        tfidf_path: str = DEFAULT_TFIDF_PATH,
        lda_path: str = DEFAULT_LDA_PATH,
    ) -> "FeatureExtractor":
        """Load a previously saved FeatureExtractor from disk."""
        extractor = cls.__new__(cls)
        with open(tfidf_path, "rb") as f:
            extractor.tfidf = pickle.load(f)
        with open(lda_path, "rb") as f:
            extractor.lda = pickle.load(f)
        extractor.max_features = extractor.tfidf.max_features
        extractor.n_topics = extractor.lda.n_components
        extractor._is_fitted = True
        logger.info("FeatureExtractor loaded from disk.")
        return extractor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform.")

    @property
    def vocab_size(self) -> int:
        self._check_fitted()
        return len(self.tfidf.vocabulary_)

    @property
    def feature_dim(self) -> int:
        """Total feature dimension: TF-IDF + LDA."""
        return self.vocab_size + self.n_topics


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessor import preprocess_batch

    sample_texts = [
        "I feel so hopeless and tired. Nothing ever gets better.",
        "My boss yelled at me today. I am furious and stressed.",
        "The financial debt is crushing me. I cannot sleep at night.",
        "I had a good day. Things are manageable and I feel okay.",
        "My anxiety about the exam is overwhelming. I cannot focus.",
    ] * 10  # Expand for LDA to have enough data

    processed = preprocess_batch(sample_texts)
    extractor = FeatureExtractor(max_features=500, n_topics=5)
    extractor.fit(processed)

    tfidf = extractor.transform_tfidf(processed[:3])
    lda = extractor.transform_lda(processed[:3])
    combined = extractor.transform_combined(processed[:3])

    print(f"TF-IDF shape: {tfidf.shape}")
    print(f"LDA shape:    {lda.shape}")
    print(f"Combined:     {combined.shape}")
    print(f"Vocab size:   {extractor.vocab_size}")
    print(f"Feature dim:  {extractor.feature_dim}")

    print("\nTop words per LDA topic:")
    for idx, words in extractor.get_top_words_per_topic(5).items():
        print(f"  Topic {idx}: {words}")
