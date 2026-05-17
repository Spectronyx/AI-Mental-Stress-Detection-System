"""
Random Forest Multi-Label Classifier for Mental Stress Detection

Architecture:
- OneVsRestClassifier wrapping RandomForestClassifier
- Input: Combined TF-IDF + LDA features (sparse → dense)
- Class imbalance handled via balanced class weights
- Output: Binary label matrix + confidence probabilities
"""

import os
import pickle
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "rf.pkl"
)


class RandomForestModel:
    """
    Random Forest multi-label classifier.

    Inputs combined TF-IDF + LDA features and outputs multi-label
    predictions with confidence probabilities.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._models: dict[str, OneVsRestClassifier] = {}
        self._label_axes: list[str] = []
        self._is_fitted = False

    def _to_dense(self, X):
        """Convert sparse matrix to dense if needed."""
        if issparse(X):
            return X.toarray()
        return X

    def fit(self, X, y_dict: dict[str, np.ndarray]) -> "RandomForestModel":
        """
        Fit one OvR-RandomForest per label axis.

        Args:
            X: Combined TF-IDF + LDA matrix (sparse or dense)
            y_dict: Dict mapping axis name → binary label matrix
        """
        X_dense = self._to_dense(X)
        self._label_axes = list(y_dict.keys())

        for axis, y in y_dict.items():
            logger.info("Fitting RandomForest for axis: %s", axis)
            base = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=1,  # Set to 1 for background stability
            )
            clf = OneVsRestClassifier(base, n_jobs=1)
            clf.fit(X_dense, y)
            self._models[axis] = clf

        self._is_fitted = True
        logger.info("RandomForestModel fitted for axes: %s", self._label_axes)
        return self

    def predict(self, X) -> dict[str, np.ndarray]:
        self._check_fitted()
        X_dense = self._to_dense(X)
        return {
            axis: clf.predict(X_dense)
            for axis, clf in self._models.items()
        }

    def predict_proba(self, X) -> dict[str, np.ndarray]:
        self._check_fitted()
        X_dense = self._to_dense(X)
        return {
            axis: clf.predict_proba(X_dense)
            for axis, clf in self._models.items()
        }

    def predict_single(self, X) -> dict[str, dict]:
        result = {}
        pred = self.predict(X)
        proba = self.predict_proba(X)
        for axis in self._label_axes:
            result[axis] = {
                "predictions": pred[axis][0].tolist(),
                "probabilities": proba[axis][0].tolist(),
            }
        return result

    def get_feature_importance(self, feature_names: list[str] = None) -> dict:
        """
        Return aggregated feature importances from the thematic classifier.
        """
        self._check_fitted()
        if "thematic" not in self._models:
            return {}
        # Average importance across all binary estimators
        importances = np.mean(
            [est.feature_importances_ for est in self._models["thematic"].estimators_],
            axis=0,
        )
        if feature_names:
            return dict(zip(feature_names, importances.tolist()))
        return {"importances": importances.tolist()}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("RandomForestModel saved to %s", path)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "RandomForestModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("RandomForestModel loaded from %s", path)
        return model

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("RandomForestModel must be fitted before predict.")
