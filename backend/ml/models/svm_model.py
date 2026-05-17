"""
SVM Multi-Label Classifier for Mental Stress Detection

Architecture:
- OneVsRestClassifier wrapping LinearSVC (fast, memory-efficient)
- CalibratedClassifierCV wrapper for probability outputs
- Input: TF-IDF sparse matrix
- Output: Binary label matrix + confidence probabilities
"""

import os
import pickle
import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "svm.pkl"
)


class SVMModel:
    """
    Support Vector Machine multi-label classifier.

    Uses OneVsRestClassifier(CalibratedClassifierCV(LinearSVC)) for:
    - Multi-label classification on each label axis
    - Probability estimates via Platt scaling
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C = C
        self.max_iter = max_iter
        self._models: dict[str, OneVsRestClassifier] = {}
        self._label_axes: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        X,
        y_dict: dict[str, np.ndarray],
    ) -> "SVMModel":
        """
        Fit one OvR-SVM per label axis.

        Args:
            X: TF-IDF sparse matrix (n_samples, n_features)
            y_dict: Dict mapping axis name → binary label matrix
                    e.g. {"thematic": ndarray(n, 5), "severity": ndarray(n, 1)}
        """
        self._label_axes = list(y_dict.keys())
        for axis, y in y_dict.items():
            logger.info("Fitting SVM for axis: %s (shape=%s)", axis, y.shape)
            base = LinearSVC(C=self.C, max_iter=self.max_iter, class_weight="balanced")
            calibrated = CalibratedClassifierCV(base, cv=2)
            clf = OneVsRestClassifier(calibrated, n_jobs=1)
            clf.fit(X, y)
            self._models[axis] = clf

        self._is_fitted = True
        logger.info("SVMModel fitted for axes: %s", self._label_axes)
        return self

    def predict(self, X) -> dict[str, np.ndarray]:
        """
        Return binary predictions per label axis.

        Returns:
            Dict: axis → binary ndarray (n_samples, n_classes)
        """
        self._check_fitted()
        return {
            axis: clf.predict(X)
            for axis, clf in self._models.items()
        }

    def predict_proba(self, X) -> dict[str, np.ndarray]:
        """
        Return class probabilities per label axis.

        Returns:
            Dict: axis → probability ndarray (n_samples, n_classes)
        """
        self._check_fitted()
        return {
            axis: clf.predict_proba(X)
            for axis, clf in self._models.items()
        }

    def predict_single(self, X) -> dict[str, dict]:
        """
        Predict for a single sample, returning probabilities.

        Returns:
            Dict: axis → {"predictions": [...], "probabilities": [...]}
        """
        result = {}
        pred = self.predict(X)
        proba = self.predict_proba(X)
        for axis in self._label_axes:
            result[axis] = {
                "predictions": pred[axis][0].tolist(),
                "probabilities": proba[axis][0].tolist(),
            }
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SVMModel saved to %s", path)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "SVMModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("SVMModel loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("SVMModel must be fitted before predict.")
