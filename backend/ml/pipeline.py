"""
Analysis Pipeline Orchestrator — Ensemble Prediction

Loads saved models + feature extractor, runs preprocessing → feature
extraction → model inference → ensemble vote → severity → alert.

Ensemble mode runs all 3 models (SVM, RF, LSTM) and uses weighted
majority voting based on each model's validation F1 score.

Usage (programmatic):
    from ml.pipeline import AnalysisPipeline
    pipeline = AnalysisPipeline()
    result = pipeline.analyze("I feel hopeless", model="ensemble")
"""

import logging
import os
import pickle
import sys
import time
from typing import Literal, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
ENCODERS_PATH = os.path.join(SAVED_MODELS_DIR, "label_encoders.pkl")
WEIGHTS_PATH = os.path.join(SAVED_MODELS_DIR, "model_weights.pkl")

LABEL_CLASSES = [
    "Normal", "Depression", "Suicidal", "Anxiety",
    "Bipolar", "Stress", "Personality Disorder",
]

SEVERITY_MAP = {
    "Normal": 1,
    "Stress": 3,
    "Anxiety": 3,
    "Personality Disorder": 2,
    "Bipolar": 4,
    "Depression": 4,
    "Suicidal": 5,
}

ModelChoice = Literal["svm", "rf", "lstm", "ensemble"]


class AnalysisPipeline:
    """
    End-to-end inference pipeline for mental stress analysis.
    Supports individual model or ensemble prediction.
    """

    def __init__(self, alert_threshold: int = 4, llm_enabled: bool = False):
        self._extractor = None
        self._encoder = None
        self._model_cache: dict = {}
        self._model_weights: dict = {}
        self._alert_engine = None
        self._alert_threshold = alert_threshold
        self._llm_enabled = llm_enabled

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_extractor(self):
        if self._extractor is None:
            from ml.feature_extractor import FeatureExtractor
            self._extractor = FeatureExtractor.load()
        return self._extractor

    def _load_encoder(self):
        if self._encoder is None:
            with open(ENCODERS_PATH, "rb") as f:
                self._encoder = pickle.load(f)
        return self._encoder

    def _load_model_weights(self) -> dict:
        if not self._model_weights:
            if os.path.exists(WEIGHTS_PATH):
                with open(WEIGHTS_PATH, "rb") as f:
                    self._model_weights = pickle.load(f)
            else:
                self._model_weights = {"svm": 1.0, "rf": 1.0, "lstm": 1.0}
        return self._model_weights

    def _load_model(self, model_name: str):
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        path_map = {
            "svm": os.path.join(SAVED_MODELS_DIR, "svm.pkl"),
            "rf": os.path.join(SAVED_MODELS_DIR, "rf.pkl"),
            "lstm": os.path.join(SAVED_MODELS_DIR, "lstm.pt"),
        }

        path = path_map.get(model_name)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        if model_name in ("svm", "rf"):
            with open(path, "rb") as f:
                model = pickle.load(f)
        elif model_name == "lstm":
            model = self._load_lstm(path)

        self._model_cache[model_name] = model
        return model

    def _load_lstm(self, path: str):
        """Load the LSTM model and return a callable wrapper."""
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device, weights_only=False)

        input_dim = state["input_dim"]
        hidden_dim = state["hidden_dim"]
        n_classes = state["n_classes"]

        class StressNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                self.lstm = nn.LSTM(
                    input_size=hidden_dim, hidden_size=hidden_dim,
                    num_layers=2, batch_first=True, bidirectional=True,
                    dropout=0.3,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, n_classes),
                )

            def forward(self, x):
                x = self.proj(x).unsqueeze(1)
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        net = StressNet().to(device)
        net.load_state_dict(state["net_state_dict"])
        net.eval()
        return net

    def _get_alert_engine(self):
        if self._alert_engine is None:
            from ml.alert_engine import AlertEngine
            self._alert_engine = AlertEngine(
                config={
                    "alert_threshold": self._alert_threshold,
                    "llm_enabled": self._llm_enabled,
                }
            )
        return self._alert_engine

    # ------------------------------------------------------------------
    # Single-model prediction
    # ------------------------------------------------------------------

    def _predict_single_model(self, model_name: str, X) -> dict:
        """Run a single model and return class probabilities."""
        import torch
        from scipy.sparse import issparse

        model = self._load_model(model_name)

        if model_name == "lstm":
            X_dense = X.toarray() if issparse(X) else X
            X_t = torch.tensor(X_dense, dtype=torch.float32)
            device = next(model.parameters()).device
            with torch.no_grad():
                logits = model(X_t.to(device))
                proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            # sklearn model — predict_proba
            proba = model.predict_proba(X)[0]

        return {
            "class_idx": int(np.argmax(proba)),
            "probabilities": proba.tolist(),
        }

    # ------------------------------------------------------------------
    # Ensemble prediction
    # ------------------------------------------------------------------

    def _predict_ensemble(self, X) -> dict:
        """Run all available models and do weighted voting."""
        weights = self._load_model_weights()
        available = []

        for name in ("svm", "rf", "lstm"):
            try:
                result = self._predict_single_model(name, X)
                available.append((name, result, weights.get(name, 1.0)))
            except (FileNotFoundError, Exception) as e:
                logger.warning("Model %s unavailable for ensemble: %s", name, e)

        if not available:
            raise FileNotFoundError("No models available for ensemble prediction.")

        n_classes = len(LABEL_CLASSES)

        # Weighted probability averaging
        weighted_proba = np.zeros(n_classes)
        total_weight = 0.0
        for name, result, w in available:
            proba = np.array(result["probabilities"])
            # Handle shape mismatch
            if len(proba) == n_classes:
                weighted_proba += proba * w
                total_weight += w

        if total_weight > 0:
            weighted_proba /= total_weight

        return {
            "class_idx": int(np.argmax(weighted_proba)),
            "probabilities": weighted_proba.tolist(),
            "models_used": [name for name, _, _ in available],
        }

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        model: ModelChoice = "ensemble",
        threshold: float = 0.5,
    ) -> dict:
        """
        Analyze text for mental stress indicators.

        Args:
            text: Raw input text
            model: 'svm', 'rf', 'lstm', or 'ensemble' (default)
            threshold: Not used for single-label, kept for API compat
        """
        start_time = time.perf_counter()

        # --- Preprocessing ---
        from ml.preprocessor import preprocess
        processed = preprocess(text)

        if not processed.strip():
            return self._empty_result("Input text is too short or contains no meaningful content.")

        # --- Feature extraction ---
        extractor = self._load_extractor()
        X = extractor.transform_combined([processed])

        # --- Model inference ---
        if model == "ensemble":
            prediction = self._predict_ensemble(X)
        else:
            prediction = self._predict_single_model(model, X)

        # --- Decode ---
        class_idx = prediction["class_idx"]
        probabilities = prediction["probabilities"]
        predicted_label = LABEL_CLASSES[class_idx]
        confidence = float(probabilities[class_idx])

        # Build score dict
        label_scores = {
            label: round(float(probabilities[i]), 4)
            for i, label in enumerate(LABEL_CLASSES)
            if i < len(probabilities)
        }

        # --- Severity (deterministic from label) ---
        severity = SEVERITY_MAP.get(predicted_label, 1)

        # --- Alert engine ---
        alert = self._get_alert_engine().check(
            severity=severity,
            thematic_labels=[predicted_label],
            original_text=text,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "success": True,
            "text_preview": text[:200],
            "model_used": model if model != "ensemble" else f"ensemble ({', '.join(prediction.get('models_used', []))})",
            "processing_time_ms": elapsed_ms,
            "thematic_labels": [predicted_label],
            "thematic_scores": label_scores,
            "categorical_labels": [],
            "categorical_scores": {},
            "trigger_labels": [],
            "trigger_scores": {},
            "severity": severity,
            "severity_probabilities": probabilities,
            "confidence_score": round(confidence, 4),
            "alert": alert,
            "disclaimer": (
                "⚠️ This tool is for research and informational purposes only. "
                "It is NOT a medical diagnostic tool. Always consult a licensed "
                "mental health professional for clinical guidance."
            ),
        }

    def analyze_batch(
        self,
        texts: list[str],
        model: ModelChoice = "ensemble",
        threshold: float = 0.5,
    ) -> list[dict]:
        """Analyze multiple texts."""
        return [self.analyze(t, model, threshold) for t in texts]

    def _empty_result(self, reason: str) -> dict:
        from ml.alert_engine import AlertEngine
        return {
            "success": False,
            "error": reason,
            "thematic_labels": [],
            "categorical_labels": [],
            "trigger_labels": [],
            "severity": 1,
            "confidence_score": 0.0,
            "alert": AlertEngine().check(1, [], ""),
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Check which models are available on disk."""
        checks = {}
        model_paths = {
            "svm": os.path.join(SAVED_MODELS_DIR, "svm.pkl"),
            "rf": os.path.join(SAVED_MODELS_DIR, "rf.pkl"),
            "lstm": os.path.join(SAVED_MODELS_DIR, "lstm.pt"),
            "feature_extractor": os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"),
            "label_encoders": ENCODERS_PATH,
        }
        for name, path in model_paths.items():
            checks[name] = os.path.exists(path)
        checks["all_ready"] = all(checks.values())
        return checks


# ---------------------------------------------------------------------------
# Singleton instance (used by Django views)
# ---------------------------------------------------------------------------
_pipeline_instance: Optional[AnalysisPipeline] = None


def get_pipeline(alert_threshold: int = 4) -> AnalysisPipeline:
    """Return a module-level singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AnalysisPipeline(
            alert_threshold=alert_threshold,
            llm_enabled=bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        )
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = AnalysisPipeline()
    print("\nHealth check:", pipeline.health_check())

    test_texts = [
        "I feel completely hopeless. Nothing matters anymore and I can't see a way forward.",
        "My anxiety about work is through the roof. I haven't slept in days.",
        "I had an okay day. Work was a bit stressful but nothing unusual.",
        "I've been drinking heavily every night. I can't control it anymore.",
        "I'm having a great day today, the weather is nice!",
    ]

    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text[:60]}...")
        try:
            result = pipeline.analyze(text, model="ensemble")
            print(f"Predicted: {result['thematic_labels']}")
            print(f"Severity: {result['severity']}")
            print(f"Confidence: {result['confidence_score']}")
            print(f"Alert: {result['alert']['alert_triggered']}")
        except Exception as e:
            print(f"Error: {e}")
