"""
Bidirectional LSTM Multi-Label Classifier for Mental Stress Detection

Architecture:
- Dense projection layer: TF-IDF+LDA features → embedding dim
- 2-layer Bidirectional LSTM
- Dropout for regularization
- Separate sigmoid output heads per label axis
- BCEWithLogitsLoss with class weights to handle imbalance

Input: Combined TF-IDF + LDA dense feature vector per sample
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "lstm.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------

class StressLSTMNet(nn.Module):
    """
    Bidirectional LSTM network for multi-label stress classification.

    Because TF-IDF features are bag-of-words (no sequence), we treat the
    feature vector as a sequence of length 1 and let the LSTM learn a
    richer representation via the dense projection. This matches the
    'sequence learning from feature vectors' approach used in research.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.4,
        output_dims: dict[str, int] = None,
    ):
        super().__init__()
        if output_dims is None:
            output_dims = {"thematic": 5, "categorical": 8, "trigger": 8, "severity": 5}

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        # Project raw features to a compact embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Separate output head per label axis
        lstm_out_dim = hidden_dim * 2  # bidirectional
        self.heads = nn.ModuleDict({
            axis: nn.Sequential(
                nn.Linear(lstm_out_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(128, n_labels),
            )
            for axis, n_labels in output_dims.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, input_dim) — dense feature vector per sample

        Returns:
            Dict: axis → logits (batch, n_labels)
        """
        # Project input → (batch, 1, hidden_dim) [sequence of length 1]
        projected = self.input_proj(x).unsqueeze(1)

        # LSTM → take last hidden state concatenated from both directions
        lstm_out, _ = self.lstm(projected)
        context = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)

        return {axis: head(context) for axis, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Trainer / wrapper
# ---------------------------------------------------------------------------

class LSTMModel:
    """
    Training/inference wrapper around StressLSTMNet.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.4,
        output_dims: dict[str, int] = None,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.output_dims = output_dims or {"thematic": 5, "categorical": 8, "trigger": 8, "severity": 5}
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.net: Optional[StressLSTMNet] = None
        self._is_fitted = False
        self._label_axes: list[str] = []

    def _to_dense_tensor(self, X) -> torch.Tensor:
        if issparse(X):
            X = X.toarray()
        return torch.tensor(X, dtype=torch.float32)

    def fit(
        self,
        X,
        y_dict: dict[str, np.ndarray],
        class_weights: dict[str, Optional[torch.Tensor]] = None,
    ) -> "LSTMModel":
        """
        Train the LSTM network.

        Args:
            X: Feature matrix (sparse or dense), shape (n_samples, n_features)
            y_dict: Dict axis → binary label matrix
            class_weights: Optional dict axis → tensor of class weights
        """
        X_t = self._to_dense_tensor(X)
        self.input_dim = X_t.shape[1]
        self._label_axes = list(y_dict.keys())

        # Initialise network
        self.net = StressLSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            output_dims={axis: y.shape[1] for axis, y in y_dict.items()},
        ).to(DEVICE)

        # Build targets tensors
        y_tensors = {
            axis: torch.tensor(y, dtype=torch.float32)
            for axis, y in y_dict.items()
        }

        # Loss functions (BCE with class weights for imbalance)
        loss_fns = {}
        for axis in self._label_axes:
            if class_weights and axis in class_weights and class_weights[axis] is not None:
                weight = class_weights[axis].to(DEVICE)
            else:
                weight = None
            loss_fns[axis] = nn.BCEWithLogitsLoss(pos_weight=weight)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs,
            steps_per_epoch=max(1, len(X_t) // self.batch_size),
        )

        # Dataset and loader
        dataset = TensorDataset(X_t, *[y_tensors[a] for a in self._label_axes])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        logger.info("Training LSTM on %s (device=%s, epochs=%d)", DEVICE, DEVICE, self.epochs)
        self.net.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for batch in loader:
                x_batch = batch[0].to(DEVICE)
                y_batches = {
                    axis: batch[i + 1].to(DEVICE)
                    for i, axis in enumerate(self._label_axes)
                }
                optimizer.zero_grad()
                logits = self.net(x_batch)
                loss = sum(
                    loss_fns[axis](logits[axis], y_batches[axis])
                    for axis in self._label_axes
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(loader), 1)
            if epoch % 5 == 0 or epoch == 1:
                logger.info("Epoch [%d/%d] Loss: %.4f", epoch, self.epochs, avg_loss)

        self._is_fitted = True
        return self

    def predict_proba(self, X) -> dict[str, np.ndarray]:
        """
        Return sigmoid probabilities per label axis.

        Returns:
            Dict: axis → ndarray (n_samples, n_labels)
        """
        self._check_fitted()
        self.net.eval()
        X_t = self._to_dense_tensor(X).to(DEVICE)

        with torch.no_grad():
            logits = self.net(X_t)

        return {
            axis: torch.sigmoid(lgt).cpu().numpy()
            for axis, lgt in logits.items()
        }

    def predict(self, X, threshold: float = 0.5) -> dict[str, np.ndarray]:
        """
        Return binary predictions using a probability threshold.
        """
        probas = self.predict_proba(X)
        return {
            axis: (proba >= threshold).astype(int)
            for axis, proba in probas.items()
        }

    def predict_single(self, X, threshold: float = 0.5) -> dict[str, dict]:
        """Single-sample prediction with probabilities."""
        pred = self.predict(X, threshold)
        proba = self.predict_proba(X)
        result = {}
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
        self._check_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "output_dims": self.output_dims,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "_label_axes": self._label_axes,
            "net_state_dict": self.net.state_dict(),
        }
        torch.save(state, path)
        logger.info("LSTMModel saved to %s", path)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "LSTMModel":
        state = torch.load(path, map_location=DEVICE)
        model = cls(
            input_dim=state["input_dim"],
            hidden_dim=state["hidden_dim"],
            n_layers=state["n_layers"],
            dropout=state["dropout"],
            output_dims=state["output_dims"],
            learning_rate=state["lr"],
            epochs=state["epochs"],
            batch_size=state["batch_size"],
        )
        model._label_axes = state["_label_axes"]
        model.net = StressLSTMNet(
            input_dim=state["input_dim"],
            hidden_dim=state["hidden_dim"],
            n_layers=state["n_layers"],
            dropout=state["dropout"],
            output_dims=state["output_dims"],
        ).to(DEVICE)
        model.net.load_state_dict(state["net_state_dict"])
        model._is_fitted = True
        logger.info("LSTMModel loaded from %s", path)
        return model

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("LSTMModel must be trained before predict.")
