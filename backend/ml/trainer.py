"""
Model Trainer — Trains SVM, Random Forest, and LSTM on merged datasets.

Merges 3 datasets:
  1. Combined Data.csv   (53k rows, 7 mental health classes)
  2. Suicide_Detection.csv (232k rows, binary → mapped to Suicidal/Normal)
  3. Dreaddit             (715 rows, binary → mapped to Stress/Normal)

Usage:
    python backend/ml/trainer.py [--model all|svm|rf|lstm] [--limit N]

Output:
    saved_models/svm.pkl
    saved_models/rf.pkl
    saved_models/lstm.pt
    saved_models/tfidf_vectorizer.pkl
    saved_models/lda_model.pkl
    saved_models/label_encoders.pkl
    saved_models/model_weights.pkl  ← per-model F1 for ensemble
"""

import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# Make backend root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.preprocessor import preprocess_batch
from ml.feature_extractor import FeatureExtractor
from ml.models.svm_model import SVMModel
from ml.models.rf_model import RandomForestModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trainer")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
ENCODERS_PATH = os.path.join(SAVED_MODELS_DIR, "label_encoders.pkl")
WEIGHTS_PATH = os.path.join(SAVED_MODELS_DIR, "model_weights.pkl")

# 7-class label set
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


# ---------------------------------------------------------------------------
# Data loading — merge 3 datasets
# ---------------------------------------------------------------------------

def load_data(limit: int = None) -> pd.DataFrame:
    """
    Merge all 3 datasets into a unified DataFrame with columns:
      - text: the raw text
      - label: one of LABEL_CLASSES
    """
    frames = []

    # --- Dataset 1: Combined Data.csv (53k, 7 classes) ---
    combined_path = os.path.join(DATASETS_DIR, "Combined Data.csv")
    if os.path.exists(combined_path):
        df1 = pd.read_csv(combined_path)
        df1 = df1.rename(columns={"statement": "text", "status": "label"})
        df1 = df1[["text", "label"]].dropna()
        # Normalize label names
        label_map = {
            "Normal": "Normal",
            "Depression": "Depression",
            "Suicidal": "Suicidal",
            "Anxiety": "Anxiety",
            "Bipolar": "Bipolar",
            "Stress": "Stress",
            "Personality disorder": "Personality Disorder",
        }
        df1["label"] = df1["label"].map(label_map)
        df1 = df1.dropna(subset=["label"])
        logger.info("Combined Data.csv: %d rows", len(df1))
        frames.append(df1)
    else:
        logger.warning("Combined Data.csv not found at %s", combined_path)

    # --- Dataset 2: Suicide_Detection.csv (232k, binary) ---
    suicide_path = os.path.join(DATASETS_DIR, "Suicide_Detection.csv")
    if os.path.exists(suicide_path):
        df2 = pd.read_csv(suicide_path)
        df2 = df2[["text", "class"]].dropna()
        # Sample to avoid overwhelming other datasets
        sample_size = 15000
        df2_suicide = df2[df2["class"] == "suicide"].sample(
            n=min(sample_size // 2, len(df2[df2["class"] == "suicide"])),
            random_state=42,
        )
        df2_normal = df2[df2["class"] == "non-suicide"].sample(
            n=min(sample_size // 2, len(df2[df2["class"] == "non-suicide"])),
            random_state=42,
        )
        df2 = pd.concat([df2_suicide, df2_normal], ignore_index=True)
        df2["label"] = df2["class"].map({"suicide": "Suicidal", "non-suicide": "Normal"})
        df2 = df2[["text", "label"]]
        logger.info("Suicide_Detection.csv: %d rows (sampled)", len(df2))
        frames.append(df2)
    else:
        logger.warning("Suicide_Detection.csv not found at %s", suicide_path)

    # --- Dataset 3: Dreaddit (715, binary stress) ---
    dreaddit_path = os.path.join(DATASETS_DIR, "dreaddit_StressAnalysis - Sheet1.csv")
    if os.path.exists(dreaddit_path):
        df3 = pd.read_csv(dreaddit_path)
        df3 = df3[["text", "label"]].dropna()
        df3["label"] = df3["label"].map({1: "Stress", 0: "Normal"})
        logger.info("Dreaddit: %d rows", len(df3))
        frames.append(df3)
    else:
        logger.warning("Dreaddit not found at %s", dreaddit_path)

    if not frames:
        logger.error("No datasets found in %s!", DATASETS_DIR)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["text"].str.strip().astype(bool)]  # Drop empty text
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    logger.info("Total merged dataset: %d rows", len(df))
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())

    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        logger.info("Limited to %d rows", limit)

    return df


# ---------------------------------------------------------------------------
# Label encoding (single axis — 7-class)
# ---------------------------------------------------------------------------

def encode_labels(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode the 'label' column into integer class indices.

    Returns:
        y: 1-D integer array of class indices
        encoder: fitted LabelEncoder
    """
    le = LabelEncoder()
    le.classes_ = np.array(LABEL_CLASSES)
    y = le.transform(df["label"])
    logger.info("Classes: %s", list(le.classes_))
    return y, le


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_svm(X, y, train_idx, test_idx):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    logger.info("=" * 50)
    logger.info("Training SVM...")

    base = LinearSVC(C=1.0, max_iter=3000, class_weight="balanced")
    calibrated = CalibratedClassifierCV(base, cv=3)
    clf = OneVsRestClassifier(calibrated, n_jobs=1)

    clf.fit(X[train_idx], y[train_idx])

    # Evaluate
    y_pred = clf.predict(X[test_idx])
    f1 = f1_score(y[test_idx], y_pred, average="weighted")
    logger.info("✅ SVM — Validation F1: %.4f", f1)

    # Save
    path = os.path.join(SAVED_MODELS_DIR, "svm.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(clf, fp)
    logger.info("SVM saved to %s", path)

    return clf, f1


def train_rf(X, y, train_idx, test_idx):
    from sklearn.ensemble import RandomForestClassifier

    logger.info("=" * 50)
    logger.info("Training Random Forest...")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X[train_idx], y[train_idx])

    y_pred = clf.predict(X[test_idx])
    f1 = f1_score(y[test_idx], y_pred, average="weighted")
    logger.info("✅ Random Forest — Validation F1: %.4f", f1)

    path = os.path.join(SAVED_MODELS_DIR, "rf.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(clf, fp)
    logger.info("RF saved to %s", path)

    return clf, f1


def train_lstm(X, y, train_idx, test_idx, n_classes: int):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        logger.error("PyTorch not installed — skipping LSTM: %s", e)
        return None, 0.0

    from scipy.sparse import issparse

    logger.info("=" * 50)
    logger.info("Training LSTM...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to dense
    X_dense = X.toarray() if issparse(X) else X
    X_train = torch.tensor(X_dense[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_test = torch.tensor(X_dense[test_idx], dtype=torch.float32)
    y_test_np = y[test_idx]

    input_dim = X_train.shape[1]
    hidden_dim = 256
    epochs = 15
    batch_size = 128
    lr = 1e-3

    # Simple classification network
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
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    # Class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[train_idx])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(loader),
    )

    net.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            logger.info("  Epoch [%d/%d] Loss: %.4f", epoch, epochs, total_loss / len(loader))

    # Evaluate
    net.eval()
    with torch.no_grad():
        logits = net(X_test.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
    f1 = f1_score(y_test_np, y_pred, average="weighted")
    logger.info("✅ LSTM — Validation F1: %.4f", f1)

    # Save
    path = os.path.join(SAVED_MODELS_DIR, "lstm.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "net_state_dict": net.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "n_classes": n_classes,
    }, path)
    logger.info("LSTM saved to %s", path)

    return net, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_choice: str = "all", limit: int = None):
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # --- Load & merge data ---
    df = load_data(limit=limit)
    logger.info("Dataset loaded: %d samples", len(df))

    # --- Preprocess text ---
    logger.info("Preprocessing text...")
    texts = preprocess_batch(df["text"].tolist())

    # --- Encode labels ---
    y, label_encoder = encode_labels(df)
    n_classes = len(label_encoder.classes_)

    # Save encoder
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info("Label encoder saved to %s", ENCODERS_PATH)

    # --- Train/test split (80/20) ---
    n = len(texts)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info("Split: %d train / %d test", len(train_idx), len(test_idx))

    # --- Feature extraction ---
    logger.info("Fitting FeatureExtractor on training set...")
    extractor = FeatureExtractor(max_features=15_000, n_topics=20)
    train_texts = [texts[i] for i in train_idx]
    extractor.fit(train_texts)
    extractor.save()

    logger.info("Extracting combined TF-IDF + LDA features...")
    X_combined = extractor.transform_combined(texts)

    from scipy.sparse import csr_matrix
    X_combined = csr_matrix(X_combined)
    logger.info("Feature shape: %s", X_combined.shape)

    # --- Train models ---
    model_f1 = {}

    if model_choice in ("all", "svm"):
        _, f1 = train_svm(X_combined, y, train_idx, test_idx)
        model_f1["svm"] = f1

    if model_choice in ("all", "rf"):
        _, f1 = train_rf(X_combined, y, train_idx, test_idx)
        model_f1["rf"] = f1

    if model_choice in ("all", "lstm"):
        _, f1 = train_lstm(X_combined, y, train_idx, test_idx, n_classes)
        model_f1["lstm"] = f1

    # Save ensemble weights
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump(model_f1, f)
    logger.info("Model F1 weights saved: %s", model_f1)

    logger.info("=" * 50)
    logger.info("🎉 All requested models trained successfully!")
    logger.info("Models saved to: %s", SAVED_MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stress detection models")
    parser.add_argument(
        "--model",
        choices=["all", "svm", "rf", "lstm"],
        default="all",
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of training samples",
    )
    args = parser.parse_args()
    main(args.model, args.limit)
