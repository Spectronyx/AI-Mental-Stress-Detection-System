"""
Model Evaluator — Compares SVM, Random Forest, and LSTM classifiers

Evaluates using:
- Accuracy, Precision, Recall, F1-score
- Standard deviation across CV folds
- Coefficient of Variation (CV%)
- Generates comparison table + charts

Usage:
    python backend/ml/evaluator.py

Output:
    evaluation_results.json
    Charts saved to backend/static/charts/
"""

import json
import logging
import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.preprocessor import preprocess_batch
from ml.feature_extractor import FeatureExtractor
from ml.models.svm_model import SVMModel
from ml.models.rf_model import RandomForestModel
from ml.models.lstm_model import LSTMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluator")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_PATH = os.path.join(BASE_DIR, "Suicide_Detection.csv")  # Default to real data
SYNTHETIC_DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_dataset.csv")
STATIC_CHARTS = os.path.join(BASE_DIR, "static", "charts")
RESULTS_PATH = os.path.join(BASE_DIR, "evaluation_results.json")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute multi-label classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def compute_cv_stats(scores: list[float]) -> dict:
    """Compute mean, std, and CV% for a list of CV fold scores."""
    arr = np.array(scores)
    mean = arr.mean()
    std = arr.std()
    cv_pct = (std / mean * 100) if mean > 0 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "cv_pct": round(cv_pct, 2)}


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_name, texts, y_thematic, n_splits=5):
    """
    Run stratified 5-fold CV for the given model.
    Folds are stratified on the first thematic label.
    """
    # Stratify on most common label per sample
    y_strat = np.argmax(y_thematic, axis=1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    logger.info("Evaluating %s (%d-fold CV)...", model_name, n_splits)

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y_strat)):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        # Feature extraction per fold
        extractor = FeatureExtractor(max_features=5_000, n_topics=10)
        extractor.fit(train_texts)

        if model_name == "svm":
            X_train = extractor.transform_tfidf(train_texts)
            X_test = extractor.transform_tfidf(test_texts)
            clf = SVMModel(C=1.0)
        else:
            X_train = extractor.transform_combined(train_texts)
            X_test = extractor.transform_combined(test_texts)
            if model_name == "rf":
                clf = RandomForestModel(n_estimators=100)
            else:  # lstm
                clf = LSTMModel(hidden_dim=128, n_layers=1, epochs=10, batch_size=64)

        y_train_dict = {"thematic": y_thematic[train_idx]}
        clf.fit(X_train, y_train_dict)

        y_pred_dict = clf.predict(X_test)
        y_pred = y_pred_dict["thematic"]
        y_true = y_thematic[test_idx]

        m = multilabel_metrics(y_true, y_pred)
        for k, v in m.items():
            fold_metrics[k].append(v)

        logger.info(
            "  Fold %d → F1=%.4f | Prec=%.4f | Rec=%.4f | Acc=%.4f",
            fold + 1, m["f1"], m["precision"], m["recall"], m["accuracy"],
        )

    summary = {metric: compute_cv_stats(scores) for metric, scores in fold_metrics.items()}
    logger.info("%s summary: %s", model_name, summary)
    return summary


# ---------------------------------------------------------------------------
# Comparison table + chart generation
# ---------------------------------------------------------------------------

def generate_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name.upper()}
        for metric, stats in metrics.items():
            row[f"{metric.capitalize()} (mean)"] = stats["mean"]
            row[f"{metric.capitalize()} (std)"] = stats["std"]
            row[f"{metric.capitalize()} (CV%)"] = stats["cv_pct"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def save_charts(results: dict, output_dir: str = STATIC_CHARTS) -> None:
    """Generate and save matplotlib comparison charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#6366f1", "#f59e0b", "#10b981"]

    # ---- Bar chart: F1 comparison ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Model Comparison — 5-Fold Cross-Validation", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        means = [results[m][metric]["mean"] for m in models]
        stds = [results[m][metric]["std"] for m in models]
        bars = ax.bar(
            [m.upper() for m in models],
            means,
            yerr=stds,
            capsize=6,
            color=colors[:len(models)],
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.set_title(metric.capitalize(), fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "model_comparison_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved chart: %s", bar_path)

    # ---- Radar chart ----
    try:
        from matplotlib.patches import FancyArrowPatch
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        for i, model_name in enumerate(models):
            values = [results[model_name][m]["mean"] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=model_name.upper(), color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("Radar: Model Performance", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        radar_path = os.path.join(output_dir, "model_comparison_radar.png")
        plt.savefig(radar_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved chart: %s", radar_path)
    except Exception as e:
        logger.warning("Radar chart generation failed: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load dataset
    from ml.trainer import load_data
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate stress detection models")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        logger.error("Dataset not found at %s.", DATA_PATH)
        sys.exit(1)

    logger.info("Loading dataset from %s...", DATA_PATH)
    df = load_data(limit=args.limit)
    texts = preprocess_batch(df["text"].tolist())

    # Encode thematic labels
    mlb = MultiLabelBinarizer()
    y_thematic = mlb.fit_transform(
        df["thematic_labels"].fillna("").apply(lambda x: x.split("|") if x else [])
    )
    logger.info("Thematic classes: %s", list(mlb.classes_))

    results = {}
    for model_name in ("svm", "rf", "lstm"):
        try:
            results[model_name] = evaluate_model(model_name, texts, y_thematic)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", model_name, e)

    if not results:
        logger.error("No models evaluated.")
        return

    # Print comparison table
    df_table = generate_comparison_table(results)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE (5-Fold CV on Thematic Labels)")
    print("=" * 70)
    print(df_table.to_string(index=False))

    # Save results JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_PATH)

    # Save charts
    save_charts(results)

    print("\n✅ Evaluation complete. Results and charts saved.")
    return results


if __name__ == "__main__":
    main()
