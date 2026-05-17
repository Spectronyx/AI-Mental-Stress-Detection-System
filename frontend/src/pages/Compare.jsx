import React, { useEffect, useState } from "react";
import ModelCompare from "../components/ModelCompare";
import { fetchModelComparison } from "../services/api";

export default function Compare() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelComparison()
      .then((res) => setData(res.results || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="page">
      <div className="container">
        <div style={{ marginBottom: "var(--sp-6)" }}>
          <h1 style={{ marginBottom: "0.4rem" }}>
            <span className="gradient-text">Model</span> Comparison
          </h1>
          <p>5-fold cross-validation results across SVM, Random Forest, and Bidirectional LSTM classifiers.</p>
        </div>

        {/* Architecture info cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginBottom: "var(--sp-6)" }}>
          {[
            {
              model: "SVM",
              color: "#6366f1",
              features: "TF-IDF (10k)",
              arch: "OneVsRest + Calibrated LinearSVC",
              strength: "Fast, memory-efficient, great for sparse features",
            },
            {
              model: "Random Forest",
              color: "#f59e0b",
              features: "TF-IDF + LDA (10k + 20)",
              arch: "OneVsRest + RF (200 trees)",
              strength: "Robust to noise, handles feature interaction, balanced class weights",
            },
            {
              model: "LSTM",
              color: "#10b981",
              features: "TF-IDF + LDA → Dense",
              arch: "Bidirectional LSTM (2-layer, 256 hidden)",
              strength: "Best at capturing sequential patterns, BCELoss with pos_weight",
            },
          ].map(({ model, color, features, arch, strength }) => (
            <div key={model} className="glass-card" style={{ padding: "1.25rem", borderTop: `3px solid ${color}` }}>
              <div style={{ fontWeight: 700, fontSize: "1.05rem", color, marginBottom: "0.5rem" }}>{model}</div>
              <div style={{ fontSize: "0.78rem", color: "var(--txt-muted)", marginBottom: "0.75rem", fontFamily: "var(--font-mono)" }}>
                Input: {features}
              </div>
              <div style={{ fontSize: "0.82rem", color: "var(--txt-secondary)", marginBottom: "0.5rem" }}>{arch}</div>
              <div style={{ fontSize: "0.78rem", color: "var(--txt-muted)", fontStyle: "italic" }}>{strength}</div>
            </div>
          ))}
        </div>

        <ModelCompare data={data} loading={loading} error={error} />

        {/* Methodology note */}
        <div className="glass-card" style={{ padding: "1.25rem 1.5rem", marginTop: "1.25rem" }}>
          <h4 style={{ marginBottom: "0.75rem" }}>📋 Evaluation Methodology</h4>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", fontSize: "0.85rem", color: "var(--txt-secondary)" }}>
            <div>
              <strong style={{ color: "var(--txt-primary)" }}>Dataset</strong>: 5,000 synthetic samples with realistic linguistic patterns across 5 thematic labels, 8 categorical labels, and 8 trigger categories.
            </div>
            <div>
              <strong style={{ color: "var(--txt-primary)" }}>Splitting</strong>: Stratified 5-fold cross-validation on thematic label. Train/Val splits held within each fold.
            </div>
            <div>
              <strong style={{ color: "var(--txt-primary)" }}>Class Imbalance</strong>: Addressed via balanced class weights (SVM/RF) and BCEWithLogitsLoss pos_weight (LSTM).
            </div>
            <div>
              <strong style={{ color: "var(--txt-primary)" }}>Metrics</strong>: Micro-averaged Accuracy, Precision, Recall, F1. StdDev and CV% measure fold-to-fold consistency.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
