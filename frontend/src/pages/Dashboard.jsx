import React, { useState, useEffect } from "react";
import TextInput from "../components/TextInput";
import AnalysisResult from "../components/AnalysisResult";
import TrendChart from "../components/TrendChart";
import { useAnalysis } from "../hooks/useAnalysis";
import { fetchStats } from "../services/api";
import { Activity, AlertTriangle, BarChart2, Zap } from "lucide-react";

export default function Dashboard() {
  const { result, loading, error, analyze, reset } = useAnalysis();
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(true);

  // Load dashboard stats
  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch(() => setStats(null))
      .finally(() => setStatsLoading(false));
  }, [result]); // Refresh after each analysis

  const handleSubmit = async (text, model) => {
    await analyze(text, model);
  };

  return (
    <div className="page">
      <div className="container">

        {/* Hero section */}
        <div style={{ textAlign: "center", marginBottom: "var(--sp-8)" }}>
          <h1 style={{ marginBottom: "0.75rem" }}>
            <span className="gradient-text">AI Mental Stress</span>
            <br />
            Detection System
          </h1>
          <p style={{ fontSize: "1.05rem", maxWidth: 580, margin: "0 auto", lineHeight: 1.7 }}>
            Multi-label NLP analysis using SVM, Random Forest, and Bidirectional LSTM
            to detect stress patterns, severity, and triggers from text.
          </p>
        </div>

        {/* Stats bar */}
        {!statsLoading && stats && (
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: "var(--sp-4)",
            marginBottom: "var(--sp-6)",
          }}>
            {[
              { label: "Total Analyses", value: stats.total_analyses, icon: Activity, iconColor: "var(--clr-primary)" },
              { label: "Alerts Triggered", value: stats.alerts_triggered, icon: AlertTriangle, iconColor: "#ef4444" },
              { label: "Avg Severity", value: stats.average_severity?.toFixed(1) ?? "—", icon: BarChart2, iconColor: "#f59e0b" },
              { label: "Models Available", value: "3", icon: Zap, iconColor: "var(--clr-accent)" },
            ].map(({ label, value, icon: Icon, iconColor }) => (
              <div key={label} className="glass-card stat-card" style={{ flexDirection: "row", alignItems: "center", gap: "1rem" }}>
                <div style={{
                  width: 44,
                  height: 44,
                  borderRadius: "var(--radius-md)",
                  background: `${iconColor}18`,
                  border: `1px solid ${iconColor}30`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}>
                  <Icon size={20} color={iconColor} />
                </div>
                <div>
                  <div className="stat-value" style={{ fontSize: "1.5rem", color: iconColor }}>{value}</div>
                  <div className="stat-label" style={{ fontSize: "0.72rem" }}>{label}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Main grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: result ? "1fr 1fr" : "1fr",
          gap: "var(--sp-6)",
          transition: "grid-template-columns 0.4s ease",
        }}>
          {/* Left: Input + error */}
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-4)" }}>
            <TextInput onSubmit={handleSubmit} loading={loading} onReset={reset} />

            {error && (
              <div style={{
                padding: "1rem 1.25rem",
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.3)",
                borderRadius: "var(--radius-md)",
                color: "#f87171",
                fontSize: "0.875rem",
              }}>
                ❌ {error}
              </div>
            )}

            {/* Trend chart on left column (below input when result is shown) */}
            {result && stats?.recent_trend?.length > 0 && (
              <TrendChart data={stats.recent_trend} />
            )}
          </div>

          {/* Right: Results */}
          {result && (
            <div>
              <AnalysisResult result={result} />
            </div>
          )}
        </div>

        {/* Trend chart when no result shown yet */}
        {!result && stats?.recent_trend?.length > 0 && (
          <div style={{ marginTop: "var(--sp-6)" }}>
            <TrendChart data={stats.recent_trend} />
          </div>
        )}

        {/* Empty state CTA */}
        {!result && !loading && (
          <div style={{ marginTop: "var(--sp-8)", textAlign: "center" }}>
            <div style={{ display: "flex", justifyContent: "center", gap: "var(--sp-4)", flexWrap: "wrap" }}>
              {[
                { icon: "🔬", label: "Multi-label Classification", desc: "Anxiety · Depression · Anger · Sadness" },
                { icon: "🎯", label: "Trigger Detection", desc: "Work · Financial · Academic · Relationships" },
                { icon: "⚡", label: "Instant Analysis", desc: "Real-time NLP pipeline < 200ms" },
                { icon: "🚨", label: "Early Warning System", desc: "Severity threshold alerts with resources" },
              ].map(({ icon, label, desc }) => (
                <div key={label} className="glass-card" style={{
                  padding: "1.25rem",
                  width: 200,
                  textAlign: "center",
                }}>
                  <div style={{ fontSize: "1.75rem", marginBottom: "0.5rem" }}>{icon}</div>
                  <div style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--txt-primary)", marginBottom: "0.3rem" }}>{label}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--txt-muted)" }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
