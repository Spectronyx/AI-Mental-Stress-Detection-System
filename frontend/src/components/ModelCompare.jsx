import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Bar, Radar } from "react-chartjs-2";
import { Loader } from "lucide-react";

ChartJS.register(
  CategoryScale, LinearScale, BarElement,
  RadialLinearScale, PointElement, LineElement,
  Tooltip, Legend, Filler
);

const MODEL_COLORS = {
  SVM:  { bg: "rgba(99,102,241,0.65)",  border: "#6366f1" },
  RF:   { bg: "rgba(245,158,11,0.65)",  border: "#f59e0b" },
  LSTM: { bg: "rgba(16,185,129,0.65)",  border: "#10b981" },
};

const METRICS = ["accuracy", "precision", "recall", "f1"];
const METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-Score"];

/**
 * ModelCompare — Bar + Radar chart comparison of SVM, RF, and LSTM.
 */
export default function ModelCompare({ data, loading, error }) {
  if (loading) {
    return (
      <div className="glass-card" style={{ padding: "3rem", display: "flex", alignItems: "center", justifyContent: "center", gap: "0.75rem" }}>
        <Loader size={20} style={{ animation: "spin 1s linear infinite", color: "var(--clr-primary)" }} />
        <span style={{ color: "var(--txt-secondary)" }}>Loading comparison data…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card" style={{ padding: "2rem", textAlign: "center" }}>
        <p style={{ color: "#f87171", marginBottom: "0.5rem" }}>⚠️ {error}</p>
        <p style={{ fontSize: "0.82rem", color: "var(--txt-muted)" }}>
          Run <code style={{ fontFamily: "var(--font-mono)", color: "var(--clr-primary)" }}>python backend/ml/evaluator.py</code> to generate comparison data.
        </p>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="glass-card" style={{ padding: "2rem", textAlign: "center" }}>
        <p style={{ color: "var(--txt-muted)", fontSize: "0.88rem" }}>No comparison data available.</p>
      </div>
    );
  }

  const modelNames = data.map((d) => d.model);

  // ---- Bar chart ----
  const barDatasets = METRICS.map((metric, i) => ({
    label: METRIC_LABELS[i],
    data: data.map((d) => +(d[metric]?.mean || 0).toFixed(4)),
    backgroundColor: ["rgba(14,165,233,0.7)", "rgba(139,92,246,0.7)", "rgba(245,158,11,0.7)", "rgba(16,185,129,0.7)"][i],
    borderRadius: 6,
    borderSkipped: false,
  }));

  const barData = { labels: modelNames, datasets: barDatasets };

  const barOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: "rgba(138,150,168,0.9)", font: { size: 11 }, padding: 16, boxWidth: 12 },
      },
      tooltip: {
        backgroundColor: "rgba(13,17,23,0.95)",
        borderColor: "rgba(99,120,160,0.25)",
        borderWidth: 1,
        titleColor: "#e8edf5",
        bodyColor: "#8a96a8",
        padding: 12,
        callbacks: {
          label: (ctx) => ` ${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(2)}%`,
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "rgba(138,150,168,0.8)", font: { size: 12, weight: "600" } },
      },
      y: {
        min: 0,
        max: 1,
        grid: { color: "rgba(255,255,255,0.05)" },
        ticks: {
          color: "rgba(138,150,168,0.8)",
          font: { size: 10 },
          callback: (v) => `${(v * 100).toFixed(0)}%`,
        },
      },
    },
  };

  // ---- Radar chart ----
  const radarData = {
    labels: METRIC_LABELS,
    datasets: data.map((d) => ({
      label: d.model,
      data: METRICS.map((m) => +(d[m]?.mean || 0).toFixed(4)),
      backgroundColor: (MODEL_COLORS[d.model] || MODEL_COLORS.LSTM).bg.replace("0.65", "0.15"),
      borderColor: (MODEL_COLORS[d.model] || MODEL_COLORS.LSTM).border,
      borderWidth: 2,
      pointBackgroundColor: (MODEL_COLORS[d.model] || MODEL_COLORS.LSTM).border,
      pointRadius: 4,
    })),
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: "rgba(138,150,168,0.9)", font: { size: 11 }, padding: 16, boxWidth: 12 },
      },
      tooltip: {
        backgroundColor: "rgba(13,17,23,0.95)",
        borderColor: "rgba(99,120,160,0.25)",
        borderWidth: 1,
        titleColor: "#e8edf5",
        bodyColor: "#8a96a8",
        padding: 12,
        callbacks: {
          label: (ctx) => ` ${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(2)}%`,
        },
      },
    },
    scales: {
      r: {
        min: 0,
        max: 1,
        grid: { color: "rgba(255,255,255,0.07)" },
        angleLines: { color: "rgba(255,255,255,0.07)" },
        pointLabels: { color: "rgba(138,150,168,0.9)", font: { size: 11 } },
        ticks: { display: false },
      },
    },
  };

  // ---- Comparison table ----
  const metricsForTable = ["accuracy", "precision", "recall", "f1"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Charts row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem" }}>
        <div className="glass-card" style={{ padding: "1.25rem 1.5rem" }}>
          <h4 style={{ marginBottom: "1rem", color: "var(--txt-secondary)", fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            📊 Metric Comparison (Bar)
          </h4>
          <div style={{ height: 260 }}>
            <Bar data={barData} options={barOptions} />
          </div>
        </div>

        <div className="glass-card" style={{ padding: "1.25rem 1.5rem" }}>
          <h4 style={{ marginBottom: "1rem", color: "var(--txt-secondary)", fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            🕸️ Radar Chart
          </h4>
          <div style={{ height: 260 }}>
            <Radar data={radarData} options={radarOptions} />
          </div>
        </div>
      </div>

      {/* Detailed table */}
      <div className="glass-card" style={{ overflow: "hidden" }}>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--clr-border)" }}>
                <th style={thStyle}>Model</th>
                {metricsForTable.map((m) => (
                  <React.Fragment key={m}>
                    <th style={thStyle}>{m.charAt(0).toUpperCase() + m.slice(1)} ↑</th>
                    <th style={{ ...thStyle, color: "var(--txt-muted)" }}>Std</th>
                    <th style={{ ...thStyle, color: "var(--txt-muted)" }}>CV%</th>
                  </React.Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr
                  key={row.model}
                  style={{
                    borderBottom: i < data.length - 1 ? "1px solid var(--clr-border)" : "none",
                    background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)",
                  }}
                >
                  <td style={{ ...tdStyle, fontWeight: 700, color: (MODEL_COLORS[row.model] || MODEL_COLORS.LSTM).border }}>
                    {row.model}
                  </td>
                  {metricsForTable.map((m) => (
                    <React.Fragment key={m}>
                      <td style={{ ...tdStyle, color: "var(--txt-primary)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                        {((row[m]?.mean || 0) * 100).toFixed(2)}%
                      </td>
                      <td style={{ ...tdStyle, color: "var(--txt-muted)", fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>
                        ±{((row[m]?.std || 0) * 100).toFixed(2)}
                      </td>
                      <td style={{ ...tdStyle, color: "var(--txt-muted)", fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>
                        {(row[m]?.cv_pct || 0).toFixed(1)}%
                      </td>
                    </React.Fragment>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p style={{ fontSize: "0.75rem", color: "var(--txt-muted)", textAlign: "center" }}>
        Results from 5-fold stratified cross-validation on thematic label classification.
      </p>
    </div>
  );
}

const thStyle = {
  padding: "0.75rem 1rem",
  textAlign: "left",
  fontSize: "0.75rem",
  fontWeight: 700,
  color: "var(--txt-secondary)",
  textTransform: "uppercase",
  letterSpacing: "0.07em",
};

const tdStyle = {
  padding: "0.7rem 1rem",
  fontSize: "0.85rem",
  color: "var(--txt-secondary)",
};
