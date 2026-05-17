import React, { useEffect, useRef } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const SEV_COLORS = {
  1: "#10b981",
  2: "#84cc16",
  3: "#f59e0b",
  4: "#ef4444",
  5: "#dc2626",
};

/**
 * TrendChart — Line chart showing historical severity scores over time.
 *
 * Props:
 *   data  {Array<{timestamp, severity}>}  — trend data points
 *   title {string}  — optional chart title override
 */
export default function TrendChart({ data = [], title = "Stress Severity Trend" }) {
  if (!data || data.length === 0) {
    return (
      <div className="glass-card" style={{ padding: "1.5rem", textAlign: "center" }}>
        <p style={{ color: "var(--txt-muted)", fontSize: "0.88rem" }}>
          No trend data yet. Submit analyses to see your history.
        </p>
      </div>
    );
  }

  const labels = data.map((d) => {
    const date = new Date(d.timestamp);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  });

  const severities = data.map((d) => d.severity);

  // Build gradient colors array per point
  const pointColors = severities.map((s) => SEV_COLORS[Math.round(s)] || "#6b7280");

  const chartData = {
    labels,
    datasets: [
      {
        label: "Severity",
        data: severities,
        borderColor: "rgba(14,165,233,0.9)",
        backgroundColor: "rgba(14,165,233,0.08)",
        pointBackgroundColor: pointColors,
        pointBorderColor: pointColors,
        pointRadius: 5,
        pointHoverRadius: 8,
        borderWidth: 2,
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(13,17,23,0.95)",
        borderColor: "rgba(99,120,160,0.25)",
        borderWidth: 1,
        titleColor: "#e8edf5",
        bodyColor: "#8a96a8",
        padding: 12,
        callbacks: {
          label: (ctx) => {
            const labels = ["", "Minimal", "Mild", "Moderate", "High", "Critical"];
            return ` Severity ${ctx.raw} — ${labels[ctx.raw] || ""}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(255,255,255,0.04)" },
        ticks: { color: "rgba(138,150,168,0.8)", font: { size: 11 } },
      },
      y: {
        min: 0,
        max: 5,
        grid: { color: "rgba(255,255,255,0.05)" },
        ticks: {
          color: "rgba(138,150,168,0.8)",
          font: { size: 11 },
          stepSize: 1,
          callback: (v) => ["", "Min", "Mild", "Mod", "High", "Crit"][v] || v,
        },
      },
    },
  };

  return (
    <div className="glass-card" style={{ padding: "1.25rem 1.5rem" }}>
      <h4 style={{ marginBottom: "1rem", color: "var(--txt-secondary)", fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        📈 {title}
      </h4>
      <div style={{ height: 220 }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
