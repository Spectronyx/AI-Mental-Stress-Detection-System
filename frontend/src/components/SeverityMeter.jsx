import React, { useEffect, useRef } from "react";

const SEV_CONFIG = [
  { label: "Minimal",  color: "#10b981", glow: "rgba(16,185,129,0.3)" },
  { label: "Mild",     color: "#84cc16", glow: "rgba(132,204,22,0.3)" },
  { label: "Moderate", color: "#f59e0b", glow: "rgba(245,158,11,0.3)" },
  { label: "High",     color: "#ef4444", glow: "rgba(239,68,68,0.3)"  },
  { label: "Critical", color: "#dc2626", glow: "rgba(220,38,38,0.4)"  },
];

/**
 * SeverityMeter — Animated SVG arc gauge displaying severity level (1-5).
 */
export default function SeverityMeter({ severity = 1, size = 160 }) {
  const canvasRef = useRef(null);

  const clamped = Math.max(1, Math.min(5, Math.round(severity)));
  const cfg = SEV_CONFIG[clamped - 1];
  const percent = (clamped - 1) / 4; // 0 → 1

  // SVG arc parameters
  const cx = size / 2;
  const cy = size / 2 + 10;
  const r = size * 0.38;
  const startAngle = Math.PI * 0.85;
  const endAngle = Math.PI * 2.15;
  const totalAngle = endAngle - startAngle;

  const polarToXY = (angle, radius) => ({
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  });

  const arcPath = (from, to) => {
    const p1 = polarToXY(from, r);
    const p2 = polarToXY(to, r);
    const largeArc = to - from > Math.PI ? 1 : 0;
    return `M ${p1.x} ${p1.y} A ${r} ${r} 0 ${largeArc} 1 ${p2.x} ${p2.y}`;
  };

  const fillAngle = startAngle + totalAngle * percent;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ filter: `drop-shadow(0 0 10px ${cfg.glow})` }}
      >
        <defs>
          <linearGradient id="sev-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#dc2626" />
          </linearGradient>
        </defs>

        {/* Track */}
        <path
          d={arcPath(startAngle, endAngle)}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={size * 0.07}
          strokeLinecap="round"
        />

        {/* Fill */}
        <path
          d={arcPath(startAngle, fillAngle)}
          fill="none"
          stroke={cfg.color}
          strokeWidth={size * 0.07}
          strokeLinecap="round"
          style={{ transition: "all 0.9s cubic-bezier(0.4, 0, 0.2, 1)" }}
        />

        {/* Severity number */}
        <text
          x={cx}
          y={cy + 4}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={size * 0.28}
          fontWeight="800"
          fontFamily="Inter, sans-serif"
          fill={cfg.color}
        >
          {clamped}
        </text>

        {/* "/5" label */}
        <text
          x={cx + size * 0.12}
          y={cy + size * 0.1}
          textAnchor="middle"
          fontSize={size * 0.1}
          fontFamily="Inter, sans-serif"
          fill="rgba(255,255,255,0.35)"
        >
          /5
        </text>
      </svg>

      {/* Status label */}
      <div style={{
        backgroundColor: `${cfg.color}20`,
        border: `1px solid ${cfg.color}50`,
        borderRadius: "var(--radius-full)",
        padding: "0.2rem 0.9rem",
        fontSize: "0.8rem",
        fontWeight: 700,
        color: cfg.color,
        letterSpacing: "0.04em",
        textTransform: "uppercase",
      }}>
        {cfg.label}
      </div>
    </div>
  );
}
