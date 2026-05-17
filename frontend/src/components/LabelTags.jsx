import React from "react";

const LABEL_COLORS = {
  // Thematic
  Anxiety:    { bg: "rgba(245,158,11,0.14)", text: "#f59e0b", border: "rgba(245,158,11,0.35)" },
  Depression: { bg: "rgba(139,92,246,0.14)", text: "#a78bfa", border: "rgba(139,92,246,0.35)" },
  Anger:      { bg: "rgba(239,68,68,0.14)",  text: "#f87171", border: "rgba(239,68,68,0.35)"  },
  Sadness:    { bg: "rgba(59,130,246,0.14)", text: "#60a5fa", border: "rgba(59,130,246,0.35)" },
  Neutral:    { bg: "rgba(107,114,128,0.14)",text: "#9ca3af", border: "rgba(107,114,128,0.3)" },
  // Categorical
  Panic:        { bg: "rgba(220,38,38,0.14)",  text: "#f87171", border: "rgba(220,38,38,0.35)" },
  Hopelessness: { bg: "rgba(124,58,237,0.14)", text: "#c4b5fd", border: "rgba(124,58,237,0.35)" },
  Frustration:  { bg: "rgba(234,88,12,0.14)",  text: "#fb923c", border: "rgba(234,88,12,0.35)" },
  Grief:        { bg: "rgba(37,99,235,0.14)",  text: "#93c5fd", border: "rgba(37,99,235,0.35)" },
  Calm:         { bg: "rgba(16,185,129,0.14)", text: "#6ee7b7", border: "rgba(16,185,129,0.35)" },
  Irritability: { bg: "rgba(217,119,6,0.14)",  text: "#fcd34d", border: "rgba(217,119,6,0.35)" },
  Loneliness:   { bg: "rgba(147,51,234,0.14)", text: "#d8b4fe", border: "rgba(147,51,234,0.35)" },
  Worry:        { bg: "rgba(202,138,4,0.14)",  text: "#fde68a", border: "rgba(202,138,4,0.35)" },
  // Triggers
  Work:         { bg: "rgba(8,145,178,0.14)",  text: "#38bdf8", border: "rgba(8,145,178,0.35)" },
  Relationships:{ bg: "rgba(219,39,119,0.14)", text: "#f9a8d4", border: "rgba(219,39,119,0.35)" },
  Financial:    { bg: "rgba(101,163,13,0.14)", text: "#bef264", border: "rgba(101,163,13,0.35)" },
  Academic:     { bg: "rgba(124,58,237,0.14)", text: "#c4b5fd", border: "rgba(124,58,237,0.35)" },
  Health:       { bg: "rgba(5,150,105,0.14)",  text: "#6ee7b7", border: "rgba(5,150,105,0.35)" },
  Social:       { bg: "rgba(217,119,6,0.14)",  text: "#fcd34d", border: "rgba(217,119,6,0.35)" },
  Family:       { bg: "rgba(220,38,38,0.14)",  text: "#fca5a5", border: "rgba(220,38,38,0.35)" },
  "Self-esteem":{ bg: "rgba(147,51,234,0.14)", text: "#e9d5ff", border: "rgba(147,51,234,0.35)" },
};

const DEFAULT_COLOR = { bg: "rgba(99,102,241,0.14)", text: "#a5b4fc", border: "rgba(99,102,241,0.35)" };

/**
 * LabelTags — Renders color-coded label chips for any label category.
 *
 * Props:
 *   labels   {string[]}  — Array of label strings
 *   scores   {object}    — Optional {label: probability} map for score display
 *   title    {string}    — Section heading
 *   compact  {boolean}   — Smaller rendering
 */
export default function LabelTags({ labels = [], scores = {}, title, compact = false }) {
  if (!labels || labels.length === 0) {
    return title ? (
      <div>
        <p style={{ fontSize: "0.75rem", color: "var(--txt-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
          {title}
        </p>
        <span style={{ color: "var(--txt-muted)", fontSize: "0.82rem" }}>None detected</span>
      </div>
    ) : null;
  }

  return (
    <div>
      {title && (
        <p style={{ fontSize: "0.75rem", color: "var(--txt-secondary)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
          {title}
        </p>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
        {labels.map((label) => {
          const clr = LABEL_COLORS[label] || DEFAULT_COLOR;
          const score = scores[label];
          return (
            <span
              key={label}
              className="tag"
              style={{
                background: clr.bg,
                color: clr.text,
                border: `1px solid ${clr.border}`,
                fontSize: compact ? "0.72rem" : "0.78rem",
                padding: compact ? "0.15rem 0.5rem" : "0.2rem 0.65rem",
              }}
            >
              {label}
              {score !== undefined && (
                <span style={{ opacity: 0.75, fontWeight: 400, fontSize: "0.7rem", marginLeft: "0.25rem" }}>
                  {Math.round(score * 100)}%
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}
