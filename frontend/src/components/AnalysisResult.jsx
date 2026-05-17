import React from "react";
import SeverityMeter from "./SeverityMeter";
import LabelTags from "./LabelTags";
import AlertBanner from "./AlertBanner";
import { Zap, Clock, Brain } from "lucide-react";

/**
 * AnalysisResult — Full analysis output card.
 */
export default function AnalysisResult({ result }) {
  if (!result) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }} className="fade-in-up">

      {/* Alert banner (if severity >= 4) */}
      <AlertBanner alert={result.alert} />

      {/* Main result card */}
      <div className="glass-card" style={{ padding: "1.5rem" }}>
        {/* Header row */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", marginBottom: "1.5rem" }}>
          <div style={{ flex: 1 }}>
            <h3 style={{ marginBottom: "0.35rem" }}>Analysis Result</h3>

            <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
              {/* Model badge */}
              <span style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.78rem", color: "var(--txt-muted)" }}>
                <Brain size={12} />
                Model: <strong style={{ color: "var(--clr-primary)" }}>{result.model_used?.toUpperCase()}</strong>
              </span>
              {/* Processing time */}
              <span style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.78rem", color: "var(--txt-muted)" }}>
                <Clock size={12} />
                {result.processing_time_ms}ms
              </span>
              {/* Confidence */}
              <span style={{ display: "flex", alignItems: "center", gap: "0.3rem", fontSize: "0.78rem", color: "var(--txt-muted)" }}>
                <Zap size={12} />
                Confidence: <strong style={{ color: "var(--clr-accent)" }}>{Math.round((result.confidence_score || 0) * 100)}%</strong>
              </span>
            </div>
          </div>

          {/* Severity gauge */}
          <SeverityMeter severity={result.severity} size={140} />
        </div>

        {/* Confidence bar */}
        <div style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--txt-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>Confidence</span>
            <span style={{ fontSize: "0.78rem", color: "var(--clr-accent)", fontWeight: 600 }}>{Math.round((result.confidence_score || 0) * 100)}%</span>
          </div>
          <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: "var(--radius-full)", overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: `${(result.confidence_score || 0) * 100}%`,
              background: "linear-gradient(90deg, var(--clr-primary), var(--clr-accent))",
              borderRadius: "var(--radius-full)",
              transition: "width 0.9s cubic-bezier(0.4,0,0.2,1)",
            }} />
          </div>
        </div>

        {/* Labels grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1.25rem" }}>
          <LabelTags
            title="Emotional State"
            labels={result.thematic_labels}
            scores={result.thematic_scores}
          />
          <LabelTags
            title="Specific Feelings"
            labels={result.categorical_labels}
            scores={result.categorical_scores}
          />
          <LabelTags
            title="Stress Triggers"
            labels={result.trigger_labels}
            scores={result.trigger_scores}
          />
        </div>

        {/* Thematic commentary from alert engine */}
        {result.alert?.thematic_commentary && (
          <div style={{
            marginTop: "1.25rem",
            padding: "0.875rem 1rem",
            background: "rgba(14,165,233,0.06)",
            borderRadius: "var(--radius-md)",
            borderLeft: "3px solid var(--clr-primary)",
          }}>
            <p style={{ fontSize: "0.875rem", color: "var(--txt-secondary)", lineHeight: 1.7 }}>
              💡 {result.alert.thematic_commentary}
            </p>
          </div>
        )}
      </div>

      {/* Severity probability breakdown */}
      {result.severity_probabilities?.length > 0 && (
        <div className="glass-card" style={{ padding: "1.25rem 1.5rem" }}>
          <h4 style={{ marginBottom: "1rem", color: "var(--txt-secondary)", fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Severity Probability Breakdown
          </h4>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {["Minimal", "Mild", "Moderate", "High", "Critical"].map((label, i) => {
              const prob = result.severity_probabilities[i] || 0;
              const colors = ["#10b981", "#84cc16", "#f59e0b", "#ef4444", "#dc2626"];
              return (
                <div key={label} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                  <span style={{ fontSize: "0.78rem", color: "var(--txt-muted)", width: 64, flexShrink: 0 }}>{label}</span>
                  <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.06)", borderRadius: "var(--radius-full)", overflow: "hidden" }}>
                    <div style={{
                      height: "100%",
                      width: `${prob * 100}%`,
                      background: colors[i],
                      borderRadius: "var(--radius-full)",
                      transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
                    }} />
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "var(--txt-muted)", fontFamily: "var(--font-mono)", width: 38, textAlign: "right" }}>
                    {Math.round(prob * 100)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div style={{
        padding: "0.65rem 1rem",
        background: "rgba(6,214,160,0.04)",
        border: "1px solid rgba(6,214,160,0.15)",
        borderRadius: "var(--radius-md)",
        fontSize: "0.75rem",
        color: "var(--txt-muted)",
        lineHeight: 1.6,
      }}>
        {result.disclaimer}
      </div>
    </div>
  );
}
