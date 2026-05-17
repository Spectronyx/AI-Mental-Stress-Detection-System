import React from "react";
import { AlertTriangle, Phone, MessageSquare } from "lucide-react";

/**
 * AlertBanner — Pulsing alert card for severity >= 4.
 */
export default function AlertBanner({ alert }) {
  if (!alert || !alert.alert_triggered) return null;

  const isCritical = alert.severity >= 5;

  return (
    <div
      className="alert-banner glass-card"
      style={{
        padding: "1.25rem 1.5rem",
        border: `1px solid ${isCritical ? "rgba(220,38,38,0.6)" : "rgba(239,68,68,0.4)"}`,
        background: isCritical ? "rgba(220,38,38,0.12)" : "rgba(239,68,68,0.08)",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem", marginBottom: "0.75rem" }}>
        <div style={{
          background: isCritical ? "#dc2626" : "#ef4444",
          borderRadius: "50%",
          width: 36,
          height: 36,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          boxShadow: `0 0 12px ${isCritical ? "rgba(220,38,38,0.5)" : "rgba(239,68,68,0.4)"}`,
        }}>
          <AlertTriangle size={18} color="#fff" />
        </div>
        <div>
          <p style={{ fontWeight: 700, fontSize: "1rem", color: isCritical ? "#fca5a5" : "#f87171", marginBottom: "0.2rem" }}>
            {isCritical ? "🆘 Critical Stress Alert" : "⚠️ Early Warning Alert"}
          </p>
          <p style={{ fontSize: "0.88rem", color: "var(--txt-secondary)", lineHeight: 1.6 }}>
            {alert.alert_message}
          </p>
        </div>
      </div>

      {/* Recommendation */}
      <div style={{
        background: "rgba(255,255,255,0.04)",
        borderRadius: "var(--radius-md)",
        padding: "0.75rem 1rem",
        marginBottom: "0.75rem",
        borderLeft: `3px solid ${isCritical ? "#dc2626" : "#ef4444"}`,
      }}>
        <p style={{ fontSize: "0.85rem", color: "var(--txt-primary)", lineHeight: 1.7 }}>
          {alert.recommendation}
        </p>
      </div>

      {/* Empathetic response */}
      {alert.empathetic_response && (
        <div style={{
          background: "rgba(6,214,160,0.06)",
          borderRadius: "var(--radius-md)",
          padding: "0.75rem 1rem",
          marginBottom: "0.75rem",
          borderLeft: "3px solid var(--clr-accent)",
          fontStyle: "italic",
          fontSize: "0.88rem",
          color: "var(--txt-primary)",
          lineHeight: 1.7,
        }}>
          💬 {alert.empathetic_response}
        </div>
      )}

      {/* Crisis resources */}
      {alert.resources && alert.resources.length > 0 && (
        <div>
          <p style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--txt-muted)", marginBottom: "0.5rem", fontWeight: 600 }}>
            Crisis Resources
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            {alert.resources.map((r, i) => (
              <div key={i} style={{
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
                background: "rgba(255,255,255,0.06)",
                borderRadius: "var(--radius-md)",
                padding: "0.35rem 0.75rem",
                fontSize: "0.78rem",
              }}>
                {r.type === "phone" ? <Phone size={12} color="#f87171" /> : <MessageSquare size={12} color="#f87171" />}
                <span style={{ color: "var(--txt-secondary)" }}>{r.name}:</span>
                <span style={{ color: "#f87171", fontWeight: 600, fontFamily: "var(--font-mono)" }}>{r.contact}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
