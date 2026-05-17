import React, { useState, useRef } from "react";
import { Send, Cpu, RotateCcw, ChevronDown } from "lucide-react";

const MODELS = [
  { value: "ensemble", label: "Ensemble (Recommended)", desc: "Weighted vote across SVM + RF + LSTM" },
  { value: "rf",   label: "Random Forest",            desc: "TF-IDF + LDA combined features" },
  { value: "svm",  label: "SVM",                      desc: "Fast TF-IDF based classification" },
  { value: "lstm", label: "LSTM",                     desc: "Deep learning classifier" },
];

const SAMPLE_TEXTS = [
  "I feel completely hopeless. Nothing matters anymore and I can't see a way forward. I want to end it all.",
  "My anxiety about work is through the roof. I haven't slept in days and my heart races constantly.",
  "I've been cycling between extreme highs and terrible lows. One day I feel invincible, the next I can't move.",
  "The pressure at work is unbearable. Deadlines, meetings, it's all too much and I'm burning out.",
  "I'm having a great day today! The weather is nice and I feel good about things.",
];

/**
 * TextInput — Main analysis input form.
 */
export default function TextInput({ onSubmit, loading, onReset }) {
  const [text, setText] = useState("");
  const [model, setModel] = useState("rf");
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [showSamples, setShowSamples] = useState(false);
  const textareaRef = useRef(null);

  const MAX_CHARS = 2000;
  const MIN_CHARS = 5;
  const charCount = text.length;
  const isValid = charCount >= MIN_CHARS && charCount <= MAX_CHARS;
  const charPct = Math.min(charCount / MAX_CHARS, 1);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isValid || loading) return;
    onSubmit(text.trim(), model);
  };

  const handleSample = (sample) => {
    setText(sample);
    setShowSamples(false);
    textareaRef.current?.focus();
  };

  const selectedModel = MODELS.find((m) => m.value === model);

  return (
    <div className="glass-card" style={{ padding: "1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <h3 style={{ color: "var(--txt-primary)" }}>
          <span style={{ marginRight: "0.5rem" }}>✍️</span> Analyze Text
        </h3>

        {/* Sample texts dropdown */}
        <div style={{ position: "relative" }}>
          <button
            type="button"
            className="btn btn-ghost"
            style={{ fontSize: "0.82rem", padding: "0.4rem 0.85rem" }}
            onClick={() => setShowSamples(!showSamples)}
          >
            Try a sample <ChevronDown size={14} />
          </button>
          {showSamples && (
            <div style={{
              position: "absolute",
              right: 0,
              top: "calc(100% + 8px)",
              background: "var(--clr-bg-elevated)",
              border: "1px solid var(--clr-border)",
              borderRadius: "var(--radius-md)",
              width: 320,
              zIndex: 50,
              overflow: "hidden",
              boxShadow: "var(--shadow-lg)",
            }}>
              {SAMPLE_TEXTS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => handleSample(s)}
                  style={{
                    display: "block",
                    width: "100%",
                    textAlign: "left",
                    padding: "0.65rem 1rem",
                    background: "transparent",
                    border: "none",
                    borderBottom: i < SAMPLE_TEXTS.length - 1 ? "1px solid var(--clr-border)" : "none",
                    color: "var(--txt-secondary)",
                    fontSize: "0.8rem",
                    cursor: "pointer",
                    lineHeight: 1.5,
                    transition: "background var(--transition-fast)",
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.04)"}
                  onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                >
                  {s.slice(0, 75)}…
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Textarea */}
        <div style={{ position: "relative", marginBottom: "1rem" }}>
          <textarea
            ref={textareaRef}
            id="stress-text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Describe how you're feeling, paste a social media post, or write anything on your mind…"
            rows={6}
            style={{
              width: "100%",
              padding: "0.875rem 1rem",
              fontSize: "0.95rem",
              lineHeight: 1.7,
              resize: "vertical",
              minHeight: 140,
            }}
          />

          {/* Char count bar */}
          <div style={{ position: "absolute", bottom: 10, right: 12 }}>
            <span style={{
              fontSize: "0.72rem",
              color: charCount > MAX_CHARS * 0.9 ? "#ef4444" : "var(--txt-muted)",
              fontFamily: "var(--font-mono)",
            }}>
              {charCount}/{MAX_CHARS}
            </span>
          </div>
        </div>

        {/* Char progress bar */}
        <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: "var(--radius-full)", marginBottom: "1rem", overflow: "hidden" }}>
          <div style={{
            height: "100%",
            width: `${charPct * 100}%`,
            background: charCount > MAX_CHARS * 0.9 ? "#ef4444" : "var(--clr-primary)",
            borderRadius: "var(--radius-full)",
            transition: "width 0.15s ease, background 0.2s ease",
          }} />
        </div>

        {/* Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
          {/* Model selector */}
          <div style={{ position: "relative", flex: 1, minWidth: 180 }}>
            <button
              type="button"
              className="btn btn-ghost"
              style={{ width: "100%", justifyContent: "space-between" }}
              onClick={() => setShowModelMenu(!showModelMenu)}
            >
              <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <Cpu size={14} />
                {selectedModel?.label}
              </span>
              <ChevronDown size={14} />
            </button>
            {showModelMenu && (
              <div style={{
                position: "absolute",
                bottom: "calc(100% + 8px)",
                left: 0,
                right: 0,
                background: "var(--clr-bg-elevated)",
                border: "1px solid var(--clr-border)",
                borderRadius: "var(--radius-md)",
                overflow: "hidden",
                zIndex: 50,
                boxShadow: "var(--shadow-lg)",
              }}>
                {MODELS.map((m) => (
                  <button
                    key={m.value}
                    type="button"
                    onClick={() => { setModel(m.value); setShowModelMenu(false); }}
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      padding: "0.65rem 1rem",
                      background: model === m.value ? "rgba(14,165,233,0.12)" : "transparent",
                      border: "none",
                      borderBottom: "1px solid var(--clr-border)",
                      cursor: "pointer",
                      transition: "background var(--transition-fast)",
                    }}
                    onMouseEnter={(e) => { if (model !== m.value) e.currentTarget.style.background = "rgba(255,255,255,0.04)"; }}
                    onMouseLeave={(e) => { if (model !== m.value) e.currentTarget.style.background = "transparent"; }}
                  >
                    <div style={{ fontSize: "0.875rem", fontWeight: 600, color: model === m.value ? "var(--clr-primary)" : "var(--txt-primary)" }}>
                      {m.label}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--txt-muted)" }}>{m.desc}</div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Reset button */}
          {text && (
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => { setText(""); onReset?.(); }}
              style={{ padding: "0.625rem 0.875rem" }}
            >
              <RotateCcw size={14} />
            </button>
          )}

          {/* Submit */}
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!isValid || loading}
            id="analyze-submit-btn"
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16 }} />
                Analyzing…
              </>
            ) : (
              <>
                <Send size={15} />
                Analyze
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
