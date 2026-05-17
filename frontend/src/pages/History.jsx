import React, { useEffect, useState } from "react";
import { fetchHistory } from "../services/api";
import LabelTags from "../components/LabelTags";
import SeverityMeter from "../components/SeverityMeter";
import TrendChart from "../components/TrendChart";
import { AlertTriangle, Search, SlidersHorizontal, ChevronLeft, ChevronRight } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

const SEV_BADGE = {
  1: { bg: "rgba(16,185,129,0.14)", text: "#6ee7b7", label: "Minimal" },
  2: { bg: "rgba(132,204,22,0.14)", text: "#bef264", label: "Mild" },
  3: { bg: "rgba(245,158,11,0.14)", text: "#fcd34d", label: "Moderate" },
  4: { bg: "rgba(239,68,68,0.14)", text: "#f87171", label: "High" },
  5: { bg: "rgba(220,38,38,0.14)", text: "#fca5a5", label: "Critical" },
};

export default function History() {
  const [records, setRecords] = useState([]);
  const [count, setCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [filterModel, setFilterModel] = useState("");
  const [filterAlert, setFilterAlert] = useState("");
  const [expanded, setExpanded] = useState(null);
  const PAGE_SIZE = 10;

  const load = () => {
    setLoading(true);
    const params = { page, page_size: PAGE_SIZE };
    if (filterModel) params.model_used = filterModel;
    if (filterAlert !== "") params.alert_triggered = filterAlert;
    if (search) params.search = search;

    fetchHistory(params)
      .then((data) => {
        setRecords(data.results || data);
        setCount(data.count || (data.results || data).length);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [page, filterModel, filterAlert]);

  const totalPages = Math.ceil(count / PAGE_SIZE);

  // Build trend data from records
  const trendData = [...records]
    .reverse()
    .map((r) => ({ timestamp: r.created_at, severity: r.severity }));

  return (
    <div className="page">
      <div className="container">
        <div style={{ marginBottom: "var(--sp-6)" }}>
          <h1 style={{ marginBottom: "0.4rem" }}><span className="gradient-text">Analysis</span> History</h1>
          <p>{count} analyses recorded</p>
        </div>

        {/* Trend chart */}
        {trendData.length > 0 && (
          <div style={{ marginBottom: "var(--sp-6)" }}>
            <TrendChart data={trendData} title="Severity Trend (Last 20)" />
          </div>
        )}

        {/* Filters */}
        <div className="glass-card" style={{ padding: "1rem 1.25rem", marginBottom: "var(--sp-4)", display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
          <SlidersHorizontal size={16} color="var(--txt-muted)" />

          <div style={{ position: "relative", flex: 1, minWidth: 180 }}>
            <Search size={14} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--txt-muted)" }} />
            <input
              type="text"
              placeholder="Search text…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && load()}
              style={{ width: "100%", paddingLeft: "2.25rem", paddingRight: "1rem", paddingBlock: "0.5rem", fontSize: "0.875rem" }}
            />
          </div>

          <select
            value={filterModel}
            onChange={(e) => { setFilterModel(e.target.value); setPage(1); }}
            style={{ padding: "0.5rem 0.875rem", fontSize: "0.875rem", minWidth: 140 }}
          >
            <option value="">All Models</option>
            <option value="lstm">LSTM</option>
            <option value="rf">Random Forest</option>
            <option value="svm">SVM</option>
          </select>

          <select
            value={filterAlert}
            onChange={(e) => { setFilterAlert(e.target.value); setPage(1); }}
            style={{ padding: "0.5rem 0.875rem", fontSize: "0.875rem", minWidth: 140 }}
          >
            <option value="">All Records</option>
            <option value="true">Alerts Only</option>
            <option value="false">No Alert</option>
          </select>

          <button className="btn btn-ghost" onClick={load} style={{ padding: "0.5rem 1rem", fontSize: "0.875rem" }}>
            Search
          </button>
        </div>

        {/* Error state */}
        {error && (
          <div style={{ padding: "1rem", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "var(--radius-md)", color: "#f87171", marginBottom: "1rem" }}>
            {error}
          </div>
        )}

        {/* Records list */}
        {loading ? (
          <div style={{ textAlign: "center", padding: "3rem", color: "var(--txt-muted)" }}>
            <span className="spinner" style={{ margin: "0 auto 1rem", display: "block", width: 24, height: 24 }} />
            Loading records…
          </div>
        ) : records.length === 0 ? (
          <div className="glass-card" style={{ padding: "3rem", textAlign: "center" }}>
            <p style={{ color: "var(--txt-muted)" }}>No records found. Submit some analyses on the dashboard!</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {records.map((record) => {
              const sev = SEV_BADGE[record.severity] || SEV_BADGE[1];
              const isOpen = expanded === record.id;
              return (
                <div
                  key={record.id}
                  className="glass-card"
                  style={{ overflow: "hidden", cursor: "pointer" }}
                  onClick={() => setExpanded(isOpen ? null : record.id)}
                >
                  {/* Row preview */}
                  <div style={{ padding: "1rem 1.25rem", display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
                    {/* Severity badge */}
                    <span style={{
                      background: sev.bg,
                      color: sev.text,
                      border: `1px solid ${sev.text}40`,
                      borderRadius: "var(--radius-full)",
                      padding: "0.2rem 0.75rem",
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      flexShrink: 0,
                    }}>
                      {sev.label}
                    </span>

                    {/* Alert icon */}
                    {record.alert_triggered && (
                      <AlertTriangle size={14} color="#ef4444" style={{ flexShrink: 0 }} />
                    )}

                    {/* Text preview */}
                    <span style={{ flex: 1, color: "var(--txt-secondary)", fontSize: "0.875rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {record.text_preview}
                    </span>

                    <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexShrink: 0 }}>
                      {/* Model badge */}
                      <span style={{ background: "rgba(14,165,233,0.1)", border: "1px solid rgba(14,165,233,0.3)", borderRadius: "var(--radius-full)", padding: "0.15rem 0.6rem", fontSize: "0.72rem", fontWeight: 600, color: "var(--clr-primary)" }}>
                        {record.model_used?.toUpperCase()}
                      </span>
                      {/* Time */}
                      <span style={{ fontSize: "0.75rem", color: "var(--txt-muted)" }}>
                        {formatDistanceToNow(new Date(record.created_at), { addSuffix: true })}
                      </span>
                    </div>
                  </div>

                  {/* Expanded detail */}
                  {isOpen && (
                    <div style={{ borderTop: "1px solid var(--clr-border)", padding: "1rem 1.25rem", display: "flex", flexDirection: "column", gap: "0.875rem" }} onClick={(e) => e.stopPropagation()}>
                      <div style={{ display: "flex", gap: "1.5rem", alignItems: "center", flexWrap: "wrap" }}>
                        <SeverityMeter severity={record.severity} size={100} />
                        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                          <LabelTags title="Emotional State" labels={record.thematic_labels} scores={record.thematic_scores} compact />
                          <LabelTags title="Triggers" labels={record.trigger_labels} scores={record.trigger_scores} compact />
                        </div>
                      </div>
                      {record.alert_log && (
                        <div style={{ padding: "0.75rem 1rem", background: "rgba(239,68,68,0.06)", borderRadius: "var(--radius-md)", borderLeft: "3px solid #ef4444", fontSize: "0.85rem", color: "var(--txt-secondary)" }}>
                          ⚠️ {record.alert_log.recommendation}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "0.75rem", marginTop: "1.5rem" }}>
            <button className="btn btn-ghost" disabled={page === 1} onClick={() => setPage(p => p - 1)} style={{ padding: "0.5rem 0.75rem" }}>
              <ChevronLeft size={16} />
            </button>
            <span style={{ color: "var(--txt-secondary)", fontSize: "0.875rem" }}>Page {page} of {totalPages}</span>
            <button className="btn btn-ghost" disabled={page === totalPages} onClick={() => setPage(p => p + 1)} style={{ padding: "0.5rem 0.75rem" }}>
              <ChevronRight size={16} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
