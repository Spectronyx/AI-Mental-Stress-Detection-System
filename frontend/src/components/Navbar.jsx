import React from "react";
import { NavLink } from "react-router-dom";
import { Brain, BarChart2, Clock, Github } from "lucide-react";

export default function Navbar() {
  return (
    <>
      {/* Ethical disclaimer bar */}
      <div className="disclaimer-bar">
        ⚠️ Research tool only — Not a medical diagnostic instrument. Always consult a licensed mental health professional.
      </div>

      <nav className="navbar">
        <div className="nav-inner">
          {/* Logo */}
          <div className="nav-logo">
            <div style={{
              width: 34,
              height: 34,
              borderRadius: "var(--radius-md)",
              background: "linear-gradient(135deg, var(--clr-primary), var(--clr-secondary))",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.1rem",
            }}>
              🧠
            </div>
            <span className="gradient-text">MindSense — Suicide & Stress Detection</span>
          </div>

          {/* Nav links */}
          <div className="nav-links">
            <NavLink
              to="/"
              end
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              <Brain size={14} style={{ display: "inline", marginRight: 4 }} />
              Analyze
            </NavLink>
            <NavLink
              to="/history"
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              <Clock size={14} style={{ display: "inline", marginRight: 4 }} />
              History
            </NavLink>
            <NavLink
              to="/compare"
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              <BarChart2 size={14} style={{ display: "inline", marginRight: 4 }} />
              Compare
            </NavLink>

            {/* GitHub link */}
            <a
              href="/api/docs/"
              target="_blank"
              rel="noreferrer"
              className="btn btn-ghost"
              style={{ fontSize: "0.82rem", padding: "0.4rem 0.85rem", marginLeft: "0.25rem" }}
            >
              API Docs
            </a>
          </div>
        </div>
      </nav>
    </>
  );
}
