import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Compare from "./pages/Compare";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/compare" element={<Compare />} />
        {/* 404 fallback */}
        <Route path="*" element={
          <div className="page" style={{ textAlign: "center" }}>
            <div className="container">
              <h2 className="gradient-text" style={{ fontSize: "4rem", marginBottom: "1rem" }}>404</h2>
              <p>Page not found.</p>
              <a href="/" className="btn btn-primary" style={{ marginTop: "1.5rem", display: "inline-flex" }}>Go Home</a>
            </div>
          </div>
        } />
      </Routes>
    </BrowserRouter>
  );
}
