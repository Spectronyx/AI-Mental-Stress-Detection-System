import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
  timeout: 60_000,
});

// Response interceptor for unified error handling
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.error ||
      err.response?.data?.detail ||
      err.message ||
      "Unknown error";
    return Promise.reject(new Error(message));
  }
);

export const analyzeText = (text, model = "lstm", threshold = 0.5) =>
  api.post("/analyze", { text, model, threshold }).then((r) => r.data);

export const fetchHistory = (params = {}) =>
  api.get("/history", { params }).then((r) => r.data);

export const fetchHistoryDetail = (id) =>
  api.get(`/history/${id}/`).then((r) => r.data);

export const fetchModelComparison = () =>
  api.get("/compare").then((r) => r.data);

export const fetchStats = () =>
  api.get("/stats").then((r) => r.data);

export const updateThreshold = (threshold) =>
  api.post("/threshold", { threshold }).then((r) => r.data);

export const checkHealth = () =>
  api.get("/health").then((r) => r.data);

export default api;
