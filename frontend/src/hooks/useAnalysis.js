import { useState, useCallback } from "react";
import { analyzeText } from "../services/api";

export function useAnalysis() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyze = useCallback(async (text, model = "lstm", threshold = 0.5) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeText(text, model, threshold);
      setResult(data);
      return data;
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, analyze, reset };
}
