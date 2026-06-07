"use client";
import { useState, useCallback } from "react";

export function useEngineControl() {
  const [loading, setLoading] = useState(false);

  const sendAction = useCallback(async (action: string) => {
    setLoading(true);
    try {
      const res = await fetch("/api/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      return await res.json();
    } catch (err) {
      return { ok: false, error: String(err) };
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    start: () => sendAction("start"),
    stop: () => sendAction("stop"),
    pause: () => sendAction("pause"),
    resume: () => sendAction("resume"),
  };
}
