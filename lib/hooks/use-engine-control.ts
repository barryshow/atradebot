"use client";
import { useState, useCallback } from "react";

export type RunModeParam = "live" | "shadow" | "backtest";

export function useEngineControl() {
  const [loading, setLoading] = useState(false);

  const sendAction = useCallback(async (action: string, mode?: RunModeParam) => {
    setLoading(true);
    try {
      const body: Record<string, unknown> = { action };
      if (mode) body.mode = mode;
      const res = await fetch("/api/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
    start: (mode?: RunModeParam) => sendAction("start", mode),
    stop: () => sendAction("stop"),
    pause: () => sendAction("pause"),
    resume: () => sendAction("resume"),
    emergencyStop: () => sendAction("stop", "live"), // force stop regardless of mode
  };
}
