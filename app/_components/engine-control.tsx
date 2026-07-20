"use client";
import { useState } from "react";
import { useEngineControl, type RunModeParam } from "@/lib/hooks/use-engine-control";
import type { EngineState } from "@/lib/types/engine";

const stateLabels: Record<EngineState, string> = {
  stopped: "已停止",
  starting: "启动中...",
  running: "运行中",
  paused: "已暂停",
  error: "异常",
};

interface LiveGateStatus {
  passed: boolean;
  reasons: string[];
  checks: Record<string, boolean>;
}

export function EngineControl({
  state,
  runMode,
  liveGate,
  onModeChange,
}: {
  state: EngineState;
  runMode?: string;
  liveGate?: LiveGateStatus;
  onModeChange?: (mode: RunModeParam) => void;
}) {
  const { loading, start, stop, pause, resume } = useEngineControl();
  const [selectedMode, setSelectedMode] = useState<RunModeParam>("shadow");

  const handleStart = () => {
    if (selectedMode === "live") {
      if (!liveGate?.passed) {
        const confirmed = confirm(
          "⚠️ LIVE 门控未全部通过:\n\n" +
            liveGate?.reasons.map((r) => `  ❌ ${r}`).join("\n") +
            "\n\n高风险: 确定要继续启动 LIVE 吗？"
        );
        if (!confirmed) return;
      } else {
        const confirmed = confirm("启动 LIVE 实盘交易？\n\n确认后将自动调用 HIBT 真实下单。");
        if (!confirmed) return;
      }
    }
    start(selectedMode);
    onModeChange?.(selectedMode);
  };

  const handleEmergencyStop = () => {
    stop();
  };

  const isLive = runMode === "LIVE";
  const isRunning = state === "running";

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <span
          className={`inline-block w-2.5 h-2.5 rounded-full ${
            state === "running"
              ? "bg-green-500 animate-pulse"
              : state === "paused"
              ? "bg-yellow-500"
              : state === "error"
              ? "bg-red-500"
              : "bg-gray-500"
          }`}
        />
        <span className="text-sm font-medium">{stateLabels[state]}</span>

        {/* Mode badge */}
        {runMode && (
          <span
            className={`text-xs font-bold px-1.5 py-0.5 rounded ${
              isLive
                ? "bg-red-900/60 text-red-400 animate-pulse"
                : "bg-yellow-900/60 text-yellow-400"
            }`}
          >
            {isLive ? "🔴 LIVE" : "🟡 SHADOW"}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {/* Mode selector */}
        {(state === "stopped" || state === "error") && (
          <select
            value={selectedMode}
            onChange={(e) => setSelectedMode(e.target.value as RunModeParam)}
            className="px-2 py-1.5 text-sm rounded bg-gray-800 border border-gray-700 text-white"
          >
            <option value="shadow">SHADOW (模拟)</option>
            <option value="live">LIVE (实盘)</option>
          </select>
        )}

        {/* Start button */}
        {(state === "stopped" || state === "error") && (
          <button
            onClick={handleStart}
            disabled={loading}
            className={`px-3 py-1.5 text-sm font-medium rounded text-white disabled:opacity-50 ${
              selectedMode === "live"
                ? "bg-red-600 hover:bg-red-500"
                : "bg-green-600 hover:bg-green-500"
            }`}
          >
            {selectedMode === "live" ? "🔴 启动 LIVE" : "启动 SHADOW"}
          </button>
        )}
        {state === "running" && (
          <button
            onClick={pause}
            disabled={loading}
            className="px-3 py-1.5 text-sm font-medium rounded bg-yellow-600 hover:bg-yellow-500 text-white disabled:opacity-50"
          >
            暂停
          </button>
        )}
        {state === "paused" && (
          <button
            onClick={resume}
            disabled={loading}
            className="px-3 py-1.5 text-sm font-medium rounded bg-green-600 hover:bg-green-500 text-white disabled:opacity-50"
          >
            恢复
          </button>
        )}
        {(state === "running" || state === "paused") && (
          <button
            onClick={stop}
            disabled={loading}
            className="px-3 py-1.5 text-sm font-medium rounded bg-red-600 hover:bg-red-500 text-white disabled:opacity-50"
          >
            停止
          </button>
        )}

        {/* Emergency stop — always visible when running */}
        {isRunning && (
          <button
            onClick={handleEmergencyStop}
            className="px-3 py-1.5 text-sm font-bold rounded bg-red-700 hover:bg-red-600 text-white border border-red-500 animate-pulse"
          >
            ⚡ 紧急停止
          </button>
        )}
      </div>

      {/* LIVE Gate status */}
      {isLive && liveGate && !liveGate.passed && (
        <div className="bg-red-900/30 border border-red-800 rounded p-2 mt-1">
          <div className="text-xs text-red-400 font-medium mb-1">LIVE BLOCKED:</div>
          {liveGate.reasons.map((r, i) => (
            <div key={i} className="text-xs text-red-300 ml-2">
              ❌ {r}
            </div>
          ))}
        </div>
      )}
      {isLive && liveGate?.passed && (
        <div className="bg-green-900/30 border border-green-800 rounded p-2 mt-1">
          <div className="text-xs text-green-400 font-medium">✅ LIVE READY</div>
        </div>
      )}
    </div>
  );
}