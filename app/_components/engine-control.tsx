"use client";
import { useEngineControl } from "@/lib/hooks/use-engine-control";
import type { EngineState } from "@/lib/types/engine";

const stateLabels: Record<EngineState, string> = {
  stopped: "已停止",
  starting: "启动中...",
  running: "运行中",
  paused: "已暂停",
  error: "异常",
};

export function EngineControl({ state }: { state: EngineState }) {
  const { loading, start, stop, pause, resume } = useEngineControl();

  return (
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

      <div className="flex gap-2 ml-4">
        {(state === "stopped" || state === "error") && (
          <button
            onClick={start}
            disabled={loading}
            className="px-3 py-1.5 text-sm font-medium rounded bg-green-600 hover:bg-green-500 text-white disabled:opacity-50"
          >
            启动
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
      </div>
    </div>
  );
}
