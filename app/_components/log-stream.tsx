"use client";
import { useEffect, useRef, useState } from "react";

export function LogStream({ logs }: { logs: string[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  if (logs.length === 0) {
    return (
      <div className="flex items-center justify-center h-24 text-gray-500 text-sm">
        等待日志输出...
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-48 overflow-y-auto font-mono text-xs space-y-0.5 bg-black/30 rounded p-2"
      onMouseEnter={() => setAutoScroll(false)}
      onMouseLeave={() => setAutoScroll(true)}
    >
      {logs.slice(-150).map((line, i) => {
        let color = "text-gray-400";
        if (line.includes("✅") || line.includes("盈利")) color = "text-green-400";
        else if (line.includes("❌") || line.includes("亏损") || line.includes("驳回")) color = "text-red-400";
        else if (line.includes("⚠️") || line.includes("翻转")) color = "text-yellow-400";
        else if (line.includes("🚀") || line.includes("实盘")) color = "text-blue-400";
        return (
          <div key={i} className={color}>
            {line}
          </div>
        );
      })}
    </div>
  );
}
