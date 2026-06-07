"use client";
import type { EngineStatus } from "@/lib/types/engine";

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="text-center">
      <div className="text-xs text-gray-400">{label}</div>
      <div className={`text-sm font-bold font-mono ${color || "text-white"}`}>{value}</div>
    </div>
  );
}

export function StatsBar({ status }: { status: EngineStatus }) {
  const total = status.wins + status.losses;
  const wr = total > 0 ? ((status.wins / total) * 100).toFixed(1) + "%" : "--";
  const profit = status.profit ?? 0;
  const profitTarget = status.bootstrapProfitTarget ?? 12;
  const isKelly = status.betMode === "kelly";

  return (
    <div className="flex items-center justify-between bg-gray-900 border border-gray-800 rounded-lg px-6 py-3">
      <Stat label="余额" value={`${(status.balance ?? 0).toFixed(2)}U`} />
      <div className="w-px h-8 bg-gray-800" />
      <Stat
        label="累计盈亏"
        value={`${profit >= 0 ? "+" : ""}${profit.toFixed(2)}U`}
        color={profit >= 0 ? "text-green-400" : "text-red-400"}
      />
      <div className="w-px h-8 bg-gray-800" />
      <Stat label="总战绩" value={`${status.wins}胜 ${status.losses}负`} />
      <div className="w-px h-8 bg-gray-800" />
      <Stat label="胜率" value={wr} color={total > 0 && status.wins / total >= 0.5 ? "text-green-400" : "text-red-400"} />
      <div className="w-px h-8 bg-gray-800" />
      <Stat label="持仓" value={`${status.activeTrades}/${status.maxConcurrentTrades}`} />
      <div className="w-px h-8 bg-gray-800" />
      <Stat
        label={isKelly ? "凯利公式" : "3U定投"}
        value={isKelly ? "✔️已激活" : `${profit >= profitTarget ? "✔️达标" : `${profit.toFixed(0)}/${profitTarget}U`}`}
        color={isKelly ? "text-yellow-400" : profit >= profitTarget ? "text-green-400" : "text-blue-400"}
      />
      <div className="w-px h-8 bg-gray-800" />
      <Stat label="运行时间" value={formatUptime(status.uptime)} />
    </div>
  );
}

function formatUptime(sec: number): string {
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `${h}h ${m}m`;
}
