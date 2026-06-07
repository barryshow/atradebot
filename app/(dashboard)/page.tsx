"use client";
import { useEngineSSE } from "@/lib/hooks/use-engine-sse";
import { EngineControl } from "@/app/_components/engine-control";
import { StatsBar } from "@/app/_components/stats-bar";
import { SymbolCard } from "@/app/_components/symbol-card";
import { CandleChart } from "@/app/_components/candle-chart";
import { TradeTable } from "@/app/_components/trade-table";
import { LogStream } from "@/app/_components/log-stream";
import { PnlChart } from "@/app/_components/pnl-chart";
import { RiskGateVisualizer } from "@/app/_components/risk-gate-visualizer";
import { HibtConfigPanel } from "@/app/_components/hibt-config";
import { SYMBOLS } from "@/lib/types/engine";
import type { TradeRecordFlat } from "@/lib/types/candles";

export default function DashboardPage() {
  const { connected, status, symbols, recentTrades, logs } = useEngineSSE();

  // Flatten trades for chart markers
  const tradeMarkers: TradeRecordFlat[] = recentTrades.map((t) => ({
    ts: t.ts,
    symbol: t.symbol,
    direction: t.direction,
    entryPrice: t.entryPrice,
    amount: t.amount,
    result: t.result,
    pnl: t.pnl,
  }));

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Header with engine control */}
      <div className="flex items-center justify-between">
        <EngineControl state={status.state} />
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-400">
            {connected ? "已连接" : "断开连接"}
          </span>
          <HibtConfigPanel />
        </div>
      </div>

      {/* Stats bar */}
      <StatsBar status={status} />

      {/* K-line chart — takes center stage */}
      <CandleChart
        activeSymbol={SYMBOLS[0]}
        trades={tradeMarkers}
      />

      {/* Symbol cards — compact row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {SYMBOLS.map((sym) => (
          <SymbolCard key={sym} snapshot={symbols[sym] || { symbol: sym }} />
        ))}
      </div>

      {/* Risk gate visualizer + PnL chart row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">风控流程</h3>
          <RiskGateVisualizer gates={recentTrades.length > 0 ? recentTrades[recentTrades.length - 1].riskGates : []} />
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <PnlChart trades={recentTrades} />
        </div>
      </div>

      {/* Trade table */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">交易记录</h3>
        <TradeTable trades={recentTrades} />
      </div>

      {/* Log stream */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">实时日志</h3>
        <LogStream logs={logs} />
      </div>
    </div>
  );
}
